#!/usr/bin/env python3
"""
hremc_engine.py — Hamiltonian Replica Exchange Monte Carlo Engine
=================================================================
SCN5A Nav1.5 (WT & RxxxH mutant) — Martini 3 CG with Go-model

Workflow per cycle:
  1. Run short MD segments for all 24 replicas (parallelizable)
  2. Extract final-frame coordinates from each replica
  3. For each adjacent pair (i, i+1):
     a. Rerun config_i with Hamiltonian_j → U_j(x_i)
     b. Rerun config_j with Hamiltonian_i → U_i(x_j)
     c. Compute ΔΔU = [U_j(x_i) - U_i(x_i)] + [U_i(x_j) - U_j(x_j)]
     d. Accept swap with P = min(1, exp(-β·ΔΔU))
  4. If accepted, swap coordinate files between replicas i and j
  5. Log acceptance rates, energies, replica trajectories

Usage:
    python hremc_engine.py --config config.yaml --system wt --equilibrate
    python hremc_engine.py --config config.yaml --system wt --production
    python hremc_engine.py --config config.yaml --system both --production
"""

import argparse
import json
import logging
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# ============================================================
# Fix OpenMP thread contention on Apple Silicon (and generally
# when running multiple GROMACS instances in parallel).
# Without this, mdrun can deadlock in busy-wait spin loops.
# ============================================================
os.environ.setdefault("OMP_WAIT_POLICY", "passive")
os.environ.setdefault("GMX_THREAD_PINNING", "off")

# ============================================================
# Constants
# ============================================================
KB = 8.314462618e-3  # kJ/(mol·K) — Boltzmann constant in GROMACS units


# ============================================================
# Data classes
# ============================================================
@dataclass
class Replica:
    """State of one HREMC replica."""
    index: int
    lam: float
    rep_dir: Path
    current_gro: str = "conf.gro"
    current_edr: str = ""
    potential_energy: float = 0.0
    cycle: int = 0
    # Track which lambda-state this replica is sampling
    # (differs from index after swaps)
    state_index: int = 0

    def __post_init__(self):
        self.state_index = self.index


@dataclass
class ExchangeRecord:
    """Record of one attempted exchange."""
    cycle: int
    pair: tuple
    delta_energy: float
    probability: float
    accepted: bool


@dataclass
class HREMCState:
    """Full state of the HREMC simulation."""
    system: str
    n_replicas: int
    lambdas: list
    temperature: float
    cycle: int = 0
    total_exchanges_attempted: int = 0
    total_exchanges_accepted: int = 0
    pair_attempts: dict = field(default_factory=dict)
    pair_accepts: dict = field(default_factory=dict)
    # permutation[i] = which state index is currently in replica slot i
    permutation: list = field(default_factory=list)
    exchange_log: list = field(default_factory=list)

    def acceptance_rate(self, pair: Optional[tuple] = None) -> float:
        if pair is not None:
            key = f"{pair[0]}-{pair[1]}"
            att = self.pair_attempts.get(key, 0)
            acc = self.pair_accepts.get(key, 0)
            return acc / att if att > 0 else 0.0
        if self.total_exchanges_attempted == 0:
            return 0.0
        return self.total_exchanges_accepted / self.total_exchanges_attempted


# ============================================================
# GROMACS Interface
# ============================================================
class GMXRunner:
    """Wrapper for GROMACS command execution."""

    def __init__(self, gmx: str = "gmx", maxwarn: int = 2,
                 n_threads: int = 2):
        self.gmx = gmx
        self.maxwarn = maxwarn
        self.n_threads = n_threads
        self.logger = logging.getLogger("GMXRunner")

    def run_cmd(self, cmd: list[str], workdir: Path,
                stdin_text: str = "", timeout: int = 3600) -> subprocess.CompletedProcess:
        """Run a shell command in a working directory."""
        self.logger.debug(f"CMD: {' '.join(cmd)}")
        self.logger.debug(f"CWD: {workdir}")
        try:
            result = subprocess.run(
                cmd,
                cwd=str(workdir),
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                self.logger.error(f"GROMACS error in {workdir}:\n{result.stderr[-2000:]}")
            return result
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout ({timeout}s) for: {' '.join(cmd)}")
            raise

    def grompp(self, mdp: str, gro: str, top: str, tpr: str,
               workdir: Path, ndx: str = None, maxwarn: int = None) -> bool:
        """Run gmx grompp."""
        mw = maxwarn if maxwarn is not None else self.maxwarn
        cmd = [self.gmx, "grompp",
               "-f", mdp, "-c", gro, "-p", top, "-o", tpr,
               "-maxwarn", str(mw)]
        if ndx and (workdir / ndx).exists():
            cmd.extend(["-n", ndx])
        result = self.run_cmd(cmd, workdir)
        return result.returncode == 0

    def mdrun(self, tpr: str, workdir: Path, deffnm: str = None,
              rerun_trr: str = None, n_threads: int = None) -> bool:
        """Run gmx mdrun."""
        nt = n_threads or self.n_threads
        cmd = [self.gmx, "mdrun", "-s", tpr, "-nt", str(nt), "-pin", "off"]
        if deffnm:
            cmd.extend(["-deffnm", deffnm])
        if rerun_trr:
            cmd.extend(["-rerun", rerun_trr])
        result = self.run_cmd(cmd, workdir, timeout=7200)
        return result.returncode == 0

    def energy(self, edr: str, workdir: Path,
               terms: str = "Potential") -> dict[str, float]:
        """Extract energy terms from an .edr file using gmx energy."""
        cmd = [self.gmx, "energy", "-f", edr]
        # Send the term selection to stdin
        result = self.run_cmd(cmd, workdir, stdin_text=terms + "\n\n")
        if result.returncode != 0:
            return {}
        # Parse output
        energies = {}
        for line in result.stdout.split("\n"):
            line = line.strip()
            # Look for lines like: "Potential                  -1.23456e+06   ..."
            if any(t in line for t in terms.split("\n")):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        energies[parts[0]] = float(parts[1])
                    except (ValueError, IndexError):
                        pass
        return energies

    def extract_last_frame(self, traj: str, tpr: str, output_gro: str,
                           workdir: Path) -> bool:
        """Extract the last frame from a trajectory."""
        # Use trjconv with -dump to get the last frame
        cmd = [self.gmx, "trjconv",
               "-f", traj, "-s", tpr, "-o", output_gro, "-dump", "999999"]
        result = self.run_cmd(cmd, workdir, stdin_text="0\n")
        return result.returncode == 0

    def make_ndx(self, gro: str, ndx: str, workdir: Path) -> bool:
        """Generate a default index file."""
        cmd = [self.gmx, "make_ndx", "-f", gro, "-o", ndx]
        result = self.run_cmd(cmd, workdir, stdin_text="q\n")
        return result.returncode == 0


# ============================================================
# Standalone function for parallel execution (must be picklable)
# ============================================================
def _run_single_replica_segment(
    rep_index: int, rep_dir: Path, cycle: int,
    segment_mdp: str, gmx_exe: str, maxwarn: int,
    n_threads: int, keep_all: bool, ckpt_interval: int,
    gpu_id: int = -1,
) -> dict:
    """Run grompp + mdrun for one replica segment.

    This is a module-level function so ProcessPoolExecutor can pickle it.
    Returns a dict with 'success', 'edr', and optionally 'error'.

    Parameters
    ----------
    gpu_id : int
        GPU device ID to use. -1 = CPU only.
    """
    rep_dir = Path(rep_dir)  # ensure Path after unpickling

    # Determine input coordinates
    if cycle == 0:
        input_gro = "state.gro"
        if not (rep_dir / input_gro).exists():
            input_gro = "npt.gro"
        if not (rep_dir / input_gro).exists():
            input_gro = "conf.gro"
    else:
        input_gro = "state.gro"

    ndx_file = "index.ndx" if (rep_dir / "index.ndx").exists() else None
    seg_prefix = f"seg_{cycle:06d}"

    # --- grompp ---
    cmd_grompp = [gmx_exe, "grompp",
                  "-f", segment_mdp, "-c", input_gro,
                  "-p", "topol.top", "-o", f"{seg_prefix}.tpr",
                  "-maxwarn", str(maxwarn)]
    if ndx_file:
        cmd_grompp.extend(["-n", ndx_file])

    result = subprocess.run(cmd_grompp, cwd=str(rep_dir),
                            capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return {"success": False, "edr": "",
                "error": f"grompp: {result.stderr[-500:]}"}

    # --- mdrun ---
    cmd_mdrun = [gmx_exe, "mdrun",
                 "-s", f"{seg_prefix}.tpr", "-deffnm", seg_prefix,
                 "-nt", str(n_threads), "-pin", "off"]
    if gpu_id >= 0:
        cmd_mdrun.extend(["-nb", "gpu", "-gpu_id", str(gpu_id)])

    result = subprocess.run(cmd_mdrun, cwd=str(rep_dir),
                            capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        return {"success": False, "edr": "",
                "error": f"mdrun: {result.stderr[-500:]}"}

    # --- Update state ---
    seg_gro = rep_dir / f"{seg_prefix}.gro"
    if seg_gro.exists():
        shutil.copy2(seg_gro, rep_dir / "state.gro")

    # --- Cleanup old segment files ---
    if not keep_all and cycle > 0:
        prev_prefix = f"seg_{cycle-1:06d}"
        for ext in [".trr", ".xtc", ".tpr", ".log", ".edr"]:
            old_file = rep_dir / f"{prev_prefix}{ext}"
            if old_file.exists() and (cycle - 1) % ckpt_interval != 0:
                old_file.unlink()

    return {"success": True, "edr": f"{seg_prefix}.edr"}


def _equilibrate_single_replica(
    rep_index: int, rep_dir: Path,
    em_mdp: str, nvt_mdp: str, npt_mdp: str,
    gmx_exe: str, maxwarn: int, n_threads: int,
    gpu_id: int = -1, max_retries: int = 3,
) -> bool:
    """Run EM → NVT → NPT for one replica. Module-level for pickling.

    Retries up to max_retries times if NVT/NPT blows up (NaN energies).
    Each retry re-runs from EM with the original conf.gro.
    """
    rep_dir = Path(rep_dir)
    ndx_file = "index.ndx" if (rep_dir / "index.ndx").exists() else None

    def _grompp(mdp, gro, tpr, ref_gro=None):
        cmd = [gmx_exe, "grompp", "-f", mdp, "-c", gro,
               "-p", "topol.top", "-o", tpr, "-maxwarn", str(maxwarn)]
        if ref_gro:
            cmd.extend(["-r", ref_gro])
        if ndx_file:
            cmd.extend(["-n", ndx_file])
        return subprocess.run(cmd, cwd=str(rep_dir),
                              capture_output=True, text=True, timeout=300)

    def _mdrun(tpr, deffnm):
        cmd = [gmx_exe, "mdrun", "-s", tpr, "-deffnm", deffnm,
               "-nt", str(n_threads), "-pin", "off"]
        if gpu_id >= 0:
            cmd.extend(["-nb", "gpu", "-gpu_id", str(gpu_id)])
        return subprocess.run(cmd, cwd=str(rep_dir),
                              capture_output=True, text=True, timeout=7200)

    def _log_failure(step, result, attempt=0):
        err_file = rep_dir / "eq_error.log"
        with open(err_file, "a") as f:
            f.write(f"=== {step} FAILED (rc={result.returncode}, attempt={attempt}) ===\n")
            f.write(f"--- STDOUT ---\n{result.stdout[-2000:] if result.stdout else ''}\n")
            f.write(f"--- STDERR ---\n{result.stderr[-2000:] if result.stderr else ''}\n\n")

    def _clean_artifacts():
        """Remove intermediate files so EM can start fresh."""
        for pattern in ["em.*", "nvt.*", "npt.*", "state.gro", "mdout.mdp",
                        "#*#", "step*.pdb"]:
            import glob
            for f in glob.glob(str(rep_dir / pattern)):
                try:
                    os.remove(f)
                except OSError:
                    pass

    for attempt in range(max_retries):
        if attempt > 0:
            _clean_artifacts()
            # Log retry
            with open(rep_dir / "eq_error.log", "a") as f:
                f.write(f"\n=== RETRY {attempt}/{max_retries} ===\n\n")

        # EM
        r = _grompp(em_mdp, "conf.gro", "em.tpr")
        if r.returncode != 0:
            _log_failure("grompp-EM", r, attempt)
            return False  # grompp fail is not retryable
        r = _mdrun("em.tpr", "em")
        if r.returncode != 0:
            _log_failure("mdrun-EM", r, attempt)
            continue  # retry EM

        # NVT (use em.gro as position restraint reference)
        r = _grompp(nvt_mdp, "em.gro", "nvt.tpr", ref_gro="em.gro")
        if r.returncode != 0:
            _log_failure("grompp-NVT", r, attempt)
            return False
        r = _mdrun("nvt.tpr", "nvt")
        if r.returncode != 0:
            _log_failure("mdrun-NVT", r, attempt)
            continue  # retry from EM

        # NPT (use em.gro as position restraint reference)
        r = _grompp(npt_mdp, "nvt.gro", "npt.tpr", ref_gro="em.gro")
        if r.returncode != 0:
            _log_failure("grompp-NPT", r, attempt)
            return False
        r = _mdrun("npt.tpr", "npt")
        if r.returncode != 0:
            _log_failure("mdrun-NPT", r, attempt)
            continue  # retry from EM

        # Copy final state
        shutil.copy2(rep_dir / "npt.gro", rep_dir / "state.gro")
        return True

    # All retries exhausted
    return False


# ============================================================
# HREMC Engine
# ============================================================
class HREMCEngine:
    """Hamiltonian Replica Exchange Monte Carlo engine."""

    def __init__(self, config: dict, system: str, mc_root: Path):
        self.config = config
        self.system = system
        self.mc_root = mc_root
        self.sys_dir = mc_root / system

        # Setup logging
        self.logger = logging.getLogger("HREMC")
        self._setup_logging()

        # GROMACS runner
        gmx_cfg = config["gromacs"]
        self.gmx = GMXRunner(
            gmx=gmx_cfg["gmx"],
            maxwarn=gmx_cfg["maxwarn"],
            n_threads=gmx_cfg["n_threads_per_replica"],
        )

        # Lambda schedule
        self.n_replicas = config["replicas"]["n_replicas"]
        self.lambdas = self._read_lambda_schedule()
        self.temperature = config["system"]["temperature"]
        self.beta = 1.0 / (KB * self.temperature)

        # Initialize replicas
        self.replicas = []
        for i in range(self.n_replicas):
            rep_dir = self.sys_dir / f"replica_{i:02d}"
            self.replicas.append(Replica(index=i, lam=self.lambdas[i],
                                         rep_dir=rep_dir))

        # State tracking
        self.state = HREMCState(
            system=system,
            n_replicas=self.n_replicas,
            lambdas=self.lambdas,
            temperature=self.temperature,
            permutation=list(range(self.n_replicas)),
        )

        # MDP paths
        self.mdp_dir = mc_root / config["paths"]["mdp_dir"]
        self.segment_mdp = self.mdp_dir / "hremc_segment.mdp"
        self.rerun_mdp = self.mdp_dir / "rerun.mdp"
        self.em_mdp = self.mdp_dir / "em.mdp"
        self.nvt_mdp = self.mdp_dir / "nvt_eq.mdp"
        self.npt_mdp = self.mdp_dir / "npt_eq.mdp"

        # GPU configuration
        gpu_cfg = config.get("gpu", {})
        self.gpu_enabled = gpu_cfg.get("enabled", False)
        if self.gpu_enabled:
            gpu_ids_str = str(gpu_cfg.get("ids", "0"))
            self.gpu_ids = [int(x) for x in gpu_ids_str.split(",")]
            self.n_gpus = len(self.gpu_ids)
            self.logger.info(f"GPU enabled: {self.n_gpus} GPUs ({self.gpu_ids})")
        else:
            self.gpu_ids = []
            self.n_gpus = 0

    def _get_gpu_id(self, rep_index: int) -> int:
        """Return the GPU ID for a given replica (round-robin assignment).
        Returns -1 if GPU is not enabled.
        """
        if not self.gpu_enabled or self.n_gpus == 0:
            return -1
        return self.gpu_ids[rep_index % self.n_gpus]

    def _setup_logging(self):
        log_dir = self.sys_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        handler = logging.FileHandler(log_dir / "hremc.log")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

        # Also log to stdout
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        self.logger.addHandler(console)

    def _read_lambda_schedule(self) -> list[float]:
        """Read lambda values from the schedule file."""
        sched_file = self.sys_dir / "lambda_schedule.dat"
        lambdas = []
        with open(sched_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    lambdas.append(float(parts[1]))
        return lambdas

    # --------------------------------------------------------
    # Index file generation
    # --------------------------------------------------------
    def generate_index_files(self):
        """Generate GROMACS index files for all replicas."""
        self.logger.info("Generating index files for all replicas...")
        for rep in self.replicas:
            if not (rep.rep_dir / "index.ndx").exists():
                self.gmx.make_ndx("conf.gro", "index.ndx", rep.rep_dir)

    # --------------------------------------------------------
    # Equilibration
    # --------------------------------------------------------
    def equilibrate(self):
        """Run energy minimization + NVT + NPT equilibration for all replicas."""
        self.logger.info(f"Starting equilibration for {self.system} "
                         f"({self.n_replicas} replicas)")

        # Generate index files first
        self.generate_index_files()

        parallel_cfg = self.config.get("parallel", {})
        use_parallel = parallel_cfg.get("enabled", False)
        max_workers = parallel_cfg.get("max_workers", self.n_replicas)

        if use_parallel and max_workers > 1:
            self._equilibrate_parallel(max_workers)
        else:
            self._equilibrate_sequential()

        self.logger.info("Equilibration complete.")

    def _equilibrate_sequential(self):
        """Run equilibration for all replicas sequentially."""
        for rep in self.replicas:
            # Skip already-completed replicas (resume support)
            if (rep.rep_dir / "state.gro").exists():
                rep.current_gro = "state.gro"
                self.logger.info(f"  Replica {rep.index:02d} already equilibrated — skipping")
                continue
            # Clean partial artifacts from previous failed/interrupted runs
            self._clean_partial_eq(rep.rep_dir)
            self.logger.info(f"  Equilibrating replica {rep.index:02d} "
                             f"(λ={rep.lam:.4f})")
            ok = _equilibrate_single_replica(
                rep.index, rep.rep_dir,
                str(self.em_mdp), str(self.nvt_mdp), str(self.npt_mdp),
                self.config["gromacs"]["gmx"],
                self.config["gromacs"]["maxwarn"],
                self.config["gromacs"]["n_threads_per_replica"],
                gpu_id=self._get_gpu_id(rep.index),
            )
            if ok:
                rep.current_gro = "state.gro"
                self.logger.info(f"    Done. Final coords → state.gro")
            else:
                self.logger.error(f"    Equilibration FAILED for replica {rep.index}")

    @staticmethod
    def _clean_partial_eq(rep_dir: Path):
        """Remove partial equilibration artifacts so a fresh run can proceed."""
        import glob as _glob
        for pattern in ["em.*", "nvt.*", "npt.*", "state.gro", "mdout.mdp",
                        "#*#", "step*.pdb", "eq_error.log"]:
            for f in _glob.glob(str(rep_dir / pattern)):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def _equilibrate_parallel(self, max_workers: int):
        """Run equilibration for all replicas in parallel."""
        # Filter out already-completed replicas (resume support)
        pending = []
        for rep in self.replicas:
            if (rep.rep_dir / "state.gro").exists():
                rep.current_gro = "state.gro"
                self.logger.info(f"  Replica {rep.index:02d} already equilibrated — skipping")
            else:
                self._clean_partial_eq(rep.rep_dir)
                pending.append(rep)

        if not pending:
            self.logger.info("  All replicas already equilibrated.")
            return

        self.logger.info(f"  Running equilibration in parallel "
                         f"(max_workers={max_workers}, {len(pending)} replicas pending)")

        futures = {}
        with ProcessPoolExecutor(max_workers=min(max_workers, len(pending))) as pool:
            for rep in pending:
                fut = pool.submit(
                    _equilibrate_single_replica,
                    rep.index, rep.rep_dir,
                    str(self.em_mdp), str(self.nvt_mdp), str(self.npt_mdp),
                    self.config["gromacs"]["gmx"],
                    self.config["gromacs"]["maxwarn"],
                    self.config["gromacs"]["n_threads_per_replica"],
                    gpu_id=self._get_gpu_id(rep.index),
                )
                futures[fut] = rep

            for fut in as_completed(futures):
                rep = futures[fut]
                try:
                    ok = fut.result()
                    if ok:
                        rep.current_gro = "state.gro"
                        self.logger.info(
                            f"  Replica {rep.index:02d} equilibrated OK")
                    else:
                        self.logger.error(
                            f"  Replica {rep.index:02d} equilibration FAILED")
                except Exception as exc:
                    self.logger.error(
                        f"  Replica {rep.index:02d} crashed: {exc}")

        self.logger.info("Equilibration complete.")

    # --------------------------------------------------------
    # Production HREMC cycle
    # --------------------------------------------------------
    def run_production(self, n_cycles: int = None, start_cycle: int = 0):
        """Run the main HREMC production loop."""
        if n_cycles is None:
            n_cycles = self.config["exchange"]["n_cycles"]

        log_interval = self.config["output"]["log_interval"]
        ckpt_interval = self.config["output"]["checkpoint_interval"]
        exchange_dir = self.config["exchange"]["exchange_direction"]

        self.state.cycle = start_cycle
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f" HREMC PRODUCTION — {self.system.upper()}")
        self.logger.info(f" Cycles: {start_cycle} → {start_cycle + n_cycles}")
        self.logger.info(f" Segment: {self.config['md_segment']['segment_time_ps']} ps")
        self.logger.info(f" Total sim time: "
                         f"{n_cycles * self.config['md_segment']['segment_time_ps'] / 1000:.1f} ns")
        self.logger.info(f"{'='*60}\n")

        # Load checkpoint if resuming
        self._load_checkpoint()

        for cycle in range(start_cycle, start_cycle + n_cycles):
            self.state.cycle = cycle
            t_start = time.time()

            # 1. Run MD segments for all replicas
            self._run_md_segments(cycle)

            # 2. Extract energies
            self._extract_energies(cycle)

            # 3. Attempt exchanges
            if exchange_dir == "alternate":
                # Even cycles: try (0,1), (2,3), ...
                # Odd cycles: try (1,2), (3,4), ...
                start = cycle % 2
            else:
                start = 0
            self._attempt_exchanges(cycle, start)

            # 4. Logging
            dt = time.time() - t_start
            if (cycle + 1) % log_interval == 0 or cycle == start_cycle:
                self._log_status(cycle, dt)

            # 5. Checkpoint
            if (cycle + 1) % ckpt_interval == 0:
                self._save_checkpoint()

        # Final checkpoint
        self._save_checkpoint()
        self._write_final_report()

    def _run_md_segments(self, cycle: int):
        """Run MD segments for all replicas — parallel or sequential."""
        parallel_cfg = self.config.get("parallel", {})
        use_parallel = parallel_cfg.get("enabled", False)
        max_workers = parallel_cfg.get("max_workers", self.n_replicas)

        if use_parallel and max_workers > 1:
            self._run_md_segments_parallel(cycle, max_workers)
        else:
            self._run_md_segments_sequential(cycle)

    def _run_md_segments_sequential(self, cycle: int):
        """Run MD segments sequentially (original behavior)."""
        for rep in self.replicas:
            result = _run_single_replica_segment(
                rep.index, rep.rep_dir, cycle,
                str(self.segment_mdp),
                self.config["gromacs"]["gmx"],
                self.config["gromacs"]["maxwarn"],
                self.config["gromacs"]["n_threads_per_replica"],
                self.config["output"]["trajectory_keep_all"],
                self.config["output"]["checkpoint_interval"],
                gpu_id=self._get_gpu_id(rep.index),
            )
            if result["success"]:
                rep.current_gro = "state.gro"
                rep.current_edr = result["edr"]
                rep.cycle = cycle
            else:
                self.logger.error(
                    f"Segment failed: replica {rep.index}, cycle {cycle}: "
                    f"{result.get('error', 'unknown')}")

    def _run_md_segments_parallel(self, cycle: int, max_workers: int):
        """Run MD segments in parallel using ProcessPoolExecutor."""
        self.logger.debug(
            f"Running {self.n_replicas} replica segments in parallel "
            f"(max_workers={max_workers})")

        futures = {}
        with ProcessPoolExecutor(max_workers=min(max_workers, self.n_replicas)) as pool:
            for rep in self.replicas:
                fut = pool.submit(
                    _run_single_replica_segment,
                    rep.index, rep.rep_dir, cycle,
                    str(self.segment_mdp),
                    self.config["gromacs"]["gmx"],
                    self.config["gromacs"]["maxwarn"],
                    self.config["gromacs"]["n_threads_per_replica"],
                    self.config["output"]["trajectory_keep_all"],
                    self.config["output"]["checkpoint_interval"],
                    gpu_id=self._get_gpu_id(rep.index),
                )
                futures[fut] = rep

            for fut in as_completed(futures):
                rep = futures[fut]
                try:
                    result = fut.result()
                    if result["success"]:
                        rep.current_gro = "state.gro"
                        rep.current_edr = result["edr"]
                        rep.cycle = cycle
                    else:
                        self.logger.error(
                            f"Segment failed: replica {rep.index}, "
                            f"cycle {cycle}: {result.get('error', '')}")
                except Exception as exc:
                    self.logger.error(
                        f"Replica {rep.index} crashed: {exc}")

    def _extract_energies(self, cycle: int):
        """Extract potential energies from the latest .edr files."""
        for rep in self.replicas:
            if rep.current_edr:
                energies = self.gmx.energy(
                    rep.current_edr, rep.rep_dir, terms="Potential")
                if "Potential" in energies:
                    rep.potential_energy = energies["Potential"]
                else:
                    self.logger.warning(
                        f"Could not extract energy for replica {rep.index}")

    def _attempt_exchanges(self, cycle: int, start: int = 0):
        """Attempt Metropolis exchanges between neighboring replicas."""
        n = self.n_replicas
        perm = self.state.permutation

        for i in range(start, n - 1, 2):
            j = i + 1
            pair_key = f"{i}-{j}"

            # Get the replicas currently at positions i and j
            rep_i = self.replicas[i]
            rep_j = self.replicas[j]

            # Compute cross-energies using rerun
            # U_j(x_i): energy of config_i under Hamiltonian_j
            # U_i(x_j): energy of config_j under Hamiltonian_i
            U_i_xi = rep_i.potential_energy  # U_i(x_i) — already have this
            U_j_xj = rep_j.potential_energy  # U_j(x_j) — already have this

            U_j_xi = self._rerun_energy(rep_i, rep_j, cycle)
            U_i_xj = self._rerun_energy(rep_j, rep_i, cycle)

            if U_j_xi is None or U_i_xj is None:
                self.logger.warning(
                    f"Rerun failed for pair ({i},{j}), skipping exchange")
                continue

            # Metropolis criterion:
            # ΔΔU = [U_j(x_i) - U_i(x_i)] + [U_i(x_j) - U_j(x_j)]
            delta = (U_j_xi - U_i_xi) + (U_i_xj - U_j_xj)
            prob = min(1.0, math.exp(-self.beta * delta)) if delta > 0 else 1.0

            accepted = random.random() < prob

            # Update statistics
            self.state.total_exchanges_attempted += 1
            self.state.pair_attempts[pair_key] = \
                self.state.pair_attempts.get(pair_key, 0) + 1

            if accepted:
                self.state.total_exchanges_accepted += 1
                self.state.pair_accepts[pair_key] = \
                    self.state.pair_accepts.get(pair_key, 0) + 1

                # Swap configurations (state.gro files)
                self._swap_configs(rep_i, rep_j)

                # Update permutation
                perm[i], perm[j] = perm[j], perm[i]
                rep_i.state_index, rep_j.state_index = \
                    rep_j.state_index, rep_i.state_index

            record = ExchangeRecord(
                cycle=cycle, pair=(i, j),
                delta_energy=delta, probability=prob, accepted=accepted)
            self.state.exchange_log.append(record)

    def _rerun_energy(self, config_replica: Replica,
                      hamiltonian_replica: Replica,
                      cycle: int) -> Optional[float]:
        """Evaluate energy of config_replica's coordinates under
        hamiltonian_replica's Hamiltonian using gmx mdrun -rerun.

        Steps:
        1. Copy config_replica's state.gro to hamiltonian_replica's dir
           as rerun_conf.gro
        2. grompp with hamiltonian_replica's topology + rerun_conf.gro
        3. mdrun -rerun to evaluate energy
        4. Extract potential energy from .edr
        """
        ham_dir = hamiltonian_replica.rep_dir
        conf_gro = config_replica.rep_dir / "state.gro"
        rerun_gro = ham_dir / "rerun_conf.gro"
        rerun_prefix = f"rerun_{cycle:06d}_{config_replica.index:02d}"

        ndx = "index.ndx" if (ham_dir / "index.ndx").exists() else None

        # Copy configuration
        shutil.copy2(conf_gro, rerun_gro)

        # grompp
        ok = self.gmx.grompp(
            str(self.rerun_mdp),
            "rerun_conf.gro", "topol.top",
            f"{rerun_prefix}.tpr",
            ham_dir, ndx=ndx, maxwarn=5)
        if not ok:
            return None

        # mdrun -rerun
        ok = self.gmx.mdrun(
            f"{rerun_prefix}.tpr", ham_dir,
            deffnm=rerun_prefix,
            rerun_trr="rerun_conf.gro")
        if not ok:
            return None

        # Extract energy
        energies = self.gmx.energy(
            f"{rerun_prefix}.edr", ham_dir, terms="Potential")

        # Clean up rerun files
        for ext in [".tpr", ".trr", ".log", ".edr", ".gro"]:
            f = ham_dir / f"{rerun_prefix}{ext}"
            if f.exists():
                f.unlink()
        if rerun_gro.exists():
            rerun_gro.unlink()

        return energies.get("Potential")

    def _swap_configs(self, rep_i: Replica, rep_j: Replica):
        """Swap state.gro files between two replicas."""
        gro_i = rep_i.rep_dir / "state.gro"
        gro_j = rep_j.rep_dir / "state.gro"
        tmp = rep_i.rep_dir / "swap_tmp.gro"
        shutil.copy2(gro_i, tmp)
        shutil.copy2(gro_j, gro_i)
        shutil.copy2(tmp, gro_j)
        tmp.unlink()

    # --------------------------------------------------------
    # Logging and checkpointing
    # --------------------------------------------------------
    def _log_status(self, cycle: int, elapsed: float):
        """Log current HREMC status."""
        rate = self.state.acceptance_rate()
        seg_ps = self.config["md_segment"]["segment_time_ps"]
        total_ns = (cycle + 1) * seg_ps / 1000.0

        msg = (f"Cycle {cycle+1:>6d} | "
               f"Time: {total_ns:>8.1f} ns | "
               f"Accept: {rate:>5.1%} | "
               f"Elapsed: {elapsed:>6.1f}s")
        self.logger.info(msg)

        # Per-pair acceptance rates
        pair_rates = []
        for i in range(self.n_replicas - 1):
            r = self.state.acceptance_rate(pair=(i, i+1))
            pair_rates.append(f"{r:.0%}")
        self.logger.debug(f"  Pair rates: {' | '.join(pair_rates)}")
        self.logger.debug(f"  Permutation: {self.state.permutation}")

    def _save_checkpoint(self):
        """Save HREMC state to disk."""
        ckpt_dir = self.sys_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        state_dict = {
            "system": self.state.system,
            "cycle": self.state.cycle,
            "n_replicas": self.state.n_replicas,
            "lambdas": self.state.lambdas,
            "temperature": self.state.temperature,
            "permutation": self.state.permutation,
            "total_exchanges_attempted": self.state.total_exchanges_attempted,
            "total_exchanges_accepted": self.state.total_exchanges_accepted,
            "pair_attempts": self.state.pair_attempts,
            "pair_accepts": self.state.pair_accepts,
            "replica_states": [
                {"index": r.index, "state_index": r.state_index,
                 "lam": r.lam, "potential_energy": r.potential_energy}
                for r in self.replicas
            ],
        }

        ckpt_file = ckpt_dir / f"checkpoint_{self.state.cycle:06d}.json"
        with open(ckpt_file, "w") as f:
            json.dump(state_dict, f, indent=2)

        # Also save latest
        latest = ckpt_dir / "checkpoint_latest.json"
        with open(latest, "w") as f:
            json.dump(state_dict, f, indent=2)

        self.logger.info(f"Checkpoint saved: cycle {self.state.cycle}")

    def _load_checkpoint(self):
        """Load the latest checkpoint if it exists."""
        ckpt_file = self.sys_dir / "checkpoints" / "checkpoint_latest.json"
        if not ckpt_file.exists():
            self.logger.info("No checkpoint found — starting fresh")
            return

        with open(ckpt_file) as f:
            state_dict = json.load(f)

        self.state.cycle = state_dict["cycle"]
        self.state.permutation = state_dict["permutation"]
        self.state.total_exchanges_attempted = state_dict["total_exchanges_attempted"]
        self.state.total_exchanges_accepted = state_dict["total_exchanges_accepted"]
        self.state.pair_attempts = state_dict["pair_attempts"]
        self.state.pair_accepts = state_dict["pair_accepts"]

        for r_data in state_dict.get("replica_states", []):
            idx = r_data["index"]
            if idx < len(self.replicas):
                self.replicas[idx].state_index = r_data["state_index"]
                self.replicas[idx].potential_energy = r_data["potential_energy"]

        self.logger.info(f"Resumed from checkpoint: cycle {self.state.cycle}")

    def _write_final_report(self):
        """Write a summary report of the HREMC run."""
        report_path = self.sys_dir / "hremc_report.txt"
        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f" HREMC REPORT — {self.system.upper()}\n")
            f.write("=" * 60 + "\n\n")

            seg_ps = self.config["md_segment"]["segment_time_ps"]
            total_ns = (self.state.cycle + 1) * seg_ps / 1000.0
            f.write(f"Total cycles:            {self.state.cycle + 1}\n")
            f.write(f"Total simulation time:   {total_ns:.1f} ns per replica\n")
            f.write(f"Total exchanges tried:   {self.state.total_exchanges_attempted}\n")
            f.write(f"Total exchanges accepted:{self.state.total_exchanges_accepted}\n")
            f.write(f"Overall acceptance rate:  {self.state.acceptance_rate():.1%}\n\n")

            f.write("Per-pair acceptance rates:\n")
            f.write("-" * 40 + "\n")
            for i in range(self.n_replicas - 1):
                pair = (i, i + 1)
                rate = self.state.acceptance_rate(pair)
                key = f"{i}-{i+1}"
                att = self.state.pair_attempts.get(key, 0)
                acc = self.state.pair_accepts.get(key, 0)
                f.write(f"  Pair ({i:2d}, {i+1:2d}): "
                        f"λ=({self.lambdas[i]:.4f}, {self.lambdas[i+1]:.4f}) "
                        f"→ {rate:.1%} ({acc}/{att})\n")

            f.write(f"\nFinal permutation: {self.state.permutation}\n")
            f.write(f"\nLambda schedule:\n")
            for i, lam in enumerate(self.lambdas):
                f.write(f"  Replica {i:02d}: λ = {lam:.6f}\n")

        self.logger.info(f"Report written to {report_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="HREMC Engine for CG SCN5A Nav1.5")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config")
    parser.add_argument("--system", choices=["wt", "mutant", "both"],
                        default="both",
                        help="Which system to run")
    parser.add_argument("--equilibrate", action="store_true",
                        help="Run equilibration (EM + NVT + NPT)")
    parser.add_argument("--production", action="store_true",
                        help="Run production HREMC cycles")
    parser.add_argument("--n-cycles", type=int, default=None,
                        help="Override number of production cycles")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc_root = config_path.parent.resolve()
    systems = ["wt", "mutant"] if args.system == "both" else [args.system]

    for system in systems:
        engine = HREMCEngine(config, system, mc_root)

        if args.equilibrate:
            engine.equilibrate()

        if args.production:
            start = 0
            if args.resume:
                ckpt = mc_root / system / "checkpoints" / "checkpoint_latest.json"
                if ckpt.exists():
                    with open(ckpt) as f:
                        start = json.load(f)["cycle"] + 1

            n_cycles = args.n_cycles or config["exchange"]["n_cycles"]
            engine.run_production(n_cycles=n_cycles, start_cycle=start)


if __name__ == "__main__":
    main()
