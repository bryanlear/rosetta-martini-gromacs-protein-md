#!/usr/bin/env python3
"""
analyze_hremc.py — Analysis tools for Hamiltonian Replica Exchange Monte Carlo
===============================================================================
SCN5A Nav1.5 (WT & RxxxH mutant)

Generates:
  - Acceptance rate plots (per pair + overall)
  - Replica exchange trajectory (state index vs cycle)
  - Round-trip time analysis
  - Lambda demultiplexing (extract physical-state trajectories)
  - Energy overlap histograms
  - Convergence diagnostics

Usage:
    python analyze_hremc.py --config config.yaml [--system wt|mutant|both]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available — skipping plots")

import yaml


# ============================================================
# Load checkpoint data
# ============================================================
def load_checkpoints(sys_dir: Path) -> list[dict]:
    """Load all checkpoint files sorted by cycle."""
    ckpt_dir = sys_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    ckpts = []
    for f in sorted(ckpt_dir.glob("checkpoint_*.json")):
        if "latest" in f.name:
            continue
        with open(f) as fh:
            ckpts.append(json.load(fh))
    return ckpts


def load_latest_checkpoint(sys_dir: Path) -> dict:
    """Load the latest checkpoint."""
    ckpt_file = sys_dir / "checkpoints" / "checkpoint_latest.json"
    if not ckpt_file.exists():
        return {}
    with open(ckpt_file) as f:
        return json.load(f)


# ============================================================
# Read exchange log from HREMC log file
# ============================================================
def parse_exchange_log(sys_dir: Path) -> list[dict]:
    """Parse the HREMC log to extract exchange records."""
    log_file = sys_dir / "logs" / "hremc.log"
    if not log_file.exists():
        return []
    # The exchange log is stored in checkpoints as well
    # For detailed analysis, we use the checkpoint data
    return []


# ============================================================
# Analysis functions
# ============================================================
def compute_acceptance_rates(state: dict) -> dict:
    """Compute per-pair and overall acceptance rates."""
    n = state.get("n_replicas", 24)
    rates = {}

    for i in range(n - 1):
        key = f"{i}-{i+1}"
        att = state.get("pair_attempts", {}).get(key, 0)
        acc = state.get("pair_accepts", {}).get(key, 0)
        rates[key] = {
            "attempts": att,
            "accepted": acc,
            "rate": acc / att if att > 0 else 0.0,
            "lambda_i": state["lambdas"][i],
            "lambda_j": state["lambdas"][i + 1],
        }

    overall_att = state.get("total_exchanges_attempted", 0)
    overall_acc = state.get("total_exchanges_accepted", 0)
    rates["overall"] = {
        "attempts": overall_att,
        "accepted": overall_acc,
        "rate": overall_acc / overall_att if overall_att > 0 else 0.0,
    }
    return rates


def compute_round_trip_times(checkpoints: list[dict], n_replicas: int) -> dict:
    """Estimate round-trip times from checkpoint permutations.

    A round trip = time for a replica to travel from state 0 → state (N-1) → state 0.
    """
    if len(checkpoints) < 2:
        return {}

    # Track each replica's state index over cycles
    trajectories = defaultdict(list)
    for ckpt in checkpoints:
        cycle = ckpt["cycle"]
        perm = ckpt["permutation"]
        for slot, state in enumerate(perm):
            trajectories[slot].append((cycle, state))

    round_trips = {}
    for replica, traj in trajectories.items():
        trips = []
        visited_min = False
        visited_max = False
        trip_start = traj[0][0]

        for cycle, state in traj:
            if state == 0:
                if visited_max:
                    trips.append(cycle - trip_start)
                    trip_start = cycle
                    visited_max = False
                visited_min = True
            elif state == n_replicas - 1:
                visited_max = True

        round_trips[replica] = {
            "n_trips": len(trips),
            "mean_time": np.mean(trips) if trips else float("inf"),
            "min_time": min(trips) if trips else float("inf"),
        }

    return round_trips


def compute_replica_flow(checkpoints: list[dict], n_replicas: int) -> np.ndarray:
    """Build a 2D array: flow[cycle_idx, replica_slot] = state_index."""
    n_ckpt = len(checkpoints)
    flow = np.zeros((n_ckpt, n_replicas), dtype=int)
    cycles = []
    for idx, ckpt in enumerate(checkpoints):
        cycles.append(ckpt["cycle"])
        perm = ckpt["permutation"]
        for slot in range(n_replicas):
            flow[idx, slot] = perm[slot]
    return np.array(cycles), flow


# ============================================================
# Plotting functions
# ============================================================
def plot_acceptance_rates(rates: dict, output_dir: Path, system: str):
    """Bar plot of per-pair acceptance rates."""
    if not HAS_MPL:
        return

    pairs = sorted([k for k in rates if k != "overall"],
                   key=lambda x: int(x.split("-")[0]))
    pair_labels = [f"{k}" for k in pairs]
    pair_rates = [rates[k]["rate"] for k in pairs]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(pairs)))
    bars = ax.bar(pair_labels, pair_rates, color=colors, edgecolor="black", alpha=0.8)

    # Target zone (20-40% is optimal for REMD)
    ax.axhspan(0.20, 0.40, color="green", alpha=0.1, label="Target (20-40%)")
    ax.axhline(y=rates["overall"]["rate"], color="red", linestyle="--",
               linewidth=2, label=f'Overall: {rates["overall"]["rate"]:.1%}')

    ax.set_xlabel("Replica pair (i — i+1)", fontsize=12)
    ax.set_ylabel("Acceptance rate", fontsize=12)
    ax.set_title(f"HREMC Acceptance Rates — {system.upper()}", fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f"acceptance_rates_{system}.png", dpi=150)
    plt.close()


def plot_replica_flow(cycles: np.ndarray, flow: np.ndarray,
                      output_dir: Path, system: str, n_replicas: int):
    """Replica exchange flow diagram — shows state mixing."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.cm.tab20(np.linspace(0, 1, n_replicas))

    for rep in range(n_replicas):
        ax.plot(cycles, flow[:, rep], "-", color=cmap[rep],
                alpha=0.6, linewidth=0.8, label=f"Rep {rep}")

    ax.set_xlabel("HREMC Cycle", fontsize=12)
    ax.set_ylabel("State index (λ-window)", fontsize=12)
    ax.set_title(f"Replica Exchange Flow — {system.upper()}", fontsize=14)
    ax.set_yticks(range(n_replicas))
    if n_replicas <= 24:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
                  fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / f"replica_flow_{system}.png", dpi=150)
    plt.close()


def plot_lambda_schedule(lambdas: list, output_dir: Path, system: str):
    """Plot the lambda schedule with effective temperatures."""
    if not HAS_MPL:
        return
    T_ref = 310.0
    T_eff = [T_ref / lam for lam in lambdas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Lambda values
    ax1.plot(range(len(lambdas)), lambdas, "o-", color="steelblue", markersize=6)
    ax1.set_xlabel("Replica index", fontsize=12)
    ax1.set_ylabel("λ (Go-contact scaling)", fontsize=12)
    ax1.set_title("Lambda Schedule", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Effective temperatures
    ax2.plot(range(len(T_eff)), T_eff, "s-", color="firebrick", markersize=6)
    ax2.set_xlabel("Replica index", fontsize=12)
    ax2.set_ylabel("Effective temperature (K)", fontsize=12)
    ax2.set_title("Effective Temperature", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"HREMC Lambda Ladder — {system.upper()}", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"lambda_schedule_{system}.png", dpi=150,
                bbox_inches="tight")
    plt.close()


def plot_energy_timeseries(checkpoints: list[dict], output_dir: Path,
                           system: str, n_replicas: int):
    """Plot potential energy time series per replica."""
    if not HAS_MPL or not checkpoints:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, n_replicas))

    for rep_idx in range(n_replicas):
        cycles = []
        energies = []
        for ckpt in checkpoints:
            for r in ckpt.get("replica_states", []):
                if r["index"] == rep_idx and r["potential_energy"] != 0:
                    cycles.append(ckpt["cycle"])
                    energies.append(r["potential_energy"])
        if cycles:
            ax.plot(cycles, energies, "-", color=cmap[rep_idx],
                    alpha=0.5, linewidth=0.8)

    ax.set_xlabel("HREMC Cycle", fontsize=12)
    ax.set_ylabel("Potential Energy (kJ/mol)", fontsize=12)
    ax.set_title(f"Energy Time Series — {system.upper()}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"energy_timeseries_{system}.png", dpi=150)
    plt.close()


# ============================================================
# Write text report
# ============================================================
def write_analysis_report(rates: dict, round_trips: dict,
                          checkpoints: list[dict],
                          lambdas: list, output_dir: Path, system: str):
    """Write a comprehensive text analysis report."""
    report_path = output_dir / f"analysis_report_{system}.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"  HREMC ANALYSIS REPORT — {system.upper()}\n")
        f.write("=" * 70 + "\n\n")

        # 1. Acceptance rates
        f.write("1. ACCEPTANCE RATES\n")
        f.write("-" * 50 + "\n")
        overall = rates["overall"]
        f.write(f"   Overall: {overall['rate']:.1%} "
                f"({overall['accepted']}/{overall['attempts']})\n\n")

        f.write(f"   {'Pair':>8s}  {'λ_i':>8s}  {'λ_j':>8s}  "
                f"{'Rate':>6s}  {'Acc':>5s}  {'Att':>5s}\n")
        for i in range(len(lambdas) - 1):
            key = f"{i}-{i+1}"
            r = rates.get(key, {})
            f.write(f"   {key:>8s}  {r.get('lambda_i',0):>8.4f}  "
                    f"{r.get('lambda_j',0):>8.4f}  "
                    f"{r.get('rate',0):>5.1%}  "
                    f"{r.get('accepted',0):>5d}  "
                    f"{r.get('attempts',0):>5d}\n")

        # Flag problematic pairs
        low_pairs = [k for k, v in rates.items()
                     if k != "overall" and v.get("rate", 0) < 0.10
                     and v.get("attempts", 0) > 10]
        high_pairs = [k for k, v in rates.items()
                      if k != "overall" and v.get("rate", 0) > 0.60
                      and v.get("attempts", 0) > 10]
        if low_pairs:
            f.write(f"\n   ⚠ Low acceptance (<10%): {', '.join(low_pairs)}\n")
            f.write("     → Consider adding intermediate λ values\n")
        if high_pairs:
            f.write(f"\n   ℹ High acceptance (>60%): {', '.join(high_pairs)}\n")
            f.write("     → Could remove some λ values for efficiency\n")

        # 2. Round trips
        f.write(f"\n2. ROUND-TRIP TIMES\n")
        f.write("-" * 50 + "\n")
        if round_trips:
            for rep, rt in sorted(round_trips.items()):
                f.write(f"   Replica {rep:>2d}: "
                        f"{rt['n_trips']} trips, "
                        f"mean = {rt['mean_time']:.0f} cycles\n")
        else:
            f.write("   Insufficient data for round-trip analysis\n")

        # 3. Lambda schedule
        f.write(f"\n3. LAMBDA SCHEDULE\n")
        f.write("-" * 50 + "\n")
        T_ref = 310.0
        for i, lam in enumerate(lambdas):
            T_eff = T_ref / lam
            f.write(f"   Replica {i:>2d}: λ = {lam:.6f}, "
                    f"T_eff = {T_eff:.1f} K\n")

        # 4. Simulation progress
        if checkpoints:
            last = checkpoints[-1]
            f.write(f"\n4. SIMULATION PROGRESS\n")
            f.write("-" * 50 + "\n")
            f.write(f"   Last cycle:   {last['cycle']}\n")
            f.write(f"   Checkpoints:  {len(checkpoints)}\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"  Report: {report_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Analyze HREMC results for CG SCN5A Nav1.5")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--system", choices=["wt", "mutant", "both"],
                        default="both")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc_root = config_path.parent
    systems = ["wt", "mutant"] if args.system == "both" else [args.system]

    for system in systems:
        sys_dir = mc_root / system
        if not sys_dir.exists():
            print(f"  Skipping {system}: directory not found")
            continue

        print(f"\n{'='*60}")
        print(f" Analyzing {system.upper()}")
        print(f"{'='*60}")

        # Output directory
        out_dir = mc_root / "analysis" / system
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        state = load_latest_checkpoint(sys_dir)
        checkpoints = load_checkpoints(sys_dir)
        n_rep = config["replicas"]["n_replicas"]

        if not state:
            print(f"  No checkpoint data found for {system}")
            # Still plot lambda schedule from config
            from setup_hremc import generate_lambda_ladder
            lambdas = generate_lambda_ladder(
                n_rep, config["replicas"]["lambda_min"],
                config["replicas"]["lambda_max"],
                config["replicas"]["spacing"])
            plot_lambda_schedule(lambdas, out_dir, system)
            continue

        lambdas = state["lambdas"]

        # 1. Acceptance rates
        rates = compute_acceptance_rates(state)
        plot_acceptance_rates(rates, out_dir, system)

        # 2. Replica flow
        if checkpoints:
            cycles, flow = compute_replica_flow(checkpoints, n_rep)
            plot_replica_flow(cycles, flow, out_dir, system, n_rep)

        # 3. Round trips
        round_trips = compute_round_trip_times(checkpoints, n_rep)

        # 4. Lambda schedule
        plot_lambda_schedule(lambdas, out_dir, system)

        # 5. Energy time series
        plot_energy_timeseries(checkpoints, out_dir, system, n_rep)

        # 6. Text report
        write_analysis_report(rates, round_trips, checkpoints,
                              lambdas, out_dir, system)

    print(f"\n Analysis complete. Results in: {mc_root / 'analysis'}/")


if __name__ == "__main__":
    main()
