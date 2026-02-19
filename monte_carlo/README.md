# Hamiltonian Replica Exchange Monte Carlo (HREMC)
## SCN5A Nav1.5 — WT & RXXXH Mutant (Martini 3 CG)

### Overview

Hamiltonian Replica Exchange Monte Carlo (HREMC) enhances conformational sampling
of the Nav1.5 sodium channel by running 24 replicas in parallel, each with a
different Go-model contact strength (ε scaling). Configurations are periodically
exchanged between replicas using the Metropolis criterion, allowing the protein to
traverse energy barriers that trap conventional MD.

**Hamiltonian perturbation**: The Go-model native contact strength (ε = 9.414 kJ/mol
from martinize2) is scaled by λ ∈ [1.0, 0.5] across 24 replicas with geometric
spacing. Lower λ weakens native contacts, effectively "heating" the protein's
conformational landscape without actually raising the temperature.

**Exchange criterion**:
```
ΔΔU = [U_j(x_i) − U_i(x_i)] + [U_i(x_j) − U_j(x_j)]
P_accept = min(1, exp(−β · ΔΔU))
```
Cross-energies U_j(x_i) are evaluated via `gmx mdrun -rerun`.

---

### Directory Structure

```
monte_carlo/
├── config.yaml              # All simulation parameters
├── setup_hremc.py           # Generates replica directories
├── hremc_engine.py          # Main HREMC engine (MD + MC exchange)
├── analyze_hremc.py         # Post-run analysis & plotting
├── run_hremc.sh             # Master run script
├── submit_hremc.sh          # HPC (SLURM) submission template
├── README.md                # This file
├── mdp/
│   ├── em.mdp               # Energy minimization
│   ├── nvt_eq.mdp           # NVT equilibration (100 ps)
│   ├── npt_eq.mdp           # NPT equilibration (200 ps)
│   ├── hremc_segment.mdp    # Production MD segment (100 ps)
│   └── rerun.mdp            # Energy re-evaluation
├── wt/
│   ├── lambda_schedule.dat  # Lambda values for all replicas
│   ├── replica_00/          # λ = 1.0000 (native)
│   │   ├── topol.top
│   │   ├── go_nbparams_scaled.itp
│   │   ├── conf.gro
│   │   ├── state.gro        # Current config (updated each cycle)
│   │   └── ...
│   ├── replica_01/          # λ = 0.9703
│   ├── ...
│   ├── replica_23/          # λ = 0.5000 (most weakened)
│   ├── checkpoints/
│   ├── logs/
│   └── hremc_report.txt
├── mutant/                  # Same structure as wt/
└── analysis/
    ├── wt/
    │   ├── acceptance_rates_wt.png
    │   ├── replica_flow_wt.png
    │   ├── lambda_schedule_wt.png
    │   └── analysis_report_wt.txt
    └── mutant/
```

---

### Quick Start

```bash
cd variantid_SCN5A/extended_simulation/monte_carlo

# 1. Setup — generate all 24 replica directories
./run_hremc.sh setup both

# 2. Equilibrate — EM + NVT + NPT for each replica
./run_hremc.sh equilibrate both

# 3. Production — run 1000 HREMC cycles (100 ns per replica)
./run_hremc.sh production both

# 4. Check status
./run_hremc.sh status

# 5. Resume from checkpoint if interrupted
./run_hremc.sh resume both

# 6. Analyze results
python analyze_hremc.py --config config.yaml
```

---

### Lambda Schedule (24 replicas)

| Replica | λ        | ε_Go (kJ/mol) | T_eff (K) |
|---------|----------|----------------|-----------|
| 00      | 1.000000 | 9.414          | 310.0     |
| 01      | 0.970338 | 9.134          | 319.5     |
| 06      | 0.834464 | 7.856          | 371.5     |
| 11      | 0.717370 | 6.753          | 432.2     |
| 17      | 0.589701 | 5.551          | 525.7     |
| 23      | 0.500000 | 4.707          | 620.0     |

Geometric spacing ensures roughly uniform acceptance rates across all pairs.

---

### Configuration

Edit `config.yaml` to adjust:

- **`replicas.n_replicas`**: Number of λ-windows (default: 24)
- **`replicas.lambda_min/max`**: Go-contact scaling range
- **`md_segment.nsteps`**: Steps per HREMC segment (100 ps default)
- **`exchange.n_cycles`**: Total HREMC cycles (1000 = 100 ns)
- **`gromacs.n_threads_per_replica`**: OpenMP threads per replica

---

### Key Design Decisions

1. **Go-contact scaling (not temperature)**: True Hamiltonian RE — only the
   protein's native contact network is perturbed, not the solvent or lipids.
   This is far more efficient than temperature REMD for large membrane systems.

2. **External MC exchange**: The Python engine manages exchanges rather than
   GROMACS's built-in `-replex`. This allows full control over the exchange
   protocol, cross-energy evaluation, and convergence monitoring.

3. **Alternating sweep direction**: Even/odd cycles attempt different pairs,
   maximizing mixing efficiency.

4. **Cross-energy evaluation via `-rerun`**: Accurate Metropolis criterion
   without approximations. Each candidate swap evaluates the full potential
   energy under the neighboring Hamiltonian.

5. **Checkpointing**: Full state saved every 50 cycles, enabling restart
   and post-hoc analysis.

---

### Analysis Outputs

- **Acceptance rates**: Per-pair bar plot + overall rate. Target: 20–40%.
- **Replica flow**: State-index trajectory showing mixing quality.
- **Round-trip times**: How many cycles for replicas to traverse the full
  λ-ladder. Shorter = better mixing.
- **Energy time series**: Convergence diagnostic per replica.

---

### Requirements

- GROMACS ≥ 2023 (with c-rescale barostat)
- Python ≥ 3.9
- PyYAML (`pip install pyyaml`)
- NumPy, Matplotlib (for analysis)

---

### References

- Sugita & Okamoto (1999). Replica-exchange molecular dynamics method for protein folding. *Chem Phys Lett*, 314, 141–151.
- Liu et al. (2005). Replica exchange with solute tempering (REST). *PNAS*, 102, 13749.
- Bussi (2014). Hamiltonian replica exchange in GROMACS. *JCTC*, 10, 2906.
- Souza et al. (2021). Martini 3: a general purpose force field for CG simulations. *Nat Methods*, 18, 382.
