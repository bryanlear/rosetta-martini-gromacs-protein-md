# SCN5A R376H All-Atom MD Simulation Report

**Date:** February 15, 2026  
**Variant:** rs199473101 (R376H missense mutation)  
**Protein:** SCN5A — Cardiac Voltage-Gated Sodium Channel Nav1.5  

---

## 1. System Composition

| Component       | Count   | Notes                                      |
|-----------------|--------:|---------------------------------------------|
| Protein atoms   | 31,834  | SCN5A R376H, backmapped from Martini 3 CG   |
| POPC            | 297     | Phosphatidylcholine (outer/inner leaflet)    |
| POPE            | 146     | Phosphatidylethanolamine                     |
| CHOL            | 236     | Cholesterol                                  |
| POPS            | 28      | Phosphatidylserine (inner leaflet)           |
| TIP3P water     | 210,376 | ~631,128 atoms                               |
| Na⁺ ions        | 615     | Charge neutralization + 0.15 M NaCl          |
| Cl⁻ ions        | 544     | Charge neutralization + 0.15 M NaCl          |
| **Total atoms** | **743,189** |                                          |

**Force field:** CHARMM36 (all-atom, backmapped from Martini 3 CG via CG2AT)  
**Software:** GROMACS 2026.0 (single precision, GPU-accelerated)

---

## 2. Hardware

| Resource       | Specification              |
|----------------|----------------------------|
| GPU            | NVIDIA H100 80 GB HBM3     |
| CPU threads    | 4 OpenMP threads, 1 MPI rank |
| GPU offload    | nb, pme, bonded, update    |
| Server         | JarvisLabs cloud instance  |

---

## 3. Simulation Pipeline & Timing

| Stage                      | Sim. Time | Steps       | dt (ps) | Wall Clock     | Performance   | Start → Finish                      |
|----------------------------|-----------|-------------|---------|----------------|---------------|--------------------------------------|
| Energy Minimization        | —         | 50,000 max  | —       | ~13 min        | —             | Feb 12, 20:00 → 20:14               |
| NVT Equilibration          | 0.5 ns    | 250,000     | 0.002   | ~1 min 46 s    | 1.17 ns/day*  | Feb 12, 21:28 → 21:30               |
| NPT Restrained             | 5.0 ns    | 2,500,000   | 0.002   | ~2 h 12 min    | 54.5 ns/day   | Feb 14 (logged in npt_run.log)       |
| NPT Unrestrained           | 6.0 ns    | 3,000,000   | 0.002   | 3 h 01 min     | 47.8 ns/day   | Feb 14, 23:05 → Feb 15, 02:06       |
| **Production**             | **10.0 ns** | **5,000,000** | **0.002** | **5 h 17 min** | **45.4 ns/day** | **Feb 15, 02:45 → 08:02**       |

> *NVT ran on 10 OpenMP threads before switching to 4-thread GPU-offload configuration.

### Total Simulation Wall Time: ~10 h 44 min  
### Total Simulated Time: 21.5 ns (0.5 + 5.0 + 6.0 + 10.0 ns)
### Total Disk Usage: 6.6 GB (pipeline) / 2.9 GB (production)

---

## 4. Production MD Parameters

| Parameter                | Value                                  |
|--------------------------|----------------------------------------|
| Integrator               | md (leap-frog)                         |
| Timestep                 | 2 fs                                   |
| Thermostat               | v-rescale, τ_T = 0.1 ps, T = 310 K    |
| Barostat                 | Parrinello-Rahman, τ_p = 12.0 ps       |
| Pressure coupling        | Semiisotropic, P_ref = 1.0 bar         |
| Compressibility          | 4.5 × 10⁻⁵ bar⁻¹                      |
| Electrostatics           | PME (GPU), r_coulomb = 1.2 nm          |
| Van der Waals            | Force-switch, r_vdw = 1.2 nm           |
| Neighbor list            | Verlet, nstlist = 40 (auto-tuned: 100) |
| Position restraints      | None                                   |
| Output frequency (xtc)   | Every 10 ps (nstxout-compressed = 5000) |
| Energy output            | Every 10 ps (nstenergy = 5000)         |
| Trajectory frames        | 1,001                                  |

---

## 5. Thermodynamic Stability

| Property       | Mean ± Std           | Target     | Status |
|----------------|----------------------|------------|--------|
| Temperature    | 310.017 ± 0.377 K   | 310 K      | ✅      |
| Pressure       | 1.38 ± 36.37 bar    | 1.0 bar    | ✅      |
| Density        | 1017.83 ± 0.73 kg/m³| ~1000–1020 | ✅      |

> Pressure fluctuations of ±36 bar are normal for NPT simulations of this system size.

---

## 6. Structural Analysis

### 6.1 Backbone RMSD (vs. initial structure)

| Metric         | Value       |
|----------------|-------------|
| Mean RMSD      | 0.736 nm    |
| Std RMSD       | 0.171 nm    |
| Final RMSD     | 0.877 nm    |

The RMSD shows a rising trend from 0 → ~0.88 nm over 10 ns, indicating the protein is still adapting from the backmapped starting structure. This is expected for a large transmembrane channel in a short production run.

### 6.2 Radius of Gyration

| Metric  | Value           |
|---------|-----------------|
| Mean Rg | 4.720 ± 0.040 nm|

Rg remains stable throughout, indicating the overall protein compactness is preserved.

### 6.3 DSSP Secondary Structure (averaged over last 1 ns)

| Structure Type    | Avg. Residues |
|-------------------|---------------|
| α-Helices         | 890           |
| Loops             | 442           |
| Bends             | 333           |
| Turns             | 180           |
| PP-II Helices     | 74            |
| β-Strands         | 40            |
| 3₁₀-Helices       | 40            |
| β-Bridges         | 18            |
| π-Helices         | <1            |

Secondary structure is well-maintained. α-helices dominate as expected for a voltage-gated ion channel.

### 6.4 Intra-Protein Hydrogen Bonds

| Metric       | Value           |
|--------------|-----------------|
| Mean H-bonds | 1,232 ± 18      |

The H-bond network is stable, consistent with maintained secondary structure.

---

## 7. Membrane Analysis

### 7.1 Area Per Lipid (APL)

| Metric  | Value                   | Literature (pure POPC) |
|---------|-------------------------|------------------------|
| APL     | 69.78 ± 0.55 Å²        | ~64–68 Å²              |
|         | 0.698 ± 0.006 nm²      |                        |

Slightly elevated APL is expected for a mixed bilayer (POPC/POPE/CHOL/POPS) with an embedded transmembrane protein.

### 7.2 Bilayer Thickness (Phosphate-to-Phosphate)

| Metric  | Value               | Literature (pure POPC) |
|---------|---------------------|------------------------|
| d_PP    | 3.927 ± 0.030 nm   | ~3.8–4.0 nm            |

Consistent with experimental and computational data for POPC-based bilayers.

### 7.3 Deuterium Order Parameters (|S_CD|) — POPC

Computed using the C-C-C angle method (MDAnalysis, 1,001 frames, 297 POPC lipids).

**sn-1 chain (palmitoyl, C32–C316):**

| Carbon | |S_CD|  | ± Std  |
|--------|---------|--------|
| C32    | 0.1955  | 0.0108 |
| C33    | 0.1750  | 0.0121 |
| C34    | 0.2096  | 0.0124 |
| C35    | 0.2158  | 0.0117 |
| C36    | 0.2307  | 0.0118 |
| C37    | 0.2275  | 0.0121 |
| C38    | 0.2282  | 0.0118 |
| C39    | 0.2170  | 0.0128 |
| C310   | 0.2088  | 0.0131 |
| C311   | 0.1896  | 0.0132 |
| C312   | 0.1735  | 0.0133 |
| C313   | 0.1494  | 0.0142 |
| C314   | 0.1263  | 0.0145 |
| C315   | 0.0923  | 0.0150 |
| C316   | 0.0313  | 0.0088 |

Plateau at C36–C38 (~0.23) with monotonic decay toward the methyl terminus — classic saturated acyl chain profile.

**sn-2 chain (oleoyl, C22–C218):**

| Carbon | |S_CD|  | ± Std  |
|--------|---------|--------|
| C22    | 0.0916  | 0.0128 |
| C23    | 0.1909  | 0.0128 |
| C24    | 0.1974  | 0.0134 |
| C25    | 0.2157  | 0.0121 |
| C26    | 0.2024  | 0.0133 |
| C27    | 0.1931  | 0.0133 |
| C28    | 0.1211  | 0.0135 |
| C29    | 0.0823  | 0.0217 |
| C210   | 0.0762  | 0.0226 |
| C211   | 0.1044  | 0.0142 |
| C212   | 0.1536  | 0.0145 |
| C213   | 0.1574  | 0.0136 |
| C214   | 0.1592  | 0.0139 |
| C215   | 0.1371  | 0.0137 |
| C216   | 0.1222  | 0.0138 |
| C217   | 0.0903  | 0.0147 |
| C218   | 0.0331  | 0.0087 |

Characteristic dip at C29–C210 (|S_CD| ~ 0.08) corresponding to the *cis* C9=C10 double bond in the oleoyl chain.

### 7.4 POPC Lateral Diffusion (MSD)

| Metric                    | Value                          |
|---------------------------|--------------------------------|
| Diffusion coefficient (D) | 0.0053 × 10⁻⁵ cm²/s           |
|                           | = 5.3 × 10⁻⁸ cm²/s            |

Computed via `gmx msd -lateral z`. The value is in the expected range for POPC in mixed bilayers (experimental: ~5–10 × 10⁻⁸ cm²/s at 310 K).

### 7.5 Membrane Density Profile

POPC partial density along the z-axis shows the expected symmetric bilayer distribution centered on the membrane midplane.

---

## 8. Production Performance Breakdown

| GPU Activity              | Wall Time (s)  | % of Total |
|---------------------------|----------------|------------|
| Neighbor search           | 3,605.7        | 19.0%      |
| Wait GPU state copy       | 10,577.3       | 55.6%      |
| Kinetic energy            | 1,584.2        | 8.3%       |
| Force                     | 1,251.6        | 6.6%       |
| Launch PP GPU ops         | 734.1          | 3.9%       |
| Wait GPU NB local         | 473.1          | 2.5%       |
| PME GPU mesh              | 304.7          | 1.6%       |
| NB X/F buffer ops         | 190.1          | 1.0%       |
| Write trajectory          | 26.7           | 0.1%       |
| **Total wall time**       | **19,019.6 s (5 h 17 min)** | |

| Performance Metric        | Value          |
|---------------------------|----------------|
| Throughput                | 45.4 ns/day    |
| Time per step             | 3.804 ms       |
| Atom throughput           | 195.4 Matom·steps/s |
| Core time (4 threads)     | 76,078 s       |
| CPU efficiency            | 400%           |

---

## 9. Generated Plots (27 total)

All plots saved to `plots_uncalibrated_all_atom/`:

| #  | Filename                         | Content                        |
|----|----------------------------------|--------------------------------|
| 01 | 01_em_potential.png              | EM potential energy            |
| 02 | 02_nvt_temperature.png           | NVT temperature                |
| 03 | 03_nvt_potential.png             | NVT potential energy           |
| 04 | 04_npt_rest_pressure.png         | NPT restrained pressure        |
| 05 | 05_npt_rest_density.png          | NPT restrained density         |
| 06 | 06_npt_rest_temperature.png      | NPT restrained temperature     |
| 07 | 07_npt_rest_box.png              | NPT restrained box dimensions  |
| 08 | 08_npt_unrest_pressure.png       | NPT unrestrained pressure      |
| 09 | 09_npt_unrest_density.png        | NPT unrestrained density       |
| 10 | 10_npt_unrest_box.png            | NPT unrestrained box dims      |
| 11 | 11_prod_pressure.png             | Production pressure             |
| 12 | 12_prod_density.png              | Production density              |
| 13 | 13_prod_temperature.png          | Production temperature          |
| 14 | 14_prod_potential.png            | Production potential energy     |
| 15 | 15_prod_total_energy.png         | Production total energy         |
| 16 | 16_prod_box_vol.png              | Production box volume           |
| 17 | 17_prod_rmsd_backbone.png        | Backbone RMSD                  |
| 18 | 18_prod_rmsf_backbone.png        | Per-residue RMSF               |
| 19 | 19_prod_gyration.png             | Radius of gyration             |
| 20 | 20_prod_dssp.png                 | DSSP secondary structure       |
| 21 | 21_prod_hbonds.png               | Intra-protein H-bonds          |
| 22 | 22_prod_density_profile.png      | POPC membrane density profile  |
| 23 | 23_prod_msd_popc.png             | POPC lateral MSD               |
| 24 | 24_prod_scd_sn1.png              | S_CD sn-1 order parameters     |
| 25 | 25_prod_scd_sn2.png              | S_CD sn-2 order parameters     |
| 26 | 26_prod_apl.png                  | Area per lipid vs time         |
| 27 | 27_prod_bilayer_thickness.png    | Bilayer P-P thickness vs time  |

---

## 10. File Inventory

### Production directory (`FINAL/production/`)

| File                    | Size  | Description                        |
|-------------------------|-------|------------------------------------|
| production.tpr          | ~340 MB | Run input file                   |
| production.xtc          | ~1.4 GB | Compressed trajectory (1,001 frames) |
| production.edr          | ~93 MB  | Energy file                       |
| production.log          | ~720 KB | GROMACS log                       |
| production.gro          | ~67 MB  | Final coordinates                 |
| production.cpt          | ~36 MB  | Checkpoint (for restart)          |
| production.mdp          | ~2 KB   | MDP parameters                   |
| membrane_analysis.py    | ~6 KB   | MDAnalysis membrane script        |
| 16 × *.xvg files       | various | Analysis data (see plots above)   |

---

## 11. Summary & Conclusions

1. **System stability:** Temperature, pressure, and density are all well-controlled throughout the 10 ns production. The Parrinello-Rahman barostat with semiisotropic coupling maintains membrane geometry correctly.

2. **Protein structure:** The backbone RMSD rises to ~0.88 nm over 10 ns, which is expected given the backmapped starting structure requiring relaxation. Secondary structure (dominated by α-helices, ~890 residues) remains stable, and the intra-protein H-bond count (~1,232) is consistent.

3. **Membrane integrity:** All membrane observables are within expected ranges:
   - APL (69.8 Å²) is appropriate for a mixed POPC/POPE/CHOL/POPS bilayer with a large embedded protein
   - Bilayer thickness (3.93 nm) matches experimental POPC values
   - S_CD order parameters show correct physical profiles for both saturated (sn-1) and unsaturated (sn-2) chains
   - POPC lateral diffusion (5.3 × 10⁻⁸ cm²/s) falls within the experimental range

4. **Limitations:** At 10 ns, the production run is short for a 743K-atom membrane protein system. The rising RMSD suggests the protein has not yet fully equilibrated in the all-atom representation. Longer simulations (100+ ns) would be needed to assess conformational convergence and the functional impact of the R376H mutation on channel gating.

5. **Performance:** The H100 GPU delivered 45.4 ns/day for this 743K-atom system, with the dominant bottleneck being GPU state copy waits (55.6% of wall time). Increasing `nstlist` could improve pair search overhead (19% of runtime).
