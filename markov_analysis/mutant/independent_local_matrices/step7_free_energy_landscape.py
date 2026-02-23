#!/usr/bin/env python
"""
Step 7 — Local Free Energy Landscapes from Stationary Distributions
=====================================================================
MSM pipeline for Mutant Nav1.5 (Martini 3 CG).

Converts the per-block local stationary distributions π_local
(from Step 6) into free energy surfaces using the Boltzmann
relationship:

    ΔG_i,local = −k_B T  ln(π_i)  −  min_j { −k_B T ln(π_j) }

so that the most populated microstate has ΔG = 0 by definition.
Because Block 2 was excluded, the two surfaces (Block 1 / Block 3)
are on **independent** local energy scales; their zeros are defined
separately.

These scalar ΔG values are projected onto four 2D collective-variable
planes to generate contour maps:

  (A) TIC 1  vs  TIC 2        — slowest kinetic modes
  (B) RMSD   vs  R_g          — structural descriptors
  (C) TIC 1  vs  R_g          — kinetic / structural hybrid
  (D) TIC 1  vs  RMSD         — kinetic / structural hybrid

Each CV pair is produced for Block 1 (native/folded basin) and
Block 3 (unfolded/excited basin) independently.

Projection method
-----------------
For every frame f in a block, the assigned microstate label gives
ΔG(f).  The per-frame CV values are already available (TICA) or
computed on-the-fly (RMSD, Rg).  2D free energy surfaces are built
by binning frames into a regular grid on (CV1, CV2), and within
each bin the **Boltzmann-weighted average** ΔG is computed:

    ΔG(bin) = −k_B T ln[ Σ_f∈bin  π(state_f) ]   (renormalised)

which is equivalent to assigning each bin the equilibrium probability
of all microstates whose CV centroids fall in it.

Inputs
------
Step 6:  stationary_distribution/msm_block{1,3}_pi_rev_tau10.npy
         stationary_distribution/msm_block{1,3}_scc_local_map.npy
Step 5:  transition_matrices/msm_block{1,3}_labels_scc.npy
         transition_matrices/scc_state_map.npy
Step 2:  tica/msm_block{1,3}_tica.npy
Step 1:  msm_block{1,3}/rg.npy
Traj:    ../trajectories/msm_block{1,3}.xtc
         ../reference/conf_replica_00.gro

Outputs
-------
free_energy/                — directory with all outputs

Usage
-----
    conda activate bio_env
    python step7_free_energy_landscape.py                 # default
    python step7_free_energy_landscape.py --lag 20
    python step7_free_energy_landscape.py --nbins 80
    python step7_free_energy_landscape.py --temp 310
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
PKG       = os.path.dirname(BASE)                          # mutant_analysis_pkg
TRANS_DIR = os.path.join(BASE, "transition_matrices")
STAT_DIR  = os.path.join(BASE, "stationary_distribution")
TICA_DIR  = os.path.join(BASE, "tica")
OUT_DIR   = os.path.join(BASE, "free_energy")
TOP       = os.path.join(PKG, "reference", "conf_replica_00.gro")
TRAJ_DIR  = BASE

BLOCKS    = ["msm_block1", "msm_block2", "msm_block3"]
DT_PS     = 100.0                              # ps per frame
KB_KJMOL  = 8.314462618e-3                     # kJ/(mol·K)
N_STATES_ORIG = 200
DEFAULT_TEMP  = 310.0                          # K  (physiological)
DEFAULT_NBINS = 60

# Domain definitions (BB bead indices, 0-based)
DOMAINS = {
    "DI":   (0, 450),
    "DII":  (450, 900),
    "DIII": (900, 1350),
    "DIV":  (1350, 2016),
}


# ────────────────────────────────────────────────────────────────
# RMSD COMPUTATION
# ────────────────────────────────────────────────────────────────

def compute_rmsd(block_name, topology=TOP, traj_dir=TRAJ_DIR):
    """Compute BB-RMSD per frame for a trajectory block using MDAnalysis.

    The reference is the first frame of the trajectory.
    Returns an array of shape (n_frames,).
    """
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms, align

    xtc = os.path.join(traj_dir, f"{block_name}.xtc")
    u = mda.Universe(topology, xtc)
    bb = u.select_atoms("name BB")
    n_frames = len(u.trajectory)

    # Store reference positions (first frame)
    u.trajectory[0]
    ref_pos = bb.positions.copy()

    rmsd_arr = np.empty(n_frames, dtype=np.float64)
    for fi, ts in enumerate(u.trajectory):
        # Optimal alignment then RMSD
        pos = bb.positions.copy()
        # Use Kabsch (MDAnalysis alignto is slow per frame; manual is faster)
        # Center both
        ref_c = ref_pos - ref_pos.mean(axis=0)
        pos_c = pos - pos.mean(axis=0)
        # Kabsch rotation
        H = pos_c.T @ ref_c
        U_svd, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U_svd.T)
        sign_mat = np.diag([1, 1, np.sign(d)])
        R = Vt.T @ sign_mat @ U_svd.T
        pos_aligned = pos_c @ R.T
        rmsd_arr[fi] = np.sqrt(np.mean(np.sum((pos_aligned - ref_c) ** 2,
                                                axis=1)))
    return rmsd_arr


# ────────────────────────────────────────────────────────────────
# FREE ENERGY FROM π
# ────────────────────────────────────────────────────────────────

def compute_deltaG(pi, kBT):
    """ΔG_i = −k_B T ln(π_i), shifted so min ΔG = 0."""
    mask = pi > 0
    dG = np.full_like(pi, np.nan, dtype=np.float64)
    dG[mask] = -kBT * np.log(pi[mask])
    dG[mask] -= np.nanmin(dG[mask])
    return dG


# ────────────────────────────────────────────────────────────────
# MAP FRAME-LEVEL ΔG
# ────────────────────────────────────────────────────────────────

def frame_deltaG(labels_scc, scc_local_map, dG_local):
    """Assign a ΔG value to every frame.

    Parameters
    ----------
    labels_scc     : (n_frames,) — microstate index in SCC-197 space
    scc_local_map  : (m_local,)  — which SCC-197 indices are in the
                                    local SCC of this block
    dG_local       : (m_local,)  — free energy of each local SCC state

    Returns
    -------
    dG_frame : (n_frames,) — free energy per frame (NaN if state is
               not in the local SCC)
    valid    : (n_frames,) bool — True where ΔG is assigned
    """
    n_frames = len(labels_scc)
    dG_frame = np.full(n_frames, np.nan, dtype=np.float64)

    # Build lookup:  SCC-197 index → local SCC index
    lookup = {}
    for loc_idx, scc197_idx in enumerate(scc_local_map):
        lookup[int(scc197_idx)] = loc_idx

    for f in range(n_frames):
        s = int(labels_scc[f])
        if s in lookup:
            dG_frame[f] = dG_local[lookup[s]]

    valid = ~np.isnan(dG_frame)
    return dG_frame, valid


# ────────────────────────────────────────────────────────────────
# 2D BINNED FREE ENERGY SURFACE
# ────────────────────────────────────────────────────────────────

def bin_free_energy_2d(cv1, cv2, pi_frame, nbins, kBT, pad=0.02):
    """Build a 2D free energy surface from per-frame π weights.

    Each bin accumulates the stationary probability of its frames,
    then ΔG(bin) = −k_B T ln(Σ π_in_bin).  Bins with zero weight
    are set to NaN.

    Parameters
    ----------
    cv1, cv2   : (N,) — collective variable values for valid frames
    pi_frame   : (N,) — π_i for the microstate of each frame
    nbins      : int
    kBT        : float (kJ/mol)

    Returns
    -------
    dG_grid    : (nbins, nbins) — free energy on a regular grid
    extent     : (x_min, x_max, y_min, y_max)
    x_edges, y_edges : bin edges
    """
    # Determine range with small padding
    x_min, x_max = cv1.min(), cv1.max()
    y_min, y_max = cv2.min(), cv2.max()
    dx = (x_max - x_min) * pad
    dy = (y_max - y_min) * pad
    x_min -= dx; x_max += dx
    y_min -= dy; y_max += dy

    x_edges = np.linspace(x_min, x_max, nbins + 1)
    y_edges = np.linspace(y_min, y_max, nbins + 1)

    # Digitise frames into bins
    ix = np.clip(np.digitize(cv1, x_edges) - 1, 0, nbins - 1)
    iy = np.clip(np.digitize(cv2, y_edges) - 1, 0, nbins - 1)

    # Accumulate probability in each bin
    prob_grid = np.zeros((nbins, nbins), dtype=np.float64)
    for f in range(len(cv1)):
        prob_grid[iy[f], ix[f]] += pi_frame[f]

    # Normalise to sum = 1
    total = prob_grid.sum()
    if total > 0:
        prob_grid /= total

    # Convert to free energy
    dG_grid = np.full_like(prob_grid, np.nan)
    mask = prob_grid > 0
    dG_grid[mask] = -kBT * np.log(prob_grid[mask])
    dG_grid[mask] -= np.nanmin(dG_grid[mask])

    extent = (x_min, x_max, y_min, y_max)
    return dG_grid, extent, x_edges, y_edges


# ────────────────────────────────────────────────────────────────
# PLOTTING
# ────────────────────────────────────────────────────────────────

def plot_fes_panel(dG_grid, extent, xlabel, ylabel, title,
                   fpath, kBT, vmax=None, cmap="viridis_r"):
    """Plot a single 2D FES contour map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if vmax is None:
        vmax_val = np.nanpercentile(dG_grid[~np.isnan(dG_grid)], 98)
    else:
        vmax_val = vmax

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(dG_grid, origin="lower", extent=extent,
                   aspect="auto", cmap=cmap,
                   vmin=0, vmax=vmax_val, interpolation="bilinear")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("ΔG  (kJ/mol)", fontsize=11)

    # Contour lines
    valid = ~np.isnan(dG_grid)
    if np.sum(valid) > 10:
        x_cents = 0.5 * (np.linspace(extent[0], extent[1],
                                      dG_grid.shape[1] + 1)[:-1] +
                          np.linspace(extent[0], extent[1],
                                      dG_grid.shape[1] + 1)[1:])
        y_cents = 0.5 * (np.linspace(extent[2], extent[3],
                                      dG_grid.shape[0] + 1)[:-1] +
                          np.linspace(extent[2], extent[3],
                                      dG_grid.shape[0] + 1)[1:])
        X, Y = np.meshgrid(x_cents, y_cents)
        # Replace NaN with large value for contouring
        dG_filled = np.where(np.isnan(dG_grid), vmax_val * 1.5, dG_grid)
        n_levels = min(12, max(4, int(vmax_val / kBT)))
        levels = np.linspace(0, vmax_val, n_levels + 1)
        ax.contour(X, Y, dG_filled, levels=levels,
                   colors="white", linewidths=0.5, alpha=0.6)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(fpath, dpi=250)
    plt.close(fig)


def plot_1d_fes(cv_vals, dG_frame, valid, xlabel, title, fpath,
                nbins=100):
    """1D free energy profile along one CV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cv = cv_vals[valid]
    dg = dG_frame[valid]

    edges = np.linspace(cv.min(), cv.max(), nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ix = np.clip(np.digitize(cv, edges) - 1, 0, nbins - 1)

    dg_bin = np.full(nbins, np.nan)
    for b in range(nbins):
        mask = ix == b
        if mask.sum() > 0:
            dg_bin[b] = np.mean(dg[mask])  # mean ΔG in bin

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(centres, dg_bin, "-", lw=1.5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("ΔG  (kJ/mol)", fontsize=11)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(fpath, dpi=200)
    plt.close(fig)


def plot_comparison_scatter(block_data, cv_pair, labels, out_dir):
    """Overlay all block FES scatter (ΔG as colour) on same axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_blocks = len(block_data)
    fig, axes = plt.subplots(1, n_blocks, figsize=(6 * n_blocks + 3, 5.5))
    if n_blocks == 1:
        axes = [axes]
    for ax, (bk, bd) in zip(axes, block_data.items()):
        valid = bd["valid"]
        cv1 = bd[cv_pair[0]][valid]
        cv2 = bd[cv_pair[1]][valid]
        dg  = bd["dG_frame"][valid]
        sc = ax.scatter(cv1, cv2, c=dg, s=1.5, alpha=0.5,
                        cmap="viridis_r", vmin=0,
                        vmax=np.nanpercentile(dg, 98))
        fig.colorbar(sc, ax=ax, shrink=0.8, label="ΔG (kJ/mol)")
        ax.set_xlabel(labels[0], fontsize=11)
        ax.set_ylabel(labels[1], fontsize=11)
        bk_num = bk.replace("msm_block", "")
        bk_label = f"Block {bk_num}"
        ax.set_title(f"{bk_label}", fontsize=11, weight="bold")
    fig.suptitle(f"Free Energy:  {labels[0]}  vs  {labels[1]}",
                 fontsize=13, weight="bold", y=1.01)
    fig.tight_layout()
    tag = f"{cv_pair[0]}_vs_{cv_pair[1]}"
    fig.savefig(os.path.join(out_dir, f"comparison_{tag}.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ────────────────────────────────────────────────────────────────

def write_summary(out_dir, block_results, kBT, lag, nbins, temp):
    """Write human-readable summary."""
    fpath = os.path.join(out_dir, f"summary_free_energy_tau{lag}.txt")
    with open(fpath, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("Step 7 — Local Free Energy Landscapes\n")
        f.write(f"Temperature   : {temp:.1f} K\n")
        f.write(f"k_B T         : {kBT:.4f} kJ/mol\n")
        f.write(f"Lag τ         : {lag} frames = {lag * DT_PS / 1000:.1f} ns\n")
        f.write(f"Grid bins     : {nbins} × {nbins}\n")
        f.write("=" * 65 + "\n\n")

        for bk, bd in block_results.items():
            bk_num = bk.replace("msm_block", "")
            bk_label = f"Block {bk_num}"
            f.write(f"──── {bk_label} ────\n\n")
            f.write(f"  Local SCC states      : {bd['n_scc_local']}\n")
            f.write(f"  Frames (total)        : {bd['n_frames']}\n")
            f.write(f"  Frames with ΔG        : {bd['n_valid']}\n")
            f.write(f"  Coverage              : {bd['n_valid']/bd['n_frames']:.1%}\n")
            f.write(f"  ΔG range              : [{bd['dG_min']:.2f}, "
                    f"{bd['dG_max']:.2f}] kJ/mol\n")
            f.write(f"  ΔG range              : [{bd['dG_min']/kBT:.1f}, "
                    f"{bd['dG_max']/kBT:.1f}] k_BT\n")
            f.write(f"  Most stable state     : SCC-197 #{bd['most_stable']}  "
                    f"(orig #{bd['most_stable_orig']})\n")
            f.write(f"  Least stable state    : SCC-197 #{bd['least_stable']}  "
                    f"(orig #{bd['least_stable_orig']})\n\n")

            # Top/bottom states
            f.write(f"  Top 10 most stable (ΔG → 0):\n")
            f.write(f"  {'Rank':>4s}  {'SCC197':>6s}  {'Orig':>5s}  "
                    f"{'ΔG(kJ/mol)':>11s}  {'ΔG(k_BT)':>9s}  {'π_i':>10s}\n")
            for rank, (idx, g, p) in enumerate(bd["top_stable"][:10], 1):
                f.write(f"  {rank:4d}  {idx:6d}  {bd['scc_map_agg'][idx]:5d}  "
                        f"{g:11.3f}  {g / kBT:9.2f}  {p:10.6e}\n")

            f.write(f"\n  Bottom 10 least stable:\n")
            for rank, (idx, g, p) in enumerate(bd["bot_stable"][:10], 1):
                f.write(f"  {rank:4d}  {idx:6d}  {bd['scc_map_agg'][idx]:5d}  "
                        f"{g:11.3f}  {g / kBT:9.2f}  {p:10.6e}\n")
            f.write("\n")

        f.write("=" * 65 + "\n")
    print(f"  Summary → {fpath}")


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 7: Local free energy landscapes")
    parser.add_argument("--lag", type=int, default=10,
                        help="Lag time in frames (default: 10)")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP,
                        help="Temperature in K (default: 310)")
    parser.add_argument("--nbins", type=int, default=DEFAULT_NBINS,
                        help="Number of bins per CV axis (default: 60)")
    args = parser.parse_args()

    lag   = args.lag
    temp  = args.temp
    nbins = args.nbins
    kBT   = KB_KJMOL * temp     # kJ/mol

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    print("=" * 65)
    print("Step 7 — Local Free Energy Landscapes")
    print(f"T = {temp:.1f} K     k_BT = {kBT:.4f} kJ/mol")
    print(f"Lag τ = {lag} frames = {lag * DT_PS / 1000:.1f} ns")
    print(f"Grid  = {nbins} × {nbins}")
    print("=" * 65)

    # Aggregated SCC state map (SCC-197 idx → original 0..199)
    scc_map_agg = np.load(os.path.join(TRANS_DIR, "scc_state_map.npy"))

    # ── Process each block ──────────────────────────────────────
    block_results = {}
    block_data_for_comparison = {}

    for bk in BLOCKS:
        bk_num = bk.replace("msm_block", "")
        bk_label = f"Block {bk_num}"
        print(f"\n{'─'*65}")
        print(f"  {bk_label}")
        print(f"{'─'*65}")

        # Load per-block π and SCC map
        pi_local  = np.load(os.path.join(
            STAT_DIR, f"{bk}_pi_rev_tau{lag}.npy"))
        scc_local = np.load(os.path.join(
            STAT_DIR, f"{bk}_scc_local_map.npy"))
        # scc_local[j] = index in SCC-197 space for local SCC state j

        # Load SCC-197 frame labels
        labels_scc = np.load(os.path.join(
            TRANS_DIR, f"{bk}_labels_scc.npy"))
        n_frames = len(labels_scc)

        # Load TICA projections (TIC1, TIC2)
        tica = np.load(os.path.join(TICA_DIR, f"{bk}_tica.npy"))
        tic1 = tica[:, 0]
        tic2 = tica[:, 1]

        # Load Rg (col 0 = total Rg in Å)
        rg_all = np.load(os.path.join(BASE, bk, "rg.npy"))
        rg     = rg_all[:, 0]

        # Compute RMSD (BB backbone, aligned to first frame)
        print(f"  Computing BB-RMSD for {bk} …")
        rmsd_arr = compute_rmsd(bk, topology=TOP, traj_dir=TRAJ_DIR)
        print(f"    RMSD range: [{rmsd_arr.min():.2f}, {rmsd_arr.max():.2f}] Å")

        # Compute ΔG per local SCC state
        dG_local = compute_deltaG(pi_local, kBT)
        print(f"  Local SCC states: {len(pi_local)}")
        print(f"  ΔG range: [{np.nanmin(dG_local):.2f}, "
              f"{np.nanmax(dG_local):.2f}] kJ/mol  "
              f"({np.nanmax(dG_local) / kBT:.1f} k_BT)")

        # Assign ΔG to every frame
        dG_frame, valid = frame_deltaG(labels_scc, scc_local, dG_local)
        n_valid = valid.sum()
        print(f"  Frames: {n_frames}  with ΔG: {n_valid}  "
              f"({n_valid / n_frames:.1%})")

        # Also compute π_frame for Boltzmann-weighted binning
        # Build lookup: SCC-197 → local SCC idx
        lookup = {}
        for loc_idx, scc197_idx in enumerate(scc_local):
            lookup[int(scc197_idx)] = loc_idx
        pi_frame = np.zeros(n_frames, dtype=np.float64)
        for f in range(n_frames):
            s = int(labels_scc[f])
            if s in lookup:
                pi_frame[f] = pi_local[lookup[s]]

        # Save per-block arrays
        np.save(os.path.join(OUT_DIR, f"{bk}_dG_frame_tau{lag}.npy"),
                dG_frame)
        np.save(os.path.join(OUT_DIR, f"{bk}_rmsd.npy"), rmsd_arr)

        # ── 2D FES contour maps ────────────────────────────────
        bk_plot_dir = os.path.join(OUT_DIR, bk)
        os.makedirs(bk_plot_dir, exist_ok=True)

        cv_pairs = [
            ("tic1", "tic2",  tic1,     tic2,     "TIC 1", "TIC 2"),
            ("rmsd", "rg",    rmsd_arr, rg,       "RMSD (Å)", "Rg (Å)"),
            ("tic1", "rg",    tic1,     rg,       "TIC 1", "Rg (Å)"),
            ("tic1", "rmsd",  tic1,     rmsd_arr, "TIC 1", "RMSD (Å)"),
        ]

        for cv1_tag, cv2_tag, cv1_full, cv2_full, lab1, lab2 in cv_pairs:
            cv1_v = cv1_full[valid]
            cv2_v = cv2_full[valid]
            pi_v  = pi_frame[valid]

            dG_grid, extent, xe, ye = bin_free_energy_2d(
                cv1_v, cv2_v, pi_v, nbins, kBT)

            tag = f"{cv1_tag}_vs_{cv2_tag}"
            fpath = os.path.join(bk_plot_dir, f"fes_{tag}.png")
            plot_fes_panel(
                dG_grid, extent, lab1, lab2,
                f"{bk_label}\nΔG({lab1}, {lab2})",
                fpath, kBT)

            # Save grid
            np.savez(os.path.join(bk_plot_dir, f"fes_{tag}.npz"),
                     dG_grid=dG_grid, x_edges=xe, y_edges=ye,
                     extent=extent)

        # ── 1D free energy profiles ────────────────────────────
        for cv_tag, cv_full, lab in [("tic1", tic1, "TIC 1"),
                                      ("tic2", tic2, "TIC 2"),
                                      ("rmsd", rmsd_arr, "RMSD (Å)"),
                                      ("rg", rg, "Rg (Å)")]:
            plot_1d_fes(cv_full, dG_frame, valid, lab,
                        f"{bk_label} — ΔG vs {lab}",
                        os.path.join(bk_plot_dir, f"fes_1d_{cv_tag}.png"))

        print(f"  Plots → {bk_plot_dir}/")

        # Store for comparison and summary
        most_stable = scc_local[np.argmin(dG_local)]
        least_stable = scc_local[np.nanargmax(dG_local)]
        sorted_idx = np.argsort(dG_local)
        top_stable = [(int(scc_local[i]), float(dG_local[i]),
                        float(pi_local[i])) for i in sorted_idx]
        bot_stable = [(int(scc_local[i]), float(dG_local[i]),
                        float(pi_local[i]))
                       for i in sorted_idx[::-1]]

        block_results[bk] = {
            "n_scc_local":     len(pi_local),
            "n_frames":        n_frames,
            "n_valid":         int(n_valid),
            "dG_min":          float(np.nanmin(dG_local)),
            "dG_max":          float(np.nanmax(dG_local)),
            "most_stable":     int(most_stable),
            "most_stable_orig": int(scc_map_agg[most_stable]),
            "least_stable":    int(least_stable),
            "least_stable_orig": int(scc_map_agg[least_stable]),
            "top_stable":      top_stable,
            "bot_stable":      bot_stable,
            "scc_map_agg":     scc_map_agg,
        }

        block_data_for_comparison[bk] = {
            "tic1":     tic1,
            "tic2":     tic2,
            "rmsd":     rmsd_arr,
            "rg":       rg,
            "dG_frame": dG_frame,
            "valid":    valid,
        }

    # ── Comparison plots (side-by-side) ─────────────────────────
    print(f"\n{'─'*65}")
    print("  Side-by-side comparison plots")
    print(f"{'─'*65}")
    comp_dir = os.path.join(OUT_DIR, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    for cv_pair, labels in [
        (("tic1", "tic2"), ("TIC 1", "TIC 2")),
        (("rmsd", "rg"),   ("RMSD (Å)", "Rg (Å)")),
        (("tic1", "rg"),   ("TIC 1", "Rg (Å)")),
        (("tic1", "rmsd"), ("TIC 1", "RMSD (Å)")),
    ]:
        plot_comparison_scatter(block_data_for_comparison, cv_pair,
                                labels, comp_dir)
    print(f"  Comparison plots → {comp_dir}/")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Writing summary")
    print(f"{'─'*65}")
    write_summary(OUT_DIR, block_results, kBT, lag, nbins, temp)

    # Cleanup probe
    probe = os.path.join(BASE, "_probe_dims.py")
    if os.path.exists(probe):
        os.remove(probe)

    dt = time.time() - t0
    print(f"\n{'='*65}")
    print(f"Step 7 complete  ({dt:.1f} s)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
