#!/usr/bin/env python
"""
Step 4 — Local Transition Count Matrices
==========================================
MSM pipeline for Mutant Nav1.5 (Martini 3 CG).

Constructs per-block empirical transition count matrices C(τ) by counting
transitions from microstate S_i at time t to S_j at time t+τ strictly
within each isolated trajectory block:

  C_ij(τ) = Σ_t  δ(a(t) ∈ S_i) · δ(a(t+τ) ∈ S_j)

Transitions across block boundaries are explicitly excluded.

The script then:
  1. Builds C(τ) for each block at the chosen lag time.
  2. Performs an implied-timescale scan over a range of lag times
     to verify the Markov property (timescales should plateau).
  3. Estimates row-stochastic transition matrices T(τ) from the
     count matrices (both non-reversible and reversible MLE).
  4. Computes eigenvalues and implied timescales of each local T(τ)
     as well as for the aggregated (summed) count matrix.
  5. Reports connectivity, sparsity, and count statistics per block.

Inputs:   clustering/msm_block{1,2,3}_labels.npy  (from Step 3)
Outputs:  count_matrices/                           (all outputs)

Usage:
    conda activate bio_env
    python step4_count_matrices.py                    # default lag=10
    python step4_count_matrices.py --lag 20           # custom lag
    python step4_count_matrices.py --no-scan          # skip lag scan
    python step4_count_matrices.py --lag 10 --scan-lags 1,2,5,10,20,50
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
CLUST_DIR   = os.path.join(BASE, "clustering")
BLOCKS      = ["msm_block1", "msm_block2", "msm_block3"]
DT_PS       = 100.0                      # trajectory stride (ps / frame)

DEFAULT_LAG       = 10                   # frames → 1.0 ns
DEFAULT_N_STATES  = 200                  # total microstates from Step 3
DEFAULT_N_EIGEN   = 10                   # eigenvalues to track

# Lag-time scan for implied-timescale convergence
LAG_SCAN_DEFAULT  = [1, 2, 5, 10, 15, 20, 30, 50, 75, 100]


# ────────────────────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────────────────────

def load_block_labels(clust_dir, block_names):
    """Load per-block discrete microstate sequences."""
    labels = []
    for bn in block_names:
        fpath = os.path.join(clust_dir, f"{bn}_labels.npy")
        bl = np.load(fpath).astype(np.int32)
        labels.append(bl)
        print(f"  {bn:12s}  {len(bl):6d} frames,  "
              f"{len(np.unique(bl)):4d} unique states")
    return labels


# ────────────────────────────────────────────────────────────────
# COUNT MATRIX CONSTRUCTION
# ────────────────────────────────────────────────────────────────

def build_count_matrix(label_seq, n_states, lag):
    """Build a transition count matrix for a single block.

    C_ij(τ) = number of times the trajectory goes from state i at t
              to state j at t + τ, within this block only.

    Parameters
    ----------
    label_seq : ndarray, shape (T,), int
        Microstate labels for one trajectory block.
    n_states : int
        Total number of microstates (matrix dimension).
    lag : int
        Lag time in frames.

    Returns
    -------
    C : ndarray, shape (n_states, n_states), int64
        Transition count matrix.
    """
    T = len(label_seq)
    if lag >= T:
        return np.zeros((n_states, n_states), dtype=np.int64)

    C = np.zeros((n_states, n_states), dtype=np.int64)
    i_states = label_seq[:-lag]
    j_states = label_seq[lag:]
    # Vectorised counting using np.add.at
    np.add.at(C, (i_states, j_states), 1)
    return C


def count_matrix_stats(C, block_name=""):
    """Compute and print statistics for a count matrix."""
    total    = C.sum()
    nonzero  = np.count_nonzero(C)
    n        = C.shape[0]
    sparsity = 1.0 - nonzero / (n * n)

    # Off-diagonal
    C_nodiag = C.copy()
    np.fill_diagonal(C_nodiag, 0)
    offdiag_total = C_nodiag.sum()
    diag_total    = np.trace(C)

    # Row sums
    row_sums = C.sum(axis=1)
    visited  = np.sum(row_sums > 0)

    # Connectivity
    C_sym = C + C.T
    G = csr_matrix((C_sym > 0).astype(int))
    n_cc, _ = connected_components(G, directed=False)
    G_dir = csr_matrix((C > 0).astype(int))
    n_scc, scc_labels = connected_components(G_dir, directed=True,
                                              connection="strong")
    scc_sizes = np.bincount(scc_labels)

    stats = {
        "total_counts":      int(total),
        "nonzero_entries":   int(nonzero),
        "sparsity":          sparsity,
        "diagonal_counts":   int(diag_total),
        "offdiag_counts":    int(offdiag_total),
        "self_transition_frac": diag_total / max(total, 1),
        "visited_states":    int(visited),
        "n_connected_comp":  n_cc,
        "n_strongly_conn":   n_scc,
        "largest_scc":       int(scc_sizes.max()),
    }

    prefix = f"  [{block_name}]  " if block_name else "  "
    print(f"{prefix}Total counts       : {total:,}")
    print(f"{prefix}Non-zero entries   : {nonzero:,}  "
          f"(sparsity {sparsity:.4f})")
    print(f"{prefix}Diagonal (self)    : {diag_total:,}  "
          f"({stats['self_transition_frac']:.1%})")
    print(f"{prefix}Off-diagonal       : {offdiag_total:,}")
    print(f"{prefix}Visited states     : {visited} / {n}")
    print(f"{prefix}Connected comp.    : {n_cc}  "
          f"(largest SCC: {scc_sizes.max()})")

    return stats


# ────────────────────────────────────────────────────────────────
# TRANSITION MATRIX ESTIMATION
# ────────────────────────────────────────────────────────────────

def row_normalise(C):
    """Row-stochastic transition matrix T from count matrix C.

    T_ij = C_ij / Σ_j C_ij .  Rows with zero counts map to uniform.
    """
    C_f = C.astype(np.float64)
    row_sums = C_f.sum(axis=1)
    zero_rows = row_sums == 0
    row_sums[zero_rows] = 1.0               # avoid division by zero
    T = C_f / row_sums[:, None]
    # Unvisited states → uniform (convention; they won't affect
    # eigenvalues of the active subspace)
    T[zero_rows, :] = 1.0 / C.shape[0]
    return T


def symmetrise_count_matrix(C):
    """Reversible symmetrisation:  C_sym = C + C^T."""
    return C + C.T


def estimate_reversible_T(C):
    """MLE reversible transition matrix via simple symmetrisation.

    T_rev = row_normalise(C + C^T).

    This is the simplest reversible estimator.  For production MSM
    one would use the maximum-likelihood estimator (e.g. Trendelkamp-
    Schroer et al., 2015), but the symmetrised estimator is standard
    for initial pipeline validation.
    """
    C_sym = symmetrise_count_matrix(C)
    return row_normalise(C_sym)


# ────────────────────────────────────────────────────────────────
# EIGENANALYSIS & IMPLIED TIMESCALES
# ────────────────────────────────────────────────────────────────

def compute_eigenvalues(T, n_eigen):
    """Leading eigenvalues of T (real parts, magnitude-sorted).

    For the reversible estimator T_sym is a real symmetric matrix after
    similarity transform, so eigenvalues are real.  For non-reversible T
    eigenvalues may be complex; we take real parts.
    """
    from scipy.linalg import eigvals
    ev = eigvals(T)
    ev_real = np.real(ev)
    idx = np.argsort(-ev_real)
    return ev_real[idx[:n_eigen]]


def implied_timescales(eigenvalues, lag, dt_ps=100.0):
    """Implied timescales from eigenvalues.

    t_i = -τ / ln(λ_i)   [in ns]

    τ is in frames; dt_ps converts frames → ps → ns.
    """
    ts = np.full_like(eigenvalues, np.nan, dtype=np.float64)
    # Skip λ_1 = 1 (stationary); compute for λ_2, λ_3, ...
    for i in range(1, len(eigenvalues)):
        lam = eigenvalues[i]
        if 0 < lam < 1:
            ts[i] = -lag * dt_ps / (1000.0 * np.log(lam))
        elif lam >= 1.0:
            ts[i] = np.inf
    return ts


# ────────────────────────────────────────────────────────────────
# LAG-TIME SCAN
# ────────────────────────────────────────────────────────────────

def lag_scan(block_labels, n_states, lag_values, n_eigen, dt_ps,
             reversible=True):
    """Scan lag times and compute implied timescales.

    For each lag τ:
      1. Build per-block C(τ).
      2. Sum across blocks → C_agg(τ).
      3. Estimate T (reversible or non-reversible).
      4. Compute eigenvalues → implied timescales.

    Also does the same per-block for individual checks.

    Returns
    -------
    results : dict
        "lags"            : array of lags
        "agg_eigenvalues" : (n_lags, n_eigen)
        "agg_timescales"  : (n_lags, n_eigen)  in ns
        "block_eigenvalues": dict[block] → (n_lags, n_eigen)
        "block_timescales" : dict[block] → (n_lags, n_eigen)
    """
    n_lags = len(lag_values)
    block_names = [f"block{i+1}" for i in range(len(block_labels))]

    agg_ev = np.zeros((n_lags, n_eigen))
    agg_ts = np.full((n_lags, n_eigen), np.nan)
    blk_ev = {bn: np.zeros((n_lags, n_eigen)) for bn in block_names}
    blk_ts = {bn: np.full((n_lags, n_eigen), np.nan) for bn in block_names}

    for li, lag in enumerate(lag_values):
        # Per-block counts
        block_Cs = []
        skip = False
        for bi, bl in enumerate(block_labels):
            if lag >= len(bl) // 2:
                print(f"  lag={lag}: block {bi+1} too short "
                      f"({len(bl)} frames), skipping this lag")
                skip = True
                break
            C = build_count_matrix(bl, n_states, lag)
            block_Cs.append(C)

        if skip:
            agg_ev[li, :] = np.nan
            agg_ts[li, :] = np.nan
            for bn in block_names:
                blk_ev[bn][li, :] = np.nan
                blk_ts[bn][li, :] = np.nan
            continue

        # Aggregated count matrix
        C_agg = sum(block_Cs)

        if reversible:
            T_agg = estimate_reversible_T(C_agg)
        else:
            T_agg = row_normalise(C_agg)

        ev = compute_eigenvalues(T_agg, n_eigen)
        ts = implied_timescales(ev, lag, dt_ps)
        agg_ev[li, :] = ev
        agg_ts[li, :] = ts

        # Per-block
        for bi, (C_b, bn) in enumerate(zip(block_Cs, block_names)):
            if reversible:
                T_b = estimate_reversible_T(C_b)
            else:
                T_b = row_normalise(C_b)
            ev_b = compute_eigenvalues(T_b, n_eigen)
            ts_b = implied_timescales(ev_b, lag, dt_ps)
            blk_ev[bn][li, :] = ev_b
            blk_ts[bn][li, :] = ts_b

        # Print summary line
        top3_ts = agg_ts[li, 1:4]
        top3_str = ", ".join(
            f"{t:.1f}" if np.isfinite(t) else "inf" for t in top3_ts)
        print(f"  lag = {lag:4d} frames ({lag * dt_ps / 1000:.1f} ns)  "
              f" →  top-3 ITS (ns): {top3_str}")

    return {
        "lags":              np.array(lag_values),
        "agg_eigenvalues":   agg_ev,
        "agg_timescales":    agg_ts,
        "block_eigenvalues": blk_ev,
        "block_timescales":  blk_ts,
    }


# ────────────────────────────────────────────────────────────────
# PLOTTING
# ────────────────────────────────────────────────────────────────

def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  [WARN] matplotlib not available — skipping plots")
        return None


def plot_implied_timescales(scan, n_show, outdir, tag="aggregated"):
    """Implied-timescale convergence plot."""
    plt = _get_plt()
    if plt is None:
        return

    lags_ns = scan["lags"] * DT_PS / 1000.0
    ts = scan["agg_timescales"] if tag == "aggregated" else scan

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = plt.cm.tab10(np.linspace(0, 1, n_show))

    for i in range(1, min(n_show + 1, ts.shape[1])):
        valid = np.isfinite(ts[:, i])
        if valid.any():
            ax.plot(lags_ns[valid], ts[valid, i], "o-",
                    color=colours[i - 1], linewidth=2, markersize=5,
                    label=f"ITS {i}")

    # Diagonal reference line: t_implied = lag
    ax.plot(lags_ns, lags_ns, "k--", alpha=0.3, label="$t = \\tau$")

    ax.set_xlabel("Lag time τ (ns)", fontsize=12)
    ax.set_ylabel("Implied timescale (ns)", fontsize=12)
    ax.set_title(f"Implied Timescales ({tag}) — Mutant Nav1.5", fontsize=14)
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    fname = f"its_convergence_{tag}.png"
    fpath = os.path.join(outdir, fname)
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved ITS plot → {fpath}")


def plot_implied_timescales_per_block(scan, n_show, outdir):
    """Per-block implied timescale comparison."""
    plt = _get_plt()
    if plt is None:
        return

    block_names = sorted(scan["block_timescales"].keys())
    n_blocks = len(block_names)

    fig, axes = plt.subplots(1, n_blocks, figsize=(6 * n_blocks, 5),
                             sharey=True)
    if n_blocks == 1:
        axes = [axes]

    lags_ns = scan["lags"] * DT_PS / 1000.0
    colours = plt.cm.tab10(np.linspace(0, 1, n_show))

    for ax, bn in zip(axes, block_names):
        ts = scan["block_timescales"][bn]
        for i in range(1, min(n_show + 1, ts.shape[1])):
            valid = np.isfinite(ts[:, i])
            if valid.any():
                ax.plot(lags_ns[valid], ts[valid, i], "o-",
                        color=colours[i - 1], linewidth=1.5, markersize=4,
                        label=f"ITS {i}")
        ax.plot(lags_ns, lags_ns, "k--", alpha=0.3)
        ax.set_xlabel("Lag time τ (ns)", fontsize=11)
        ax.set_title(bn, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    axes[0].set_ylabel("Implied timescale (ns)", fontsize=11)
    axes[-1].legend(fontsize=8, ncol=2, loc="upper left")
    fig.suptitle("Per-Block Implied Timescales — Mutant Nav1.5", fontsize=14)
    plt.tight_layout()
    fpath = os.path.join(outdir, "its_convergence_per_block.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved per-block ITS → {fpath}")


def plot_count_matrix_heatmap(C, outdir, tag="aggregated"):
    """Log-scale heatmap of the count matrix."""
    plt = _get_plt()
    if plt is None:
        return

    C_plot = C.astype(np.float64).copy()
    C_plot[C_plot == 0] = np.nan
    C_log = np.log10(C_plot)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(C_log, origin="lower", cmap="viridis", aspect="auto",
                   interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, label="log₁₀(counts)")
    ax.set_xlabel("State j (destination)", fontsize=11)
    ax.set_ylabel("State i (source)", fontsize=11)
    ax.set_title(f"Count Matrix C(τ) [{tag}] — Mutant Nav1.5", fontsize=13)
    plt.tight_layout()
    fpath = os.path.join(outdir, f"count_matrix_{tag}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved count matrix heatmap → {fpath}")


def plot_eigenvalue_spectrum(eigenvalues, lag, outdir, tag="aggregated"):
    """Bar chart of leading eigenvalues."""
    plt = _get_plt()
    if plt is None:
        return

    n = len(eigenvalues)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, n + 1), eigenvalues, color="steelblue",
           edgecolor="white", width=0.8)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Eigenvalue index", fontsize=12)
    ax.set_ylabel("λ", fontsize=12)
    ax.set_title(f"MSM Eigenvalue Spectrum (τ={lag}, {tag}) — Mutant Nav1.5",
                 fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fpath = os.path.join(outdir, f"eigenvalue_spectrum_{tag}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved eigenvalue spectrum → {fpath}")


def plot_stationary_distribution(T, outdir, tag="aggregated"):
    """Stationary distribution π from the left eigenvector of T."""
    plt = _get_plt()
    if plt is None:
        return

    from scipy.linalg import eig
    ev, vl = eig(T, left=True, right=False)
    # Find eigenvector for λ closest to 1
    idx = np.argmin(np.abs(ev - 1.0))
    pi = np.real(vl[:, idx])
    pi = pi / pi.sum()
    pi = np.abs(pi)  # ensure non-negative

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(pi)), np.sort(pi)[::-1], color="darkorange",
           edgecolor="none", width=1.0)
    ax.set_xlabel("State rank", fontsize=12)
    ax.set_ylabel("Stationary probability π", fontsize=12)
    ax.set_title(f"Stationary Distribution ({tag}) — Mutant Nav1.5", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fpath = os.path.join(outdir, f"stationary_dist_{tag}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved stationary distribution → {fpath}")

    return pi


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 4 — Local transition count matrices")
    parser.add_argument("--lag",       type=int,   default=DEFAULT_LAG,
                        help="Lag time in frames (default: %(default)s)")
    parser.add_argument("--n-states",  type=int,   default=DEFAULT_N_STATES,
                        help="Number of microstates (default: %(default)s)")
    parser.add_argument("--n-eigen",   type=int,   default=DEFAULT_N_EIGEN,
                        help="Eigenvalues to track (default: %(default)s)")
    parser.add_argument("--no-scan",   action="store_true",
                        help="Skip lag-time scan")
    parser.add_argument("--scan-lags", type=str,   default=None,
                        help="Comma-separated lag values for scan")
    parser.add_argument("--reversible", action="store_true", default=True,
                        help="Use reversible (symmetrised) estimator (default)")
    parser.add_argument("--no-reversible", dest="reversible",
                        action="store_false",
                        help="Use non-reversible (raw row-normalised) estimator")
    args = parser.parse_args()

    lag       = args.lag
    n_states  = args.n_states
    n_eigen   = args.n_eigen
    rev       = args.reversible

    if args.scan_lags is not None:
        scan_lags = [int(x.strip()) for x in args.scan_lags.split(",")]
    else:
        scan_lags = LAG_SCAN_DEFAULT

    OUT = os.path.join(BASE, "count_matrices")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 70)
    print("  Step 4 — Local Transition Count Matrices  |  Mutant Nav1.5  (CG)")
    print("=" * 70)
    print(f"  Lag time         : {lag} frames ({lag * DT_PS / 1000:.2f} ns)")
    print(f"  Microstates      : {n_states}")
    print(f"  Eigenvalues      : {n_eigen}")
    print(f"  Reversible       : {'yes' if rev else 'no'}")
    print(f"  Lag scan         : {'no' if args.no_scan else scan_lags}")
    print(f"  Output dir       : {OUT}")

    # ── 1. Load per-block labels ──
    print("\n[1] Loading per-block microstate labels …")
    block_labels = load_block_labels(CLUST_DIR, BLOCKS)
    block_sizes = [len(bl) for bl in block_labels]

    # ── 2. Build per-block count matrices at chosen lag ──
    print(f"\n[2] Building per-block C(τ={lag}) …")
    block_Cs = []
    all_stats = []
    for bn, bl in zip(BLOCKS, block_labels):
        t0 = time.time()
        C = build_count_matrix(bl, n_states, lag)
        dt = time.time() - t0
        block_Cs.append(C)
        print(f"\n  --- {bn} (τ={lag}) ---  [{dt:.3f} s]")
        stats = count_matrix_stats(C, bn)
        all_stats.append(stats)

        # Save per-block count matrix
        fpath = os.path.join(OUT, f"{bn}_C_tau{lag}.npy")
        np.save(fpath, C)

    # ── 3. Aggregated count matrix ──
    print(f"\n[3] Aggregated count matrix C_agg(τ={lag}) …")
    C_agg = sum(block_Cs)
    fpath = os.path.join(OUT, f"C_agg_tau{lag}.npy")
    np.save(fpath, C_agg)
    agg_stats = count_matrix_stats(C_agg, "aggregated")

    # Symmetrised
    C_agg_sym = symmetrise_count_matrix(C_agg)
    fpath = os.path.join(OUT, f"C_agg_sym_tau{lag}.npy")
    np.save(fpath, C_agg_sym)

    # ── 4. Transition matrix estimation ──
    print(f"\n[4] Estimating transition matrices …")
    if rev:
        T_agg = estimate_reversible_T(C_agg)
        est_label = "reversible (symmetrised)"
    else:
        T_agg = row_normalise(C_agg)
        est_label = "non-reversible (row-normalised)"
    print(f"  Estimator: {est_label}")

    fpath = os.path.join(OUT, f"T_agg_tau{lag}.npy")
    np.save(fpath, T_agg)

    # Per-block T
    block_Ts = []
    for bn, C in zip(BLOCKS, block_Cs):
        if rev:
            T_b = estimate_reversible_T(C)
        else:
            T_b = row_normalise(C)
        block_Ts.append(T_b)
        np.save(os.path.join(OUT, f"{bn}_T_tau{lag}.npy"), T_b)

    # ── 5. Eigenanalysis ──
    print(f"\n[5] Eigenanalysis of T(τ={lag}) …")

    # Aggregated
    ev_agg = compute_eigenvalues(T_agg, n_eigen)
    ts_agg = implied_timescales(ev_agg, lag, DT_PS)
    np.save(os.path.join(OUT, f"eigenvalues_agg_tau{lag}.npy"), ev_agg)

    print(f"\n  Aggregated T — leading eigenvalues & implied timescales:")
    print(f"  {'#':>3s}  {'λ':>10s}  {'ITS (ns)':>12s}")
    print(f"  {'-'*3:>3s}  {'-'*10:>10s}  {'-'*12:>12s}")
    for i in range(n_eigen):
        ts_str = (f"{ts_agg[i]:.2f}" if np.isfinite(ts_agg[i])
                  else ("∞" if ts_agg[i] == np.inf else "—"))
        print(f"  {i+1:3d}  {ev_agg[i]:10.6f}  {ts_str:>12s}")

    # Per-block
    for bn, T_b in zip(BLOCKS, block_Ts):
        ev_b = compute_eigenvalues(T_b, n_eigen)
        ts_b = implied_timescales(ev_b, lag, DT_PS)
        np.save(os.path.join(OUT, f"eigenvalues_{bn}_tau{lag}.npy"), ev_b)
        print(f"\n  {bn} — top-3 ITS (ns): ", end="")
        top3 = [f"{ts_b[i]:.1f}" if np.isfinite(ts_b[i]) else "—"
                for i in range(1, min(4, n_eigen))]
        print(", ".join(top3))

    # ── 6. Lag-time scan ──
    if not args.no_scan:
        print(f"\n[6] Lag-time scan for implied-timescale convergence …")
        min_block = min(len(bl) for bl in block_labels)
        valid_lags = [l for l in scan_lags if l < min_block // 2]
        if lag not in valid_lags:
            valid_lags.append(lag)
            valid_lags.sort()
        print(f"  Lag values: {valid_lags}")

        scan = lag_scan(block_labels, n_states, valid_lags, n_eigen,
                        DT_PS, reversible=rev)

        np.savez(os.path.join(OUT, "lag_scan_results.npz"),
                 lags=scan["lags"],
                 agg_eigenvalues=scan["agg_eigenvalues"],
                 agg_timescales=scan["agg_timescales"])

        # Also save per-block scan
        for bn in scan["block_timescales"]:
            np.savez(os.path.join(OUT, f"lag_scan_{bn}.npz"),
                     lags=scan["lags"],
                     eigenvalues=scan["block_eigenvalues"][bn],
                     timescales=scan["block_timescales"][bn])

        # Plot
        plot_implied_timescales(scan, min(n_eigen - 1, 6), OUT)
        plot_implied_timescales_per_block(scan, min(n_eigen - 1, 6), OUT)
    else:
        print("\n[6] Lag-time scan skipped (--no-scan)")

    # ── 7. Diagnostic plots ──
    print(f"\n[7] Generating diagnostic plots …")
    plot_count_matrix_heatmap(C_agg, OUT, "aggregated")
    for bn, C in zip(BLOCKS, block_Cs):
        plot_count_matrix_heatmap(C, OUT, bn)
    plot_eigenvalue_spectrum(ev_agg, lag, OUT, "aggregated")
    pi = plot_stationary_distribution(T_agg, OUT, "aggregated")

    if pi is not None:
        np.save(os.path.join(OUT, f"stationary_dist_tau{lag}.npy"), pi)

    # ── 8. Summary ──
    print(f"\n[8] Writing summary …")
    summary_path = os.path.join(OUT, "count_matrix_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Step 4 — Local Transition Count Matrices Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Lag time           : {lag} frames ({lag * DT_PS / 1000:.2f} ns)\n")
        f.write(f"Microstates        : {n_states}\n")
        f.write(f"Estimator          : {est_label}\n")
        f.write(f"Total frames       : {sum(block_sizes)}\n\n")

        f.write("Per-block statistics:\n")
        for bn, s, bs in zip(BLOCKS, all_stats, block_sizes):
            f.write(f"\n  {bn} ({bs} frames):\n")
            f.write(f"    Total counts       : {s['total_counts']:,}\n")
            f.write(f"    Non-zero entries   : {s['nonzero_entries']:,}\n")
            f.write(f"    Sparsity           : {s['sparsity']:.4f}\n")
            f.write(f"    Self-transition %  : {s['self_transition_frac']:.1%}\n")
            f.write(f"    Visited states     : {s['visited_states']}\n")
            f.write(f"    Connected comp.    : {s['n_connected_comp']}\n")
            f.write(f"    Largest SCC        : {s['largest_scc']}\n")

        f.write(f"\nAggregated statistics:\n")
        f.write(f"  Total counts       : {agg_stats['total_counts']:,}\n")
        f.write(f"  Non-zero entries   : {agg_stats['nonzero_entries']:,}\n")
        f.write(f"  Sparsity           : {agg_stats['sparsity']:.4f}\n")
        f.write(f"  Self-transition %  : {agg_stats['self_transition_frac']:.1%}\n")
        f.write(f"  Visited states     : {agg_stats['visited_states']}\n")
        f.write(f"  Connected comp.    : {agg_stats['n_connected_comp']}\n")
        f.write(f"  Largest SCC        : {agg_stats['largest_scc']}\n")

        f.write(f"\nAggregated eigenvalues & implied timescales (τ={lag}):\n")
        f.write(f"  {'#':>3s}  {'λ':>12s}  {'ITS (ns)':>12s}\n")
        for i in range(n_eigen):
            ts_str = (f"{ts_agg[i]:.2f}" if np.isfinite(ts_agg[i])
                      else ("inf" if ts_agg[i] == np.inf else "nan"))
            f.write(f"  {i+1:3d}  {ev_agg[i]:12.6f}  {ts_str:>12s}\n")

        f.write(f"\nOutputs in {OUT}/:\n")
        for bn in BLOCKS:
            f.write(f"  {bn}_C_tau{lag}.npy       — per-block count matrix\n")
            f.write(f"  {bn}_T_tau{lag}.npy       — per-block transition matrix\n")
        f.write(f"  C_agg_tau{lag}.npy            — aggregated count matrix\n")
        f.write(f"  C_agg_sym_tau{lag}.npy        — symmetrised count matrix\n")
        f.write(f"  T_agg_tau{lag}.npy            — aggregated transition matrix\n")
        f.write(f"  eigenvalues_agg_tau{lag}.npy  — aggregated eigenvalues\n")
        f.write(f"  stationary_dist_tau{lag}.npy  — stationary distribution\n")
        if not args.no_scan:
            f.write(f"  lag_scan_results.npz         — lag-scan data\n")

        f.write(f"\nPlots:\n")
        if not args.no_scan:
            f.write(f"  its_convergence_aggregated.png\n")
            f.write(f"  its_convergence_per_block.png\n")
        f.write(f"  count_matrix_aggregated.png\n")
        for bn in BLOCKS:
            f.write(f"  count_matrix_{bn}.png\n")
        f.write(f"  eigenvalue_spectrum_aggregated.png\n")
        f.write(f"  stationary_dist_aggregated.png\n")

    print(f"  Summary → {summary_path}")
    print(f"\n  Step 4 (count matrices) complete.")


if __name__ == "__main__":
    main()
