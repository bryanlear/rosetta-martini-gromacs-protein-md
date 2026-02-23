#!/usr/bin/env python
"""
Step 6 — Local Stationary Distribution via Eigenvector Decomposition
=====================================================================
MSM pipeline for Mutant Nav1.5 (Martini 3 CG).

Computes the local stationary distribution π for each isolated
thermodynamic basin (trajectory block) and for the aggregated
(all blocks) model.

Theory
------
The stationary distribution π is defined as the left eigenvector
of the row-stochastic transition probability matrix T(τ) that
corresponds to the Perron eigenvalue λ₁ = 1:

    πᵀ T = πᵀ          (left eigenvector equation)

subject to the local probability normalization constraint:

    Σ_i  π_i = 1  ,   π_i ≥ 0   ∀ i

Each π_i represents the thermodynamic probability of the system
occupying microstate S_i relative only to the other microstates
within that defined contiguous block.

For the **reversible** estimator (T_rev from Step 5) the stationary
distribution is analytically given by the row sums of C_sym (which
are proportional to π).  This script verifies that the left
eigenvector decomposition reproduces the analytic result to machine
precision.

For the **non-reversible** estimator (T_nonrev) the stationary
distribution is computed solely from the eigenvector decomposition.

Per-block analysis
------------------
In addition to the aggregated model, the script builds local T
matrices *per block* (all blocks), restricts each to its own
SCC within the 197 active states, and computes local π.  These
per-block distributions reveal the extent of basin-specific sampling.

Inputs
------
From Step 5:  transition_matrices/T_rev_tau10.npy
              transition_matrices/T_nonrev_tau10.npy
              transition_matrices/C_scc_tau10.npy
              transition_matrices/scc_state_map.npy
              transition_matrices/stationary_dist_tau10.npy   (count-based)
              transition_matrices/msm_block{1,3}_labels_scc.npy

Outputs
-------
stationary_distribution/    — directory with all outputs

Usage
-----
    conda activate bio_env
    python step6_stationary_distribution.py               # default lag=10
    python step6_stationary_distribution.py --lag 20
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
TRANS_DIR = os.path.join(BASE, "transition_matrices")
COUNT_DIR = os.path.join(BASE, "count_matrices")
OUT_DIR   = os.path.join(BASE, "stationary_distribution")
BLOCKS    = ["msm_block1", "msm_block2", "msm_block3"]
DT_PS     = 100.0                               # ps per frame
N_STATES_ORIG = 200                             # original k


# ────────────────────────────────────────────────────────────────
# HELPER: left eigenvector for λ = 1
# ────────────────────────────────────────────────────────────────

def stationary_from_eigenvector(T, label=""):
    """Compute π as the left eigenvector of T for λ₁ = 1.

    Left eigenvector:  πᵀ T = λ πᵀ   ⟺   Tᵀ π = λ π

    We solve the *right* eigenproblem of Tᵀ and look for the
    eigenvector whose eigenvalue is closest to 1.

    Returns
    -------
    pi     : ndarray (m,) — normalised stationary distribution
    lam1   : float        — the eigenvalue (should be 1.0)
    all_ev : ndarray      — all eigenvalues of T (sorted descending
                            by real part)
    """
    m = T.shape[0]
    # Full eigendecomposition of Tᵀ
    eigenvalues, eigenvectors = linalg.eig(T.T)

    # Sort by descending real part
    idx = np.argsort(-eigenvalues.real)
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # The leading eigenvalue should be 1.0
    lam1 = eigenvalues[0].real
    pi_raw = eigenvectors[:, 0].real

    # Ensure positive orientation (Perron–Frobenius: entries ≥ 0)
    if pi_raw.sum() < 0:
        pi_raw = -pi_raw

    # Warn if any entry is significantly negative
    min_val = pi_raw.min()
    if min_val < -1e-10:
        print(f"  ⚠ {label}: min(π_raw) = {min_val:.3e}  "
              f"(should be ≥ 0 for an irreducible chain)")

    # Clip tiny numerical negatives, then normalise
    pi_raw = np.clip(pi_raw, 0.0, None)
    pi = pi_raw / pi_raw.sum()

    return pi, lam1, eigenvalues.real


def validate_stationary(pi, T, label=""):
    """Validate πᵀ T = πᵀ  and normalization."""
    m = T.shape[0]
    residual = pi @ T - pi
    l2   = np.linalg.norm(residual)
    linf = np.max(np.abs(residual))

    print(f"\n  {'─'*50}")
    print(f"  Validation — {label}")
    print(f"  {'─'*50}")
    print(f"  m (active states)       : {m}")
    print(f"  Σ π_i                   : {pi.sum():.15f}")
    print(f"  min(π_i)                : {pi.min():.6e}")
    print(f"  max(π_i)                : {pi.max():.6e}")
    print(f"  ||πT - π||₂            : {l2:.3e}")
    print(f"  ||πT - π||_∞           : {linf:.3e}")
    print(f"  Non-negative            : {np.all(pi >= 0)}")
    ok = (l2 < 1e-10) and (abs(pi.sum() - 1.0) < 1e-12) and np.all(pi >= 0)
    print(f"  PASS                    : {ok}")
    return ok


# ────────────────────────────────────────────────────────────────
# PER-BLOCK LOCAL T AND π
# ────────────────────────────────────────────────────────────────

def build_block_T(labels, n_states, lag):
    """Build count matrix from frame labels, row-normalise → T.

    Returns C, T (n_states × n_states) — may contain zero rows.
    """
    C = np.zeros((n_states, n_states), dtype=np.int64)
    for t in range(len(labels) - lag):
        i, j = labels[t], labels[t + lag]
        if i >= 0 and j >= 0:          # -1 = unmapped sentinel
            C[i, j] += 1
    row_sums = C.sum(axis=1).astype(float)
    T = np.zeros_like(C, dtype=float)
    mask = row_sums > 0
    T[mask] = C[mask] / row_sums[mask, None]
    return C, T


def restrict_scc(C, label=""):
    """Find largest SCC of C; return restricted C, T, and index map."""
    n = C.shape[0]
    G = csr_matrix((C > 0).astype(int))
    n_scc, scc_labels = connected_components(G, directed=True,
                                              connection="strong")
    sizes = np.bincount(scc_labels)
    lid = np.argmax(sizes)
    scc_mask = scc_labels == lid
    scc_map  = np.where(scc_mask)[0]
    m = len(scc_map)

    C_scc = C[np.ix_(scc_map, scc_map)]
    row_sums = C_scc.sum(axis=1).astype(float)
    T_scc = C_scc / row_sums[:, None]

    print(f"  {label}: {n} → {m} SCC states  "
          f"(dropped {n - m}, total SCCs = {n_scc})")
    return C_scc, T_scc, scc_map


# ────────────────────────────────────────────────────────────────
# PLOTTING
# ────────────────────────────────────────────────────────────────

def plot_all(out_dir, results):
    """Generate all diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ── 1. Aggregated π distribution (bar) ──────────────────────
    pi_rev  = results["pi_rev_agg"]
    pi_nrev = results["pi_nonrev_agg"]
    scc_map = results["scc_state_map"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(scc_map, pi_rev, width=1.0, alpha=0.7,
           label="Reversible (eigenvector)")
    ax.bar(scc_map, pi_nrev, width=0.5, alpha=0.5, color="C1",
           label="Non-reversible (eigenvector)")
    ax.set_xlabel("Original microstate index")
    ax.set_ylabel("π_i")
    ax.set_title("Aggregated stationary distribution  (all blocks)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "pi_agg_bar.png"), dpi=200)
    plt.close(fig)

    # ── 2. Reversible π: eigenvector vs count-based ─────────────
    pi_counts = results["pi_counts_agg"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.scatter(pi_counts, pi_rev, s=8, alpha=0.7)
    lo, hi = min(pi_counts.min(), pi_rev.min()), max(pi_counts.max(), pi_rev.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
    ax.set_xlabel("π (count-based, Step 5)")
    ax.set_ylabel("π (eigenvector, Step 6)")
    ax.set_title("Reversible:  eigenvector  vs  counts")
    ax.legend(fontsize=8)

    ax = axes[1]
    diff = pi_rev - pi_counts
    ax.stem(scc_map, diff, markerfmt=".", linefmt="C2-", basefmt="k-")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Microstate index")
    ax.set_ylabel("Δπ  (eigenvec − counts)")
    ax.set_title(f"Residual   ||Δπ||₂ = {np.linalg.norm(diff):.3e}")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "pi_rev_eigvec_vs_counts.png"), dpi=200)
    plt.close(fig)

    # ── 3. Sorted π (rank-ordered) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lbl, pi_arr, c in [("Rev (eigvec)", pi_rev, "C0"),
                            ("Non-rev (eigvec)", pi_nrev, "C1")]:
        pi_sorted = np.sort(pi_arr)[::-1]
        ax.semilogy(np.arange(len(pi_sorted)) + 1, pi_sorted,
                     "-", lw=1.2, color=c, label=lbl)
    ax.set_xlabel("Rank")
    ax.set_ylabel("π_i  (log scale)")
    ax.set_title("Rank-ordered stationary probabilities")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "pi_ranked.png"), dpi=200)
    plt.close(fig)

    # ── 4. Eigenvalue spectrum (top 30) ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n_show = min(30, len(results["eigenvalues_rev_agg"]))
    ax.plot(range(1, n_show + 1),
            results["eigenvalues_rev_agg"][:n_show],
            "o-", ms=4, label="Reversible T")
    ax.plot(range(1, n_show + 1),
            results["eigenvalues_nonrev_agg"][:n_show],
            "s-", ms=3, alpha=0.7, label="Non-reversible T")
    ax.axhline(1.0, color="k", ls="--", lw=0.6)
    ax.set_xlabel("Eigenvalue rank i")
    ax.set_ylabel("λ_i")
    ax.set_title("Leading eigenvalue spectrum of T")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "eigenvalue_spectrum.png"), dpi=200)
    plt.close(fig)

    # ── 5. Per-block π comparison ───────────────────────────────
    block_data = results.get("per_block", {})
    if len(block_data) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

        # (a) overlay bars for each block
        ax = axes[0]
        for bk, bdata in block_data.items():
            smap = bdata["scc_map_local"]
            # map local SCC indices back to original state space
            orig_ids = scc_map[smap]
            ax.bar(orig_ids, bdata["pi_rev"], width=0.8,
                   alpha=0.55, label=bk)
        ax.set_xlabel("Original microstate index")
        ax.set_ylabel("π_i (local)")
        ax.set_title("Per-block local stationary distributions")
        ax.legend(fontsize=9)

        # (b) scatter block1 π vs block3 π on shared SCC states
        ax = axes[1]
        b1 = block_data.get("msm_block1")
        b3 = block_data.get("msm_block3")
        if b1 is not None and b3 is not None:
            # find common states (in SCC-197 index space)
            set1 = set(b1["scc_map_local"].tolist())
            set3 = set(b3["scc_map_local"].tolist())
            common = sorted(set1 & set3)
            if len(common) > 0:
                pi1_common = np.array([b1["pi_rev"][
                    np.where(b1["scc_map_local"] == s)[0][0]]
                    for s in common])
                pi3_common = np.array([b3["pi_rev"][
                    np.where(b3["scc_map_local"] == s)[0][0]]
                    for s in common])
                ax.scatter(pi1_common, pi3_common, s=10, alpha=0.6)
                lo = min(pi1_common.min(), pi3_common.min())
                hi = max(pi1_common.max(), pi3_common.max())
                ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
                ax.set_xlabel("π (block 1)")
                ax.set_ylabel("π (block 3)")
                ax.set_title(f"Block 1 vs Block 3 — {len(common)} shared states")
            else:
                ax.text(0.5, 0.5, "No shared SCC states",
                        transform=ax.transAxes, ha="center")

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "pi_per_block.png"), dpi=200)
        plt.close(fig)

    # ── 6. Entropy and cumulative distribution ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Shannon entropy = -Σ π_i ln π_i
    for lbl, pi_arr, c in [("Rev", pi_rev, "C0"),
                            ("Non-rev", pi_nrev, "C1")]:
        mask = pi_arr > 0
        H = -np.sum(pi_arr[mask] * np.log(pi_arr[mask]))
        H_max = np.log(len(pi_arr))
        axes[0].bar(lbl, H, color=c, alpha=0.7)
        axes[0].axhline(H_max, color="k", ls="--", lw=0.6)
        # cumulative
        pi_sorted = np.sort(pi_arr)[::-1]
        cdf = np.cumsum(pi_sorted)
        axes[1].plot(np.arange(len(cdf)) + 1, cdf,
                     "-", lw=1.3, color=c, label=lbl)

    axes[0].set_ylabel("Shannon entropy  H(π)")
    axes[0].set_title(f"Entropy  (H_max = {np.log(len(pi_rev)):.2f})")
    axes[1].axhline(0.90, color="gray", ls=":", lw=0.8, label="90%")
    axes[1].set_xlabel("Number of states  (rank-ordered)")
    axes[1].set_ylabel("Cumulative probability")
    axes[1].set_title("Cumulative distribution")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "entropy_cumulative.png"), dpi=200)
    plt.close(fig)

    n_plots = 6 if len(block_data) == 2 else 5
    print(f"\n  {n_plots} plots saved → {plot_dir}/")


# ────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ────────────────────────────────────────────────────────────────

def write_summary(out_dir, results, lag):
    """Write a human-readable summary file."""
    fpath = os.path.join(out_dir, f"summary_stationary_tau{lag}.txt")
    with open(fpath, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("Step 6 — Local Stationary Distribution (Eigenvector)\n")
        f.write(f"Lag τ = {lag} frames = {lag * DT_PS / 1000:.1f} ns\n")
        f.write("=" * 65 + "\n\n")

        pi_rev  = results["pi_rev_agg"]
        pi_nrev = results["pi_nonrev_agg"]
        scc_map = results["scc_state_map"]
        m = len(pi_rev)

        f.write(f"Active (SCC) states : {m}\n")
        f.write(f"Original states     : {N_STATES_ORIG}\n\n")

        f.write("─── Aggregated (all blocks) ───\n\n")
        for tag, pi_arr, ev in [
            ("Reversible",     pi_rev,  results["eigenvalues_rev_agg"]),
            ("Non-reversible", pi_nrev, results["eigenvalues_nonrev_agg"]),
        ]:
            f.write(f"  {tag}:\n")
            f.write(f"    λ₁          = {ev[0]:.15f}\n")
            f.write(f"    Σ π_i       = {pi_arr.sum():.15f}\n")
            f.write(f"    min(π_i)    = {pi_arr.min():.6e}\n")
            f.write(f"    max(π_i)    = {pi_arr.max():.6e}\n")
            f.write(f"    π_i > 0     = {np.sum(pi_arr > 0)} / {m}\n")
            mask = pi_arr > 0
            H = -np.sum(pi_arr[mask] * np.log(pi_arr[mask]))
            H_max = np.log(m)
            f.write(f"    H(π)        = {H:.4f}  "
                    f"(H_max = {H_max:.4f},  efficiency = {H / H_max:.4f})\n")
            # 90% cumulative
            ps = np.sort(pi_arr)[::-1]
            n90 = np.searchsorted(np.cumsum(ps), 0.90) + 1
            f.write(f"    90% mass in = {n90} / {m} states\n\n")

        # Comparison with counts-based π
        pi_c = results["pi_counts_agg"]
        diff = pi_rev - pi_c
        f.write(f"  Rev eigvec vs count-based π:\n")
        f.write(f"    ||Δπ||₂     = {np.linalg.norm(diff):.3e}\n")
        f.write(f"    ||Δπ||_∞    = {np.max(np.abs(diff)):.3e}\n\n")

        # Top populated states
        f.write("─── Top 20 most populated states (reversible) ───\n\n")
        top = np.argsort(pi_rev)[::-1][:20]
        f.write(f"  {'Rank':>4s}  {'SCC_id':>6s}  {'Orig_id':>7s}  {'π_i':>12s}\n")
        f.write(f"  {'----':>4s}  {'------':>6s}  {'-------':>7s}  {'----':>12s}\n")
        for rank, idx in enumerate(top, 1):
            f.write(f"  {rank:4d}  {idx:6d}  {scc_map[idx]:7d}  "
                    f"{pi_rev[idx]:12.6e}\n")

        # Bottom 20
        f.write("\n─── Bottom 20 least populated states (reversible) ───\n\n")
        bot = np.argsort(pi_rev)[:20]
        f.write(f"  {'Rank':>4s}  {'SCC_id':>6s}  {'Orig_id':>7s}  {'π_i':>12s}\n")
        f.write(f"  {'----':>4s}  {'------':>6s}  {'-------':>7s}  {'----':>12s}\n")
        for rank, idx in enumerate(bot, 1):
            f.write(f"  {rank:4d}  {idx:6d}  {scc_map[idx]:7d}  "
                    f"{pi_rev[idx]:12.6e}\n")

        # Per-block
        block_data = results.get("per_block", {})
        if block_data:
            f.write("\n─── Per-block local stationary distributions ───\n\n")
            for bk, bd in block_data.items():
                f.write(f"  {bk}:\n")
                f.write(f"    Local SCC states    = {len(bd['pi_rev'])}\n")
                f.write(f"    λ₁ (rev)            = {bd['lam1_rev']:.15f}\n")
                f.write(f"    λ₁ (nonrev)         = {bd['lam1_nonrev']:.15f}\n")
                f.write(f"    min(π_rev)          = {bd['pi_rev'].min():.6e}\n")
                f.write(f"    max(π_rev)          = {bd['pi_rev'].max():.6e}\n")
                H = -np.sum(bd['pi_rev'][bd['pi_rev'] > 0] *
                            np.log(bd['pi_rev'][bd['pi_rev'] > 0]))
                H_max = np.log(len(bd['pi_rev']))
                f.write(f"    H(π_rev)            = {H:.4f}  "
                        f"(efficiency = {H / H_max:.4f})\n\n")

        f.write("=" * 65 + "\n")
    print(f"  Summary → {fpath}")


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Local stationary distribution via "
                    "eigenvector decomposition")
    parser.add_argument("--lag", type=int, default=10,
                        help="Lag time in frames (default: 10)")
    args = parser.parse_args()
    lag = args.lag

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    print("=" * 65)
    print("Step 6 — Local Stationary Distribution (Eigenvector Decomposition)")
    print(f"Lag τ = {lag} frames  =  {lag * DT_PS / 1000:.1f} ns")
    print("=" * 65)

    # ── Load Step 5 outputs ─────────────────────────────────────
    print("\n[1] Loading transition matrices from Step 5 …")
    T_rev    = np.load(os.path.join(TRANS_DIR, f"T_rev_tau{lag}.npy"))
    T_nonrev = np.load(os.path.join(TRANS_DIR, f"T_nonrev_tau{lag}.npy"))
    C_scc    = np.load(os.path.join(TRANS_DIR, f"C_scc_tau{lag}.npy"))
    scc_map  = np.load(os.path.join(TRANS_DIR, "scc_state_map.npy"))
    pi_counts = np.load(os.path.join(TRANS_DIR,
                                      f"stationary_dist_tau{lag}.npy"))
    m = T_rev.shape[0]
    print(f"  T_rev      : ({m}, {m})  row-stochastic reversible")
    print(f"  T_nonrev   : ({m}, {m})  row-stochastic non-reversible")
    print(f"  C_scc      : ({m}, {m})  counts = {C_scc.sum():,}")
    print(f"  SCC states : {m}  (of {N_STATES_ORIG} original)")

    # ── 2. Eigenvector decomposition — Reversible T ─────────────
    print("\n[2] Eigenvector decomposition — Reversible T …")
    pi_rev, lam1_rev, ev_rev = stationary_from_eigenvector(
        T_rev, label="T_rev")
    print(f"  λ₁     = {lam1_rev:.15f}")
    print(f"  |λ₁-1| = {abs(lam1_rev - 1.0):.3e}")
    ok_rev = validate_stationary(pi_rev, T_rev, label="Reversible (eigvec)")

    # Cross-check with count-based π from Step 5
    diff_rc = pi_rev - pi_counts
    print(f"\n  Cross-check with count-based π (Step 5):")
    print(f"    ||π_eigvec − π_counts||₂  = {np.linalg.norm(diff_rc):.3e}")
    print(f"    ||π_eigvec − π_counts||_∞ = {np.max(np.abs(diff_rc)):.3e}")
    print(f"    max |Δπ/π|                = "
          f"{np.max(np.abs(diff_rc) / np.clip(pi_counts, 1e-30, None)):.3e}")

    # ── 3. Eigenvector decomposition — Non-reversible T ─────────
    print("\n[3] Eigenvector decomposition — Non-reversible T …")
    pi_nrev, lam1_nrev, ev_nrev = stationary_from_eigenvector(
        T_nonrev, label="T_nonrev")
    print(f"  λ₁     = {lam1_nrev:.15f}")
    print(f"  |λ₁-1| = {abs(lam1_nrev - 1.0):.3e}")
    ok_nrev = validate_stationary(pi_nrev, T_nonrev,
                                   label="Non-reversible (eigvec)")

    # Compare reversible vs non-reversible π
    diff_rn = pi_rev - pi_nrev
    print(f"\n  Reversible vs Non-reversible π:")
    print(f"    ||Δπ||₂     = {np.linalg.norm(diff_rn):.3e}")
    print(f"    ||Δπ||_∞    = {np.max(np.abs(diff_rn)):.3e}")
    print(f"    cos(π_rev, π_nrev) = "
          f"{np.dot(pi_rev, pi_nrev) / (np.linalg.norm(pi_rev) * np.linalg.norm(pi_nrev)):.6f}")

    # ── 4. Per-block local stationary distributions ─────────────
    print("\n[4] Per-block local stationary distributions …")
    per_block = {}
    n_scc = m   # 197 active states in aggregated SCC space

    for bk in BLOCKS:
        print(f"\n  ── {bk} ──")
        labels = np.load(os.path.join(TRANS_DIR,
                                       f"{bk}_labels_scc.npy"))
        print(f"  Frames: {len(labels)}")

        # Build local count matrix in SCC-197 index space
        C_local, T_local_full = build_block_T(labels, n_scc, lag)
        n_counts = C_local.sum()
        n_visited = np.sum(C_local.sum(axis=1) > 0)
        print(f"  Local counts: {n_counts:,}   visited: {n_visited}")

        # Restrict to local SCC (within the 197 states)
        C_loc_scc, T_loc_scc, scc_local = restrict_scc(
            C_local, label=bk)

        # Reversible local T
        C_sym_local = C_loc_scc + C_loc_scc.T
        rs_sym = C_sym_local.sum(axis=1).astype(float)
        T_loc_rev = C_sym_local / rs_sym[:, None]

        # π via eigenvector
        pi_loc_rev, lam1_loc_rev, _ = stationary_from_eigenvector(
            T_loc_rev, label=f"{bk} rev")
        print(f"  λ₁ (rev)   = {lam1_loc_rev:.15f}")
        validate_stationary(pi_loc_rev, T_loc_rev,
                           label=f"{bk} reversible")

        pi_loc_nrev, lam1_loc_nrev, _ = stationary_from_eigenvector(
            T_loc_scc, label=f"{bk} nonrev")
        print(f"  λ₁ (nonrev)= {lam1_loc_nrev:.15f}")
        validate_stationary(pi_loc_nrev, T_loc_scc,
                           label=f"{bk} non-reversible")

        per_block[bk] = {
            "C_local_scc": C_loc_scc,
            "T_rev":       T_loc_rev,
            "T_nonrev":    T_loc_scc,
            "pi_rev":      pi_loc_rev,
            "pi_nonrev":   pi_loc_nrev,
            "lam1_rev":    lam1_loc_rev,
            "lam1_nonrev": lam1_loc_nrev,
            "scc_map_local": scc_local,   # indices into 0 .. 196
        }

    # ── 5. Thermodynamic analysis ───────────────────────────────
    print("\n[5] Thermodynamic summary …")
    for tag, pi_arr in [("Rev (agg)", pi_rev), ("Non-rev (agg)", pi_nrev)]:
        mask = pi_arr > 0
        H     = -np.sum(pi_arr[mask] * np.log(pi_arr[mask]))
        H_max = np.log(len(pi_arr))
        ps = np.sort(pi_arr)[::-1]
        n90  = np.searchsorted(np.cumsum(ps), 0.90) + 1
        n99  = np.searchsorted(np.cumsum(ps), 0.99) + 1
        print(f"  {tag:16s}  H = {H:.4f}  "
              f"H/H_max = {H / H_max:.4f}  "
              f"90%→{n90}  99%→{n99} states")

    # ── 6. Save outputs ─────────────────────────────────────────
    print("\n[6] Saving outputs …")
    np.save(os.path.join(OUT_DIR, f"pi_rev_eigvec_tau{lag}.npy"), pi_rev)
    np.save(os.path.join(OUT_DIR, f"pi_nonrev_eigvec_tau{lag}.npy"), pi_nrev)
    np.save(os.path.join(OUT_DIR, f"eigenvalues_rev_tau{lag}.npy"), ev_rev)
    np.save(os.path.join(OUT_DIR, f"eigenvalues_nonrev_tau{lag}.npy"), ev_nrev)
    np.save(os.path.join(OUT_DIR, f"scc_state_map.npy"), scc_map)

    # Per-block
    for bk, bd in per_block.items():
        np.save(os.path.join(OUT_DIR, f"{bk}_pi_rev_tau{lag}.npy"),
                bd["pi_rev"])
        np.save(os.path.join(OUT_DIR, f"{bk}_pi_nonrev_tau{lag}.npy"),
                bd["pi_nonrev"])
        np.save(os.path.join(OUT_DIR, f"{bk}_scc_local_map.npy"),
                bd["scc_map_local"])

    print(f"  All arrays saved → {OUT_DIR}/")

    # ── 7. Plots ────────────────────────────────────────────────
    print("\n[7] Generating plots …")
    results = {
        "pi_rev_agg":          pi_rev,
        "pi_nonrev_agg":       pi_nrev,
        "pi_counts_agg":       pi_counts,
        "eigenvalues_rev_agg": ev_rev,
        "eigenvalues_nonrev_agg": ev_nrev,
        "scc_state_map":       scc_map,
        "per_block":           per_block,
    }
    plot_all(OUT_DIR, results)

    # ── 8. Summary file ────────────────────────────────────────
    print("\n[8] Writing summary …")
    write_summary(OUT_DIR, results, lag)

    dt = time.time() - t0
    print(f"\n{'='*65}")
    print(f"Step 6 complete  ({dt:.1f} s)")
    ok_all = ok_rev and ok_nrev
    print(f"Overall validation: {'PASS ✓' if ok_all else 'ISSUES ⚠'}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
