#!/usr/bin/env python
"""
Step 2 — Time-lagged Independent Component Analysis (TICA)
============================================================
MSM pipeline for Mutant Nav1.5 (Martini 3 CG).

Implements TICA following Pérez-Hernández et al., JCTC 2013:

  1. Load per-block internal feature matrices from Step 1.
  2. Mean-centre features using global statistics.
  3. **PCA-whitening** to reduce dimensionality from d → k (effective rank).
     This eliminates the rank-deficiency problem when d >> N_frames.
     After whitening, C(0) = I in the reduced space.
  4. Build the time-lagged covariance  C(τ) = <z(t) z(t+τ)^T>  in the
     whitened space, respecting block boundaries.
  5. Symmetrise:  C_sym(τ) = 0.5 [C(τ) + C(τ)^T]
  6. Diagonalise C_sym(τ) — a standard symmetric eigenvalue problem
     (no generalised problem needed since C(0) = I after whitening).
  7. Eigenvalues λ_i ∈ (0, 1] are autocorrelations at lag τ.
     Implied timescales:  t_i = −τ / ln(λ_i).
  8. Project each block onto the top TICs.
  9. Lag-time scan for implied-timescale convergence diagnostic.

All outputs go to  independent_local_matrices/tica/
Inputs:           independent_local_matrices/msm_block{1,2,3}/features_combined.npy

Usage:
    conda activate bio_env
    python step2_tica.py                   # defaults: lag=10, var=0.95
    python step2_tica.py --lag 20          # custom lag
    python step2_tica.py --var 0.99        # retain 99 % PCA variance
    python step2_tica.py --pca-dim 500     # fixed PCA dimension
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
from scipy import linalg

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
BLOCKS   = ["msm_block1", "msm_block2", "msm_block3"]
DT_PS    = 100.0                         # trajectory stride (ps per frame)

# Defaults (overridable via CLI)
DEFAULT_LAG      = 10                    # frames → 1.0 ns at 100 ps/frame
DEFAULT_N_TICS   = 20                    # how many TICs to keep
DEFAULT_PCA_VAR  = 0.95                  # PCA explained-variance threshold
DEFAULT_PCA_DIM  = None                  # if set, overrides variance threshold

# Lag-time scan range for implied-timescale plot
LAG_SCAN = [1, 2, 5, 10, 20, 50, 100, 200]


# ────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ────────────────────────────────────────────────────────────────

def load_blocks(base_dir, block_names):
    """Load per-block feature matrices (promote to float64)."""
    data = []
    for bn in block_names:
        fpath = os.path.join(base_dir, bn, "features_combined.npy")
        X = np.load(fpath).astype(np.float64)
        print(f"  Loaded {bn:12s}  shape {X.shape}")
        data.append(X)
    return data


def global_mean(blocks):
    """Compute the global mean across all blocks."""
    n_total = sum(X.shape[0] for X in blocks)
    mean = np.zeros(blocks[0].shape[1], dtype=np.float64)
    for X in blocks:
        mean += X.sum(axis=0)
    mean /= n_total
    return mean


def centre_blocks(blocks, mean):
    """Subtract global mean from each block."""
    return [X - mean for X in blocks]


# ────────────────────────────────────────────────────────────────
# PCA WHITENING
# ────────────────────────────────────────────────────────────────

def pca_whitening(blocks, var_threshold=0.95, max_dim=None):
    """
    PCA-whiten the centred data.

    Strategy (N < d case, which applies here: 11,760 frames < 24,349 features):
      1. Economy SVD of X:  X = U S V^T  with  U (N,N), S (N,), V^T (N,d)
      2. Covariance eigenvalues:  σ_i = s_i² / (N-1)
      3. Keep k components explaining ≥ var_threshold of variance.
      4. Whitening matrix W = V_k diag(1/√σ_k)  so z = X @ W has C(0) = I.

    Parameters
    ----------
    blocks : list of (T_i, d) arrays — centred feature blocks
    var_threshold : float — cumulative explained variance to retain
    max_dim : int or None — hard cap on PCA dimension

    Returns
    -------
    W         : (d, k) whitening matrix
    explained : (k,) array of explained variance ratios
    eigenvals : (k,) covariance eigenvalues
    """
    X = np.vstack(blocks)    # (N, d)
    N, d = X.shape
    print(f"  Data matrix: N={N} frames, d={d} features")

    t0 = time.time()

    if N < d:
        # ── Gram-matrix path (N < d) ──
        print(f"  Using economy SVD (N={N} < d={d})")
        # Economy SVD: U(N,N), s(N,), Vt(N,d)
        # RAM ≈ N*d*8 bytes for Vt ≈ 2.3 GB — feasible
        U, s, Vt = linalg.svd(X, full_matrices=False)
        elapsed = time.time() - t0
        print(f"  SVD done in {elapsed:.1f} s   (rank ≤ {len(s)})")
        eigvals = s ** 2 / (N - 1)
    else:
        # ── Covariance path (d ≤ N) ──
        print(f"  Using covariance eigendecomposition (d={d} ≤ N={N})")
        Sigma = (X.T @ X) / (N - 1)
        eigvals_full, V_full = linalg.eigh(Sigma)
        idx = np.argsort(eigvals_full)[::-1]
        eigvals = eigvals_full[idx]
        Vt = V_full[:, idx].T
        elapsed = time.time() - t0
        print(f"  Eigendecomposition done in {elapsed:.1f} s")

    # Determine how many components to keep
    total_var = eigvals.sum()
    cumvar = np.cumsum(eigvals) / total_var
    if max_dim is not None:
        k = min(max_dim, len(eigvals))
        print(f"  Fixed PCA dimension: k = {k}")
    else:
        k = int(np.searchsorted(cumvar, var_threshold) + 1)
        k = min(k, len(eigvals))
        print(f"  Variance threshold {var_threshold:.2%} → k = {k} components")

    print(f"  Cumulative variance at k={k}: {cumvar[k-1]:.4%}")
    print(f"  Top 5 eigenvalues: {eigvals[:5]}")
    print(f"  Eigenvalue at k={k}: {eigvals[k-1]:.6e}")

    # Whitening matrix
    Vk = Vt[:k, :].T                       # (d, k)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(eigvals[:k], 1e-12))
    W = Vk * inv_sqrt[np.newaxis, :]        # (d, k)

    explained = eigvals[:k] / total_var

    return W, explained, eigvals[:k]


def whiten_blocks(blocks, W):
    """Apply whitening transform: z = X_centred @ W."""
    return [X @ W for X in blocks]


# ────────────────────────────────────────────────────────────────
# TICA IN WHITENED SPACE
# ────────────────────────────────────────────────────────────────

def build_lagged_covariance(blocks_w, lag):
    """
    Build C(τ) in the whitened space from multiple blocks.
    Block boundaries respected — no cross-block pairs.
    """
    k = blocks_w[0].shape[1]
    Ct = np.zeros((k, k), dtype=np.float64)
    n_pairs = 0

    for Z in blocks_w:
        T = Z.shape[0]
        if T <= lag:
            continue
        Z_t  = Z[:-lag]
        Z_tl = Z[lag:]
        Ct += Z_t.T @ Z_tl
        n_pairs += (T - lag)

    if n_pairs > 0:
        Ct /= n_pairs
    return Ct


def solve_tica_whitened(Ct, n_tics):
    """
    Solve TICA in the whitened space where C(0) = I.

    Diagonalise:  C_sym(τ) v = λ v   (standard symmetric eigenproblem).

    Returns
    -------
    eigenvalues  : (n_tics,) — autocorrelations, descending
    eigenvectors : (k, n_tics) — TIC directions in the whitened basis
    timescales   : (n_tics,) — implied timescales in units of lag
    """
    Ct_sym = 0.5 * (Ct + Ct.T)

    print(f"  Diagonalising C_sym(τ)  ({Ct_sym.shape[0]} × {Ct_sym.shape[1]}) …")
    t0 = time.time()
    eigvals, eigvecs = linalg.eigh(Ct_sym)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f} s")

    # eigh returns ascending → reverse to descending
    idx = np.argsort(eigvals)[::-1]
    eigvals  = eigvals[idx]
    eigvecs  = eigvecs[:, idx]

    n = min(n_tics, len(eigvals))
    eigvals  = eigvals[:n]
    eigvecs  = eigvecs[:, :n]

    # Clip to valid range for timescale computation
    eigvals_clipped = np.clip(eigvals, 1e-10, 1.0 - 1e-10)
    timescales = -1.0 / np.log(eigvals_clipped)     # units of lag

    return eigvals, eigvecs, timescales


def project_blocks(blocks_w, tica_eigvecs):
    """Project whitened blocks onto TICA components."""
    return [Z @ tica_eigvecs for Z in blocks_w]


def full_projection_matrix(W, tica_eigvecs):
    """Combined transform: raw centred features → TICA.  T = W @ V_tica."""
    return W @ tica_eigvecs


# ────────────────────────────────────────────────────────────────
# LAG-TIME SCAN
# ────────────────────────────────────────────────────────────────

def lag_time_scan(blocks_w, lags, n_tics):
    """Compute TICA eigenvalues at many lag times."""
    all_evals, all_ts, valid_lags = [], [], []

    for lag in lags:
        max_T = max(Z.shape[0] for Z in blocks_w)
        if lag >= max_T:
            print(f"  lag={lag} exceeds max block length ({max_T}), skipping")
            continue
        print(f"  lag = {lag:>4d} frames ({lag * DT_PS / 1000:.2f} ns) …", end="")
        Ct = build_lagged_covariance(blocks_w, lag)
        evals, _, ts = solve_tica_whitened(Ct, n_tics)
        ts_frames = ts * lag
        all_evals.append(evals)
        all_ts.append(ts_frames)
        valid_lags.append(lag)
        print(f"  top-3 λ = {evals[0]:.4f}  {evals[1]:.4f}  {evals[2]:.4f}")

    return {
        "lags":        np.array(valid_lags),
        "eigenvalues": np.array(all_evals),
        "timescales":  np.array(all_ts),
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
        return None


def plot_implied_timescales(scan, dt_ps, outdir):
    plt = _get_plt()
    if plt is None:
        return

    lags_ns = scan["lags"] * dt_ps / 1000.0
    ts_ns   = scan["timescales"] * dt_ps / 1000.0
    n_show  = min(10, ts_ns.shape[1])

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_show):
        valid = np.isfinite(ts_ns[:, i]) & (ts_ns[:, i] > 0)
        if valid.any():
            ax.plot(lags_ns[valid], ts_ns[valid, i], "o-",
                    label=f"TIC {i+1}", linewidth=1.5, markersize=4)

    ax.plot(lags_ns, 5 * lags_ns, "k--", alpha=0.3, label="t = 5τ")
    ax.set_xlabel("Lag time τ (ns)", fontsize=12)
    ax.set_ylabel("Implied timescale (ns)", fontsize=12)
    ax.set_title("TICA Implied Timescales — Mutant Nav1.5", fontsize=14)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(outdir, "tica_implied_timescales.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved implied-timescale plot → {fpath}")


def plot_pca_scree(explained, outdir):
    plt = _get_plt()
    if plt is None:
        return

    cumvar = np.cumsum(explained)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.semilogy(range(1, len(explained) + 1), explained, "o-", markersize=2,
                 color="steelblue", linewidth=0.8)
    ax1.set_xlabel("PCA component", fontsize=11)
    ax1.set_ylabel("Explained variance ratio", fontsize=11)
    ax1.set_title("PCA Scree Plot", fontsize=13)
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, len(cumvar) + 1), cumvar * 100, "-",
             color="darkorange", linewidth=1.5)
    ax2.axhline(95, color="red", ls="--", alpha=0.5, label="95 %")
    ax2.axhline(99, color="green", ls="--", alpha=0.5, label="99 %")
    ax2.set_xlabel("Number of PCA components", fontsize=11)
    ax2.set_ylabel("Cumulative variance (%)", fontsize=11)
    ax2.set_title("Cumulative Explained Variance", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(outdir, "pca_whitening_scree.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved PCA scree plot → {fpath}")


def plot_eigenvalue_spectrum(eigenvalues, outdir):
    plt = _get_plt()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, len(eigenvalues) + 1), eigenvalues,
           color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("TIC index", fontsize=12)
    ax.set_ylabel("Eigenvalue λ", fontsize=12)
    ax.set_title("TICA Eigenvalue Spectrum — Mutant Nav1.5", fontsize=14)
    ax.axhline(0, color="red", linestyle="--", alpha=0.4)
    ax.axhline(1, color="gray", linestyle=":", alpha=0.4)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    fpath = os.path.join(outdir, "tica_eigenvalue_spectrum.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved eigenvalue spectrum → {fpath}")


def plot_tic_projections(projections, block_names, dt_ps, outdir):
    plt = _get_plt()
    if plt is None:
        return

    # TIC-1 vs TIC-2 scatter per block
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    cmap = plt.cm.viridis
    for i, (proj, bname) in enumerate(zip(projections, block_names)):
        t_ns = np.arange(proj.shape[0]) * dt_ps / 1000.0
        sc = axes[i].scatter(proj[:, 0], proj[:, 1], c=t_ns, cmap=cmap,
                             s=2, alpha=0.5, rasterized=True)
        axes[i].set_title(bname, fontsize=11)
        axes[i].set_xlabel("TIC-1", fontsize=10)
        if i == 0:
            axes[i].set_ylabel("TIC-2", fontsize=10)
        plt.colorbar(sc, ax=axes[i], label="time (ns)", shrink=0.8)
    fig.suptitle("TICA Projection — Mutant Nav1.5", fontsize=14)
    plt.tight_layout()
    fpath = os.path.join(outdir, "tica_projection_tic1_vs_tic2.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved TIC-1 vs TIC-2 scatter → {fpath}")

    # TIC-1 time series
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)
    for i, (proj, bname) in enumerate(zip(projections, block_names)):
        t_ns = np.arange(proj.shape[0]) * dt_ps / 1000.0
        axes[i].plot(t_ns, proj[:, 0], linewidth=0.4, color="steelblue")
        axes[i].set_ylabel("TIC-1", fontsize=10)
        axes[i].set_title(bname, fontsize=11)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (ns)", fontsize=11)
    fig.suptitle("TIC-1 Time Series — Mutant Nav1.5", fontsize=14)
    plt.tight_layout()
    fpath = os.path.join(outdir, "tica_tic1_timeseries.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved TIC-1 time-series → {fpath}")

    # 2D histograms (combined)
    all_proj = np.vstack(projections)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs  = [(0, 1), (0, 2), (1, 2)]
    labels = [("TIC-1", "TIC-2"), ("TIC-1", "TIC-3"), ("TIC-2", "TIC-3")]
    for ax, (ci, cj), (lx, ly) in zip(axes, pairs, labels):
        ax.hist2d(all_proj[:, ci], all_proj[:, cj], bins=100,
                  cmap="inferno", density=True)
        ax.set_xlabel(lx, fontsize=10)
        ax.set_ylabel(ly, fontsize=10)
    fig.suptitle("TICA 2D Histograms (all blocks) — Mutant Nav1.5", fontsize=14)
    plt.tight_layout()
    fpath = os.path.join(outdir, "tica_2d_histograms.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved 2D histograms → {fpath}")


def plot_cumulative_kinetic_variance(eigenvalues, outdir):
    plt = _get_plt()
    if plt is None:
        return

    valid = (eigenvalues > 0) & (eigenvalues < 1.0)
    kv = np.zeros_like(eigenvalues)
    kv[valid] = eigenvalues[valid] ** 2 / (1 - eigenvalues[valid] ** 2)
    if kv.sum() > 0:
        cumvar = np.cumsum(kv) / kv.sum()
    else:
        cumvar = np.cumsum(kv)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, "o-",
            color="darkorange", linewidth=2, markersize=5)
    ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="90 %")
    ax.axhline(0.95, color="green", linestyle="--", alpha=0.5, label="95 %")
    ax.set_xlabel("Number of TICs", fontsize=12)
    ax.set_ylabel("Cumulative kinetic variance", fontsize=12)
    ax.set_title("Cumulative Kinetic Variance — Mutant Nav1.5", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(outdir, "tica_cumulative_kinetic_variance.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved cumulative kinetic variance → {fpath}")


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 2 — TICA (PCA-whitened)")
    parser.add_argument("--lag",     type=int,   default=DEFAULT_LAG,
                        help="Lag time in frames (default: %(default)s)")
    parser.add_argument("--ntics",   type=int,   default=DEFAULT_N_TICS,
                        help="Number of TICs to keep (default: %(default)s)")
    parser.add_argument("--var",     type=float, default=DEFAULT_PCA_VAR,
                        help="PCA variance threshold (default: %(default)s)")
    parser.add_argument("--pca-dim", type=int,   default=DEFAULT_PCA_DIM,
                        help="Fixed PCA dimension (overrides --var)")
    parser.add_argument("--no-scan", action="store_true",
                        help="Skip lag-time scan")
    args = parser.parse_args()

    lag      = args.lag
    n_tics   = args.ntics
    var_thr  = args.var
    pca_dim  = args.pca_dim

    OUT = os.path.join(BASE, "tica")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 65)
    print("  Step 2 — TICA  (PCA-whitened)  |  Mutant Nav1.5  (Martini 3 CG)")
    print("=" * 65)
    print(f"  Lag time         : {lag} frames  ({lag * DT_PS / 1000:.2f} ns)")
    print(f"  N TICs           : {n_tics}")
    if pca_dim is None:
        print(f"  PCA var threshold: {var_thr:.2%}")
    else:
        print(f"  PCA fixed dim    : {pca_dim}")
    print(f"  Output dir       : {OUT}")

    # ── 1. Load data ──
    print("\n[1] Loading per-block feature matrices …")
    raw_blocks = load_blocks(BASE, BLOCKS)

    # ── 2. Centre ──
    print("\n[2] Computing global mean and centring …")
    mean = global_mean(raw_blocks)
    blocks = centre_blocks(raw_blocks, mean)
    del raw_blocks
    d = blocks[0].shape[1]
    n_total = sum(X.shape[0] for X in blocks)
    print(f"  Features: {d},  Total frames: {n_total}")
    np.save(os.path.join(OUT, "feature_mean.npy"), mean)

    # ── 3. PCA whitening ──
    print("\n[3] PCA whitening …")
    W, explained, pca_eigvals = pca_whitening(blocks, var_threshold=var_thr,
                                               max_dim=pca_dim)
    k = W.shape[1]
    print(f"  Whitened dimension: k = {k}")

    np.save(os.path.join(OUT, "pca_whitening_matrix.npy"), W)
    np.save(os.path.join(OUT, "pca_explained_variance.npy"), explained)
    np.save(os.path.join(OUT, "pca_eigenvalues.npy"), pca_eigvals)

    # Whiten blocks
    blocks_w = whiten_blocks(blocks, W)
    del blocks

    # Sanity check: C(0) ≈ I
    print("  Verifying whitening (C(0) ≈ I) …", end="")
    Z_all = np.vstack(blocks_w)
    C0_check = (Z_all.T @ Z_all) / (Z_all.shape[0] - 1)
    off_diag = C0_check - np.eye(k)
    print(f"  max |C(0) - I| = {np.abs(off_diag).max():.2e}  ✓")
    del Z_all, C0_check

    # ── 4. Build C(τ) ──
    print(f"\n[4] Building C(τ={lag}) in whitened space ({k} × {k}) …")
    t0 = time.time()
    Ct = build_lagged_covariance(blocks_w, lag)
    print(f"  Built in {time.time() - t0:.1f} s")

    # ── 5. Solve TICA ──
    print(f"\n[5] Solving TICA eigenvalue problem …")
    eigenvalues, tica_eigvecs, timescales_lag = solve_tica_whitened(Ct, n_tics)

    ts_frames = timescales_lag * lag
    ts_ns     = ts_frames * DT_PS / 1000.0

    print(f"\n  Top {min(10, n_tics)} TICA eigenvalues & implied timescales:")
    print(f"  {'TIC':>5s}  {'λ':>10s}  {'t (frames)':>12s}  {'t (ns)':>10s}")
    print(f"  {'-'*5:>5s}  {'-'*10:>10s}  {'-'*12:>12s}  {'-'*10:>10s}")
    for i in range(min(10, n_tics)):
        print(f"  {i+1:5d}  {eigenvalues[i]:10.6f}  {ts_frames[i]:12.1f}  {ts_ns[i]:10.2f}")

    # Full projection matrix: raw centred features → TICA
    T_full = full_projection_matrix(W, tica_eigvecs)

    np.save(os.path.join(OUT, "tica_eigenvalues.npy"),        eigenvalues)
    np.save(os.path.join(OUT, "tica_eigenvectors_white.npy"), tica_eigvecs)
    np.save(os.path.join(OUT, "tica_transform_full.npy"),     T_full)
    np.save(os.path.join(OUT, "tica_timescales_frames.npy"),  ts_frames)
    np.save(os.path.join(OUT, "tica_timescales_ns.npy"),      ts_ns)
    np.save(os.path.join(OUT, "Ct_whitened.npy"),             Ct)

    # ── 6. Project blocks ──
    print(f"\n[6] Projecting blocks onto {n_tics} TICs …")
    projections = project_blocks(blocks_w, tica_eigvecs)
    for bn, proj in zip(BLOCKS, projections):
        fpath = os.path.join(OUT, f"{bn}_tica.npy")
        np.save(fpath, proj.astype(np.float32))
        print(f"  {bn:12s}  →  {fpath}   shape {proj.shape}")

    all_proj = np.vstack(projections).astype(np.float32)
    np.save(os.path.join(OUT, "tica_all_blocks.npy"), all_proj)
    print(f"  All blocks  →  tica_all_blocks.npy   shape {all_proj.shape}")

    # ── 7. Lag-time scan ──
    if not args.no_scan:
        print(f"\n[7] Lag-time scan for implied timescales …")
        min_block = min(Z.shape[0] for Z in blocks_w)
        valid_lags = [l for l in LAG_SCAN if l < min_block // 2]
        scan = lag_time_scan(blocks_w, valid_lags, n_tics)
        np.savez(os.path.join(OUT, "tica_lag_scan.npz"),
                 lags=scan["lags"],
                 eigenvalues=scan["eigenvalues"],
                 timescales=scan["timescales"])
        plot_implied_timescales(scan, DT_PS, OUT)
    else:
        print("\n[7] Lag-time scan skipped (--no-scan)")

    # ── 8. Diagnostic plots ──
    print(f"\n[8] Generating diagnostic plots …")
    plot_pca_scree(explained, OUT)
    plot_eigenvalue_spectrum(eigenvalues, OUT)
    plot_tic_projections(projections, BLOCKS, DT_PS, OUT)
    plot_cumulative_kinetic_variance(eigenvalues, OUT)

    # ── 9. Summary ──
    summary_path = os.path.join(OUT, "tica_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Step 2 — TICA Summary (PCA-whitened)\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Lag time           : {lag} frames ({lag * DT_PS / 1000:.2f} ns)\n")
        f.write(f"N TICs kept        : {n_tics}\n")
        f.write(f"Original features  : {d}\n")
        f.write(f"PCA components (k) : {k}\n")
        f.write(f"PCA variance       : {np.sum(explained):.4%}\n")
        f.write(f"Total frames       : {n_total}\n\n")
        f.write("Block shapes (whitened):\n")
        for bn, Z in zip(BLOCKS, blocks_w):
            f.write(f"  {bn}: {Z.shape}\n")
        f.write(f"\nTICA Eigenvalues & Implied Timescales:\n")
        f.write(f"  {'TIC':>5s}  {'λ':>12s}  {'t (frames)':>12s}  {'t (ns)':>10s}\n")
        for i in range(n_tics):
            f.write(f"  {i+1:5d}  {eigenvalues[i]:12.6f}  {ts_frames[i]:12.1f}  {ts_ns[i]:10.2f}\n")
        f.write(f"\nOutputs in {OUT}/:\n")
        f.write(f"  feature_mean.npy              — global mean ({d},)\n")
        f.write(f"  pca_whitening_matrix.npy      — W ({d}, {k})\n")
        f.write(f"  pca_explained_variance.npy    — ({k},)\n")
        f.write(f"  pca_eigenvalues.npy           — ({k},)\n")
        f.write(f"  tica_eigenvalues.npy          — ({n_tics},)\n")
        f.write(f"  tica_eigenvectors_white.npy   — ({k}, {n_tics}) whitened basis\n")
        f.write(f"  tica_transform_full.npy       — ({d}, {n_tics}) raw→TICA\n")
        f.write(f"  tica_timescales_frames.npy    — ({n_tics},)\n")
        f.write(f"  tica_timescales_ns.npy        — ({n_tics},)\n")
        f.write(f"  Ct_whitened.npy               — C(τ) ({k}, {k})\n")
        for bn in BLOCKS:
            f.write(f"  {bn}_tica.npy             — per-block projection\n")
        f.write(f"  tica_all_blocks.npy           — concatenated projection\n")
        if not args.no_scan:
            f.write(f"  tica_lag_scan.npz             — lag-time scan data\n")
        f.write(f"\nPlots:\n")
        f.write(f"  pca_whitening_scree.png\n")
        f.write(f"  tica_eigenvalue_spectrum.png\n")
        f.write(f"  tica_implied_timescales.png\n")
        f.write(f"  tica_projection_tic1_vs_tic2.png\n")
        f.write(f"  tica_tic1_timeseries.png\n")
        f.write(f"  tica_2d_histograms.png\n")
        f.write(f"  tica_cumulative_kinetic_variance.png\n")
    print(f"\n  Summary → {summary_path}")
    print("\n  Step 2 (TICA) complete.")


if __name__ == "__main__":
    main()
