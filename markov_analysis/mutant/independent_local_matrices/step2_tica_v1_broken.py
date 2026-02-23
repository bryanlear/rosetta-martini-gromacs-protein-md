#!/usr/bin/env python
"""
Step 2 — Time-lagged Independent Component Analysis (TICA)
============================================================
MSM pipeline for Mutant Nav1.5 (Martini 3 CG).

Implements TICA following Pérez-Hernández et al., JCTC 2021 / Molgedey & Schuster 1994:

  1. Load per-block internal feature matrices from Step 1.
  2. Mean-centre and standardise features (z-score) using *global* statistics.
  3. Build the instantaneous covariance  C(0) = <x(t) x(t)^T>
     and time-lagged covariance          C(τ) = <x(t) x(t+τ)^T>
     respecting block boundaries (no cross-block pairs).
  4. Symmetrise the time-lagged matrix: C_sym(τ) = 0.5 [C(τ) + C(τ)^T]
     to enforce real eigenvalues.
  5. Regularise C(0) with a Tikhonov term (ε·I) for numerical stability.
  6. Solve the generalised eigenvalue problem  C_sym(τ) v = λ C(0) v.
  7. Sort eigenvalues in descending order; compute implied timescales
        t_i = −τ / ln(λ_i)
  8. Project each block onto the top-k TICA components (TICs).
  9. Save:  eigenvalues, eigenvectors, per-block projections, lag-time scan.

All outputs go to  independent_local_matrices/tica/
Inputs:           independent_local_matrices/msm_block{1,2,3}/features_combined.npy

Usage:
    conda activate bio_env
    python step2_tica.py              # default lag τ = 10 frames (1 ns)
    python step2_tica.py --lag 20     # custom lag
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
DEFAULT_LAG     = 10                     # frames  → 1.0 ns at 100 ps/frame
DEFAULT_N_TICS  = 20                     # how many TICs to keep
DEFAULT_EPS     = 1e-6                   # Tikhonov regularisation on C(0)

# Lag-time scan range for implied-timescale plot
LAG_SCAN        = [1, 2, 5, 10, 20, 50, 100, 200]

# ────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────

def load_blocks(base_dir, block_names):
    """Load per-block feature matrices and return as a list of arrays."""
    data = []
    for bn in block_names:
        fpath = os.path.join(base_dir, bn, "features_combined.npy")
        X = np.load(fpath).astype(np.float64)    # promote to f64 for precision
        print(f"  Loaded {bn:12s}  shape {X.shape}")
        data.append(X)
    return data


def global_mean_std(blocks):
    """Compute global mean and std over concatenated blocks (Welford-like)."""
    n_total = sum(X.shape[0] for X in blocks)
    d = blocks[0].shape[1]
    mean = np.zeros(d, dtype=np.float64)
    for X in blocks:
        mean += X.sum(axis=0)
    mean /= n_total

    var = np.zeros(d, dtype=np.float64)
    for X in blocks:
        var += ((X - mean) ** 2).sum(axis=0)
    var /= (n_total - 1)
    std = np.sqrt(var)
    std[std < 1e-12] = 1.0                # avoid div-by-zero for constant features
    return mean, std


def standardise_blocks(blocks, mean, std):
    """Z-score each block in-place using global statistics."""
    out = []
    for X in blocks:
        out.append((X - mean) / std)
    return out


def build_covariance_matrices(blocks, lag):
    """
    Build C(0) and C(τ) from multiple independent blocks.

    C(0) = (1/N_total) Σ_blocks Σ_t  x(t) x(t)^T
    C(τ) = (1/N_total) Σ_blocks Σ_t  x(t) x(t+τ)^T

    Block boundaries are respected: no pairs that span two blocks.
    """
    d = blocks[0].shape[1]
    C0  = np.zeros((d, d), dtype=np.float64)
    Ct  = np.zeros((d, d), dtype=np.float64)
    n0  = 0
    nt  = 0

    for X in blocks:
        T = X.shape[0]
        if T <= lag:
            print(f"  WARNING: block has {T} frames < lag {lag}, skipping for C(τ)")
            # Still add to C(0)
            C0 += X.T @ X
            n0 += T
            continue

        # C(0): use all frames that participate in *some* lagged pair
        # Following PyEMMA convention: use frames [0, T-lag) for x(t)
        # and [lag, T) for x(t+τ).  For C(0) use the union = all T frames.
        C0 += X.T @ X
        n0 += T

        # C(τ): pairs  (x(t), x(t+τ))  for t = 0 … T-τ-1
        X_t   = X[:-lag]            # (T-lag, d)
        X_tl  = X[lag:]             # (T-lag, d)
        Ct += X_t.T @ X_tl
        nt += (T - lag)

    C0 /= n0
    Ct /= nt
    return C0, Ct


def solve_tica(C0, Ct, n_tics, eps):
    """
    Solve the symmetrised generalised eigenvalue problem:

        C_sym(τ) v = λ C_reg(0) v

    where C_sym = 0.5 (C(τ) + C(τ)^T)  and  C_reg = C(0) + ε I.

    Returns
    -------
    eigenvalues  : (n_tics,)
    eigenvectors : (d, n_tics)  — columns are the TICA eigenvectors
    timescales   : (n_tics,)    — implied timescales in *frames*  (multiply by dt for physical)
    """
    d = C0.shape[0]

    # Symmetrise the lagged covariance
    Ct_sym = 0.5 * (Ct + Ct.T)

    # Regularise C(0)
    C0_reg = C0 + eps * np.eye(d, dtype=np.float64)

    print(f"  Solving generalised eigenvalue problem  ({d} × {d}) …")
    t0 = time.time()

    # Use scipy.linalg.eigh for the symmetric generalised problem
    # eigh returns eigenvalues in *ascending* order
    eigenvalues, eigenvectors = linalg.eigh(Ct_sym, C0_reg)

    # Reverse to get descending order (slowest modes first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    elapsed = time.time() - t0
    print(f"  Eigenvalue problem solved in {elapsed:.1f} s")

    # Keep top n_tics
    eigenvalues  = eigenvalues[:n_tics]
    eigenvectors = eigenvectors[:, :n_tics]

    # Implied timescales  t_i = -τ / ln(λ_i)  (in frames)
    # Only valid if 0 < λ_i < 1
    timescales = np.full(n_tics, np.nan)
    valid = (eigenvalues > 0) & (eigenvalues < 1.0)
    timescales[valid] = -1.0 / np.log(eigenvalues[valid])   # in units of lag

    return eigenvalues, eigenvectors, timescales


def project_blocks(blocks, eigenvectors):
    """Project each block onto the TICA eigenvectors."""
    projections = []
    for X in blocks:
        projections.append(X @ eigenvectors)          # (T_i, n_tics)
    return projections


def lag_time_scan(blocks, lags, n_tics, eps):
    """
    Compute the top eigenvalues at many lag times for an implied-timescale plot.

    Returns
    -------
    scan : dict  with keys 'lags', 'eigenvalues', 'timescales'
        eigenvalues  : (n_lags, n_tics)
        timescales   : (n_lags, n_tics)   in frames
    """
    all_evals = []
    all_ts    = []
    valid_lags = []
    for lag in lags:
        max_frames = max(X.shape[0] for X in blocks)
        if lag >= max_frames:
            print(f"  lag={lag} exceeds max block length ({max_frames}), skipping")
            continue
        print(f"  lag = {lag:>4d} frames ({lag * DT_PS / 1000:.2f} ns) …", end="")
        C0, Ct = build_covariance_matrices(blocks, lag)
        evals, _, ts = solve_tica(C0, Ct, n_tics, eps)
        # Implied timescales in *frames* (then convert to physical units)
        # ts is already in units of "lag", convert to frames: t_frames = ts * lag
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


def plot_implied_timescales(scan, dt_ps, outdir):
    """Plot implied timescale vs lag time — the classic TICA/MSM diagnostic."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    lags_ns  = scan["lags"] * dt_ps / 1000.0
    ts_ns    = scan["timescales"] * dt_ps / 1000.0      # (n_lags, n_tics)
    n_show   = min(10, ts_ns.shape[1])

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_show):
        ax.plot(lags_ns, ts_ns[:, i], "o-", label=f"TIC {i+1}", linewidth=1.5, markersize=4)

    # Reference line: t = 5 × lag
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


def plot_eigenvalue_spectrum(eigenvalues, outdir):
    """Bar chart of the TICA eigenvalue spectrum."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, len(eigenvalues) + 1), eigenvalues, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("TIC index", fontsize=12)
    ax.set_ylabel("Eigenvalue λ", fontsize=12)
    ax.set_title("TICA Eigenvalue Spectrum — Mutant Nav1.5", fontsize=14)
    ax.axhline(0, color="red", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fpath = os.path.join(outdir, "tica_eigenvalue_spectrum.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved eigenvalue spectrum → {fpath}")


def plot_tic_projections(projections, block_names, dt_ps, outdir):
    """Scatter / time-series of TIC-1 vs TIC-2 and TIC-1(t)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        return

    # ── TIC-1 vs TIC-2 scatter ──
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

    # ── TIC-1 time series ──
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

    # ── TIC-1,2,3 2D histograms (combined) ──
    all_proj = np.vstack(projections)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [(0, 1), (0, 2), (1, 2)]
    labels = [("TIC-1", "TIC-2"), ("TIC-1", "TIC-3"), ("TIC-2", "TIC-3")]
    for ax, (ci, cj), (lx, ly) in zip(axes, pairs, labels):
        ax.hist2d(all_proj[:, ci], all_proj[:, cj], bins=100, cmap="inferno",
                  density=True)
        ax.set_xlabel(lx, fontsize=10)
        ax.set_ylabel(ly, fontsize=10)
    fig.suptitle("TICA 2D Histograms (all blocks) — Mutant Nav1.5", fontsize=14)
    plt.tight_layout()
    fpath = os.path.join(outdir, "tica_2d_histograms.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved 2D histograms → {fpath}")


def plot_cumulative_kinetic_variance(eigenvalues, outdir):
    """Cumulative kinetic variance captured by TICs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Kinetic variance ∝ λ_i^2 / (1 - λ_i^2)  for eigenvalues in (0,1)
    valid = (eigenvalues > 0) & (eigenvalues < 1.0)
    kv = np.zeros_like(eigenvalues)
    kv[valid] = eigenvalues[valid] ** 2 / (1 - eigenvalues[valid] ** 2)
    cumvar = np.cumsum(kv) / kv.sum() if kv.sum() > 0 else np.cumsum(kv)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, "o-", color="darkorange", linewidth=2, markersize=5)
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
    parser = argparse.ArgumentParser(description="Step 2 — TICA on MSM block features")
    parser.add_argument("--lag",    type=int,   default=DEFAULT_LAG,    help="Lag time in frames (default: %(default)s)")
    parser.add_argument("--ntics",  type=int,   default=DEFAULT_N_TICS, help="Number of TICs to keep (default: %(default)s)")
    parser.add_argument("--eps",    type=float, default=DEFAULT_EPS,    help="Tikhonov regularisation ε (default: %(default)s)")
    parser.add_argument("--no-scan", action="store_true",               help="Skip lag-time scan")
    args = parser.parse_args()

    lag    = args.lag
    n_tics = args.ntics
    eps    = args.eps

    OUT = os.path.join(BASE, "tica")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 65)
    print("  Step 2 — TICA  |  Mutant Nav1.5  (Martini 3 CG)")
    print("=" * 65)
    print(f"  Lag time   : {lag} frames  ({lag * DT_PS / 1000:.2f} ns)")
    print(f"  N TICs     : {n_tics}")
    print(f"  ε (reg.)   : {eps}")
    print(f"  Output dir : {OUT}")

    # ── 1. Load data ──
    print("\n[1] Loading per-block feature matrices …")
    raw_blocks = load_blocks(BASE, BLOCKS)

    # ── 2. Standardise ──
    print("\n[2] Computing global mean/std and standardising …")
    mean, std = global_mean_std(raw_blocks)
    blocks = standardise_blocks(raw_blocks, mean, std)
    # Free raw memory
    del raw_blocks

    n_const = (std == 1.0).sum() - ((std > 1e-12) & (std != 1.0)).sum()
    # actually count features with near-zero variance
    n_const = int((std < 1e-10).sum())  # features forced to std=1 to avoid div-by-zero
    print(f"  Features : {blocks[0].shape[1]}")
    print(f"  Constant features (std ≈ 0) set to 1.0: {n_const}")
    np.save(os.path.join(OUT, "feature_mean.npy"), mean)
    np.save(os.path.join(OUT, "feature_std.npy"),  std)

    # ── 3. Build covariance matrices ──
    print(f"\n[3] Building C(0) and C(τ={lag}) …")
    t0 = time.time()
    C0, Ct = build_covariance_matrices(blocks, lag)
    print(f"  C(0) shape : {C0.shape}")
    print(f"  C(τ) shape : {Ct.shape}")
    print(f"  Built in {time.time() - t0:.1f} s")
    np.save(os.path.join(OUT, "C0.npy"), C0)
    np.save(os.path.join(OUT, "Ct.npy"), Ct)

    # ── 4. Solve TICA ──
    print(f"\n[4] Solving TICA eigenvalue problem …")
    eigenvalues, eigenvectors, timescales_frames = solve_tica(C0, Ct, n_tics, eps)

    timescales_ns = timescales_frames * lag * DT_PS / 1000.0

    print(f"\n  Top {min(10, n_tics)} TICA eigenvalues & implied timescales:")
    print(f"  {'TIC':>5s}  {'λ':>10s}  {'t (frames)':>12s}  {'t (ns)':>10s}")
    print(f"  {'-'*5:>5s}  {'-'*10:>10s}  {'-'*12:>12s}  {'-'*10:>10s}")
    for i in range(min(10, n_tics)):
        tf = timescales_frames[i] * lag if not np.isnan(timescales_frames[i]) else np.nan
        tn = timescales_ns[i]
        print(f"  {i+1:5d}  {eigenvalues[i]:10.6f}  {tf:12.1f}  {tn:10.2f}")

    np.save(os.path.join(OUT, "tica_eigenvalues.npy"),  eigenvalues)
    np.save(os.path.join(OUT, "tica_eigenvectors.npy"), eigenvectors)
    np.save(os.path.join(OUT, "tica_timescales_frames.npy"), timescales_frames * lag)
    np.save(os.path.join(OUT, "tica_timescales_ns.npy"), timescales_ns)

    # ── 5. Project blocks ──
    print(f"\n[5] Projecting blocks onto {n_tics} TICs …")
    projections = project_blocks(blocks, eigenvectors)
    for bn, proj in zip(BLOCKS, projections):
        fpath = os.path.join(OUT, f"{bn}_tica.npy")
        np.save(fpath, proj.astype(np.float32))
        print(f"  {bn:12s}  →  {fpath}   shape {proj.shape}")

    # Also save a concatenated projection (all blocks)
    all_proj = np.vstack(projections).astype(np.float32)
    np.save(os.path.join(OUT, "tica_all_blocks.npy"), all_proj)
    print(f"  All blocks  →  tica_all_blocks.npy   shape {all_proj.shape}")

    # ── 6. Lag-time scan ──
    if not args.no_scan:
        print(f"\n[6] Lag-time scan for implied timescales …")
        # Filter out lags larger than shortest block
        min_block = min(X.shape[0] for X in blocks)
        valid_lags = [l for l in LAG_SCAN if l < min_block // 2]
        scan = lag_time_scan(blocks, valid_lags, n_tics, eps)
        np.savez(os.path.join(OUT, "tica_lag_scan.npz"),
                 lags=scan["lags"],
                 eigenvalues=scan["eigenvalues"],
                 timescales=scan["timescales"])
        plot_implied_timescales(scan, DT_PS, OUT)
    else:
        print("\n[6] Lag-time scan skipped (--no-scan)")

    # ── 7. Diagnostic plots ──
    print(f"\n[7] Generating diagnostic plots …")
    plot_eigenvalue_spectrum(eigenvalues, OUT)
    plot_tic_projections(projections, BLOCKS, DT_PS, OUT)
    plot_cumulative_kinetic_variance(eigenvalues, OUT)

    # ── 8. Summary ──
    summary_path = os.path.join(OUT, "tica_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Step 2 — TICA Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Lag time       : {lag} frames ({lag * DT_PS / 1000:.2f} ns)\n")
        f.write(f"N TICs kept    : {n_tics}\n")
        f.write(f"Regularisation : ε = {eps}\n")
        f.write(f"Feature dim    : {blocks[0].shape[1]}\n")
        f.write(f"Constant feats : {n_const}\n\n")
        f.write("Block shapes (standardised):\n")
        for bn, X in zip(BLOCKS, blocks):
            f.write(f"  {bn}: {X.shape}\n")
        f.write(f"\nEigenvalues & Implied Timescales:\n")
        f.write(f"  {'TIC':>5s}  {'λ':>12s}  {'t (ns)':>10s}\n")
        for i in range(n_tics):
            f.write(f"  {i+1:5d}  {eigenvalues[i]:12.6f}  {timescales_ns[i]:10.2f}\n")
        f.write(f"\nOutputs saved to: {OUT}/\n")
        f.write(f"  tica_eigenvalues.npy       — ({n_tics},)\n")
        f.write(f"  tica_eigenvectors.npy      — ({blocks[0].shape[1]}, {n_tics})\n")
        f.write(f"  tica_timescales_frames.npy — ({n_tics},)\n")
        f.write(f"  tica_timescales_ns.npy     — ({n_tics},)\n")
        f.write(f"  feature_mean.npy           — ({blocks[0].shape[1]},)\n")
        f.write(f"  feature_std.npy            — ({blocks[0].shape[1]},)\n")
        f.write(f"  C0.npy, Ct.npy             — covariance matrices\n")
        for bn in BLOCKS:
            f.write(f"  {bn}_tica.npy          — per-block TICA projection\n")
        f.write(f"  tica_all_blocks.npy        — concatenated projection\n")
        if not args.no_scan:
            f.write(f"  tica_lag_scan.npz           — lag-time scan data\n")
    print(f"\n  Summary → {summary_path}")
    print("\n  Step 2 (TICA) complete.")


if __name__ == "__main__":
    main()
