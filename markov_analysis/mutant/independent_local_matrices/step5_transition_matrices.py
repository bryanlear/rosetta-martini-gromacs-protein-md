#!/usr/bin/env python
"""
Step 5 — Local Transition Probability Matrix Estimation
=========================================================
MSM pipeline for Mutant Nav1.5 (Martini 3 CG).

**All three blocks are included** for the mutant pipeline.
(Block 2 was excluded in the WT due to poor internal connectivity.)

Row-normalises the local transition count matrices to produce
transition probability matrices T(τ):

    T_ij(τ) = C_ij(τ) / Σ_j C_ij(τ)

Each element T_ij(τ) is the conditional probability of transitioning
to microstate S_j given the system is currently in S_i.

Two estimators are produced:
  (a) **Non-reversible** — direct row-normalisation of C.
  (b) **Reversible** — row-normalisation of C_sym = C + C^T,
      enforcing detailed balance (simplest reversible MLE).

The script also:
  1. Restricts to the largest strongly connected component (SCC) to
     guarantee an ergodic, irreducible Markov chain.
  2. Computes eigenvalues, implied timescales, and the stationary
     distribution π from the reversible T.
  3. Validates row-stochasticity and detailed balance.
  4. Performs a Chapman-Kolmogorov self-consistency test.
  5. Generates diagnostic plots.

Inputs:   count_matrices/msm_block{1,3}_C_tau{lag}.npy  (from Step 4)
Outputs:  transition_matrices/                           (all outputs)

Usage:
    conda activate bio_env
    python step5_transition_matrices.py               # default lag=10
    python step5_transition_matrices.py --lag 20
    python step5_transition_matrices.py --no-restrict  # keep all states
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
BASE       = os.path.dirname(os.path.abspath(__file__))
COUNT_DIR  = os.path.join(BASE, "count_matrices")
# Block 2 excluded due to poor internal connectivity
BLOCKS     = ["msm_block1", "msm_block2", "msm_block3"]
DT_PS      = 100.0                        # ps per frame

DEFAULT_LAG      = 10
DEFAULT_N_STATES = 200
DEFAULT_N_EIGEN  = 20


# ────────────────────────────────────────────────────────────────
# COUNT MATRIX LOADING & AGGREGATION
# ────────────────────────────────────────────────────────────────

def load_and_aggregate(count_dir, block_names, lag, n_states):
    """Load per-block count matrices and sum."""
    C_agg = np.zeros((n_states, n_states), dtype=np.int64)
    for bn in block_names:
        fpath = os.path.join(count_dir, f"{bn}_C_tau{lag}.npy")
        C = np.load(fpath)
        C_agg += C
        rs = C.sum(axis=1)
        print(f"  {bn:12s}  counts = {C.sum():>6,}  "
              f"visited = {np.sum(rs > 0):>3d}")
    print(f"  {'aggregated':12s}  counts = {C_agg.sum():>6,}  "
          f"visited = {np.sum(C_agg.sum(axis=1) > 0):>3d}")
    return C_agg


# ────────────────────────────────────────────────────────────────
# LARGEST STRONGLY CONNECTED COMPONENT
# ────────────────────────────────────────────────────────────────

def restrict_to_largest_scc(C):
    """Restrict count matrix to the largest strongly connected component.

    Returns
    -------
    C_scc   : ndarray (m, m)  — restricted count matrix
    scc_map : ndarray (m,)    — original state indices in the SCC
    full_to_scc : ndarray (n,) — mapping from full index → SCC index
                                  (-1 if not in SCC)
    """
    n = C.shape[0]
    G = csr_matrix((C > 0).astype(int))
    n_scc, scc_labels = connected_components(G, directed=True,
                                              connection="strong")
    scc_sizes = np.bincount(scc_labels)
    largest_id = np.argmax(scc_sizes)
    scc_mask = scc_labels == largest_id
    scc_map = np.where(scc_mask)[0]
    m = len(scc_map)

    C_scc = C[np.ix_(scc_map, scc_map)]

    full_to_scc = np.full(n, -1, dtype=np.int64)
    full_to_scc[scc_map] = np.arange(m)

    print(f"  Full states: {n},  SCC states: {m}  "
          f"(dropped {n - m} disconnected states)")
    if n - m > 0:
        dropped = np.where(~scc_mask)[0]
        pops = C.sum(axis=1)
        print(f"  Dropped states: {dropped.tolist()}")
        print(f"  Their row-sum counts: {pops[dropped].tolist()}")

    return C_scc, scc_map, full_to_scc


# ────────────────────────────────────────────────────────────────
# TRANSITION MATRIX ESTIMATION
# ────────────────────────────────────────────────────────────────

def row_normalise(C):
    """Non-reversible row-stochastic T from count matrix C."""
    C_f = C.astype(np.float64)
    row_sums = C_f.sum(axis=1)
    zero_rows = row_sums == 0
    row_sums[zero_rows] = 1.0
    T = C_f / row_sums[:, None]
    T[zero_rows, :] = 1.0 / C.shape[0]
    return T


def reversible_T(C):
    """Reversible transition matrix via symmetrisation.

    T_rev_ij = (C_ij + C_ji) / Σ_j (C_ij + C_ji)

    This enforces detailed balance: π_i T_ij = π_j T_ji
    where π_i ∝ Σ_j (C_ij + C_ji).
    """
    C_sym = (C + C.T).astype(np.float64)
    row_sums = C_sym.sum(axis=1)
    zero_rows = row_sums == 0
    row_sums[zero_rows] = 1.0
    T = C_sym / row_sums[:, None]
    T[zero_rows, :] = 1.0 / C.shape[0]
    return T


def stationary_distribution_from_counts(C_sym):
    """Stationary distribution from symmetrised counts: π_i ∝ Σ_j C_sym_ij."""
    pi = C_sym.astype(np.float64).sum(axis=1)
    pi /= pi.sum()
    return pi


# ────────────────────────────────────────────────────────────────
# EIGENANALYSIS
# ────────────────────────────────────────────────────────────────

def eigendecomposition(T, n_eigen):
    """Right eigendecomposition of T.

    Returns eigenvalues (real, descending) and right eigenvectors.
    For the reversible case, eigenvalues are guaranteed real.
    """
    eigenvalues, eigvecs = linalg.eig(T, left=False, right=True)
    ev_real = np.real(eigenvalues)
    idx = np.argsort(-ev_real)
    ev_sorted = ev_real[idx[:n_eigen]]
    vr_sorted = np.real(eigvecs[:, idx[:n_eigen]])
    return ev_sorted, vr_sorted


def implied_timescales(eigenvalues, lag, dt_ps=100.0):
    """t_i = -τ / ln(λ_i)  in ns.  Skips λ_1 ≈ 1 (stationary)."""
    ts = np.full_like(eigenvalues, np.nan, dtype=np.float64)
    for i in range(1, len(eigenvalues)):
        lam = eigenvalues[i]
        if 0 < lam < 1:
            ts[i] = -lag * dt_ps / (1000.0 * np.log(lam))
        elif lam >= 1.0:
            ts[i] = np.inf
    return ts


# ────────────────────────────────────────────────────────────────
# VALIDATION
# ────────────────────────────────────────────────────────────────

def validate_transition_matrix(T, label="T"):
    """Check row-stochasticity and non-negativity."""
    checks = {}

    # Non-negativity
    min_val = T.min()
    checks["min_element"] = float(min_val)
    checks["non_negative"] = bool(min_val >= 0)

    # Row sums
    row_sums = T.sum(axis=1)
    max_dev = np.abs(row_sums - 1.0).max()
    checks["max_row_sum_deviation"] = float(max_dev)
    checks["row_stochastic"] = bool(max_dev < 1e-12)

    print(f"  [{label}] Non-negative     : "
          f"{'✓' if checks['non_negative'] else '✗'}  "
          f"(min = {min_val:.2e})")
    print(f"  [{label}] Row-stochastic   : "
          f"{'✓' if checks['row_stochastic'] else '✗'}  "
          f"(max |Σ_j T_ij − 1| = {max_dev:.2e})")
    return checks


def validate_detailed_balance(T, pi, label="T_rev"):
    """Check detailed balance: π_i T_ij ≈ π_j T_ji."""
    n = T.shape[0]
    flux_ij = pi[:, None] * T           # π_i T_ij
    flux_ji = pi[None, :] * T.T         # π_j T_ji  (same as flux_ij transposed)
    diff = np.abs(flux_ij - flux_ji)
    max_diff = diff.max()
    mean_diff = diff.mean()

    ok = max_diff < 1e-10
    print(f"  [{label}] Detailed balance : "
          f"{'✓' if ok else '✗'}  "
          f"(max |π_i T_ij − π_j T_ji| = {max_diff:.2e}, "
          f"mean = {mean_diff:.2e})")
    return {"max_db_violation": float(max_diff),
            "mean_db_violation": float(mean_diff),
            "detailed_balance": bool(ok)}


def chapman_kolmogorov_test(block_labels, scc_map, full_to_scc,
                            n_scc, lag, n_steps, dt_ps):
    """Chapman-Kolmogorov self-consistency test.

    Compare T(nτ) estimated directly from data vs T(τ)^n.
    Reports the L2 norm of the difference for each n.
    """
    print(f"  CK test: comparing T(nτ) vs T(τ)^n for n = 1..{n_steps}")

    # Build T(τ) from aggregated counts at base lag (SCC-restricted)
    from step4_count_matrices import build_count_matrix
    C_base = np.zeros((200, 200), dtype=np.int64)
    for bl in block_labels:
        C_base += build_count_matrix(bl, 200, lag)
    C_base_scc = C_base[np.ix_(scc_map, scc_map)]
    T_base = reversible_T(C_base_scc)

    results = {"n": [], "lag_n": [], "l2_norm": [], "linf_norm": []}
    T_power = np.eye(n_scc)

    for n in range(1, n_steps + 1):
        T_power = T_power @ T_base

        # T(nτ) directly from data
        nlag = n * lag
        C_n = np.zeros((200, 200), dtype=np.int64)
        for bl in block_labels:
            if nlag < len(bl):
                C_n += build_count_matrix(bl, 200, nlag)
        C_n_scc = C_n[np.ix_(scc_map, scc_map)]

        if C_n_scc.sum() == 0:
            break

        T_n = reversible_T(C_n_scc)

        diff = T_n - T_power
        l2 = np.sqrt(np.mean(diff ** 2))
        linf = np.abs(diff).max()

        results["n"].append(n)
        results["lag_n"].append(nlag)
        results["l2_norm"].append(l2)
        results["linf_norm"].append(linf)

        print(f"    n={n:2d}  τ={nlag:4d} ({nlag * dt_ps / 1000:.1f} ns)  "
              f"L2 = {l2:.6f}  L∞ = {linf:.6f}")

    return results


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
        print("  [WARN] matplotlib not available")
        return None


def plot_eigenvalue_spectrum(ev, lag, outdir):
    plt = _get_plt()
    if plt is None:
        return
    n = len(ev)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, n + 1), ev, color="steelblue", edgecolor="white",
           width=0.8)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Eigenvalue index", fontsize=12)
    ax.set_ylabel("λ", fontsize=12)
    ax.set_title(f"MSM Eigenvalue Spectrum (τ={lag}, reversible, "
                 f"all blocks) — Mutant Nav1.5", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fpath = os.path.join(outdir, "eigenvalue_spectrum.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved eigenvalue spectrum → {fpath}")


def plot_implied_timescales_table(ev, lag, dt_ps, outdir):
    plt = _get_plt()
    if plt is None:
        return
    ts = implied_timescales(ev, lag, dt_ps)
    n_show = min(15, len(ts) - 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(2, n_show + 2)
    vals = ts[1:n_show + 1]
    valid = np.isfinite(vals)
    ax.bar([i for i, v in zip(x, valid) if v],
           [v for v, ok in zip(vals, valid) if ok],
           color="darkorange", edgecolor="white", width=0.7)
    ax.set_xlabel("Process index", fontsize=12)
    ax.set_ylabel("Implied timescale (ns)", fontsize=12)
    ax.set_title(f"Implied Timescales (τ={lag}, all blocks) — Mutant Nav1.5",
                 fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fpath = os.path.join(outdir, "implied_timescales.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved ITS bar chart → {fpath}")


def plot_stationary_distribution(pi, scc_map, outdir):
    plt = _get_plt()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sorted
    ax = axes[0]
    sorted_pi = np.sort(pi)[::-1]
    ax.bar(range(len(sorted_pi)), sorted_pi, color="darkorange",
           edgecolor="none", width=1.0)
    ax.set_xlabel("State rank", fontsize=12)
    ax.set_ylabel("π", fontsize=12)
    ax.set_title("Stationary Distribution (sorted)", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    # By original state index
    ax = axes[1]
    ax.bar(scc_map, pi, color="steelblue", edgecolor="none", width=1.0)
    ax.set_xlabel("Original state index", fontsize=12)
    ax.set_ylabel("π", fontsize=12)
    ax.set_title("Stationary Distribution (by state)", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Stationary Distribution — Mutant Nav1.5 (all blocks)",
                 fontsize=14)
    plt.tight_layout()
    fpath = os.path.join(outdir, "stationary_distribution.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved stationary distribution → {fpath}")
    return


def plot_transition_matrix_heatmap(T, outdir, tag="reversible"):
    plt = _get_plt()
    if plt is None:
        return
    T_plot = T.copy()
    T_plot[T_plot == 0] = np.nan

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.log10(T_plot), origin="lower", cmap="viridis",
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="log₁₀(T_ij)")
    ax.set_xlabel("State j", fontsize=11)
    ax.set_ylabel("State i", fontsize=11)
    ax.set_title(f"Transition Matrix T(τ) [{tag}] — Mutant Nav1.5", fontsize=13)
    plt.tight_layout()
    fpath = os.path.join(outdir, f"T_heatmap_{tag}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved T heatmap → {fpath}")


def plot_ck_test(ck_results, dt_ps, outdir):
    plt = _get_plt()
    if plt is None:
        return
    ns = np.array(ck_results["n"])
    lag_ns = np.array(ck_results["lag_n"]) * dt_ps / 1000.0
    l2 = np.array(ck_results["l2_norm"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lag_ns, l2, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.set_xlabel("Effective lag nτ (ns)", fontsize=12)
    ax.set_ylabel("L₂ error ‖T(nτ) − T(τ)ⁿ‖", fontsize=12)
    ax.set_title("Chapman-Kolmogorov Test — Mutant Nav1.5 (all blocks)",
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(outdir, "ck_test.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved CK test plot → {fpath}")


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 5 — Transition probability matrix estimation "
                    "(all blocks only)")
    parser.add_argument("--lag",        type=int, default=DEFAULT_LAG,
                        help="Lag time in frames (default: %(default)s)")
    parser.add_argument("--n-states",   type=int, default=DEFAULT_N_STATES,
                        help="Total microstates (default: %(default)s)")
    parser.add_argument("--n-eigen",    type=int, default=DEFAULT_N_EIGEN,
                        help="Eigenvalues to compute (default: %(default)s)")
    parser.add_argument("--no-restrict", action="store_true",
                        help="Do NOT restrict to largest SCC")
    parser.add_argument("--ck-steps",   type=int, default=10,
                        help="CK test: max n for T(τ)^n (default: 10)")
    args = parser.parse_args()

    lag       = args.lag
    n_states  = args.n_states
    n_eigen   = args.n_eigen
    restrict  = not args.no_restrict
    ck_steps  = args.ck_steps

    OUT = os.path.join(BASE, "transition_matrices")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 70)
    print("  Step 5 — Transition Probability Matrices  |  Mutant Nav1.5  (CG)")
    print("=" * 70)
    print(f"  Blocks used      : {', '.join(BLOCKS)}")
    print(f"  Lag time         : {lag} frames ({lag * DT_PS / 1000:.2f} ns)")
    print(f"  Microstates      : {n_states}")
    print(f"  Restrict to SCC  : {'yes' if restrict else 'no'}")
    print(f"  Eigenvalues      : {n_eigen}")
    print(f"  CK test steps    : {ck_steps}")
    print(f"  Output dir       : {OUT}")

    # ── 1. Load & aggregate counts ──
    print(f"\n[1] Loading count matrices …")
    C_agg = load_and_aggregate(COUNT_DIR, BLOCKS, lag, n_states)
    np.save(os.path.join(OUT, f"C_agg_tau{lag}.npy"), C_agg)

    # ── 2. Restrict to largest SCC ──
    if restrict:
        print(f"\n[2] Restricting to largest strongly connected component …")
        C_scc, scc_map, full_to_scc = restrict_to_largest_scc(C_agg)
        m = C_scc.shape[0]
    else:
        print(f"\n[2] Using all {n_states} states (no restriction)")
        C_scc = C_agg
        m = n_states
        scc_map = np.arange(n_states)
        full_to_scc = np.arange(n_states)

    np.save(os.path.join(OUT, "scc_state_map.npy"), scc_map)
    np.save(os.path.join(OUT, "full_to_scc_map.npy"), full_to_scc)
    np.save(os.path.join(OUT, f"C_scc_tau{lag}.npy"), C_scc)

    n_eigen = min(n_eigen, m)

    # ── 3. Non-reversible T ──
    print(f"\n[3] Estimating non-reversible T (row-normalisation) …")
    T_nonrev = row_normalise(C_scc)
    np.save(os.path.join(OUT, f"T_nonrev_tau{lag}.npy"), T_nonrev)
    validate_transition_matrix(T_nonrev, "T_nonrev")

    ev_nonrev, vr_nonrev = eigendecomposition(T_nonrev, n_eigen)
    ts_nonrev = implied_timescales(ev_nonrev, lag, DT_PS)
    np.save(os.path.join(OUT, f"eigenvalues_nonrev_tau{lag}.npy"), ev_nonrev)

    print(f"  Non-reversible top-5 ITS (ns):", end="")
    for i in range(1, min(6, n_eigen)):
        s = f"  {ts_nonrev[i]:.1f}" if np.isfinite(ts_nonrev[i]) else "  ∞"
        print(s, end="")
    print()

    # ── 4. Reversible T ──
    print(f"\n[4] Estimating reversible T (symmetrised) …")
    T_rev = reversible_T(C_scc)
    np.save(os.path.join(OUT, f"T_rev_tau{lag}.npy"), T_rev)
    checks_rev = validate_transition_matrix(T_rev, "T_rev")

    # Stationary distribution
    C_sym = (C_scc + C_scc.T).astype(np.float64)
    pi = stationary_distribution_from_counts(C_sym)
    np.save(os.path.join(OUT, f"stationary_dist_tau{lag}.npy"), pi)

    print(f"  π: min = {pi.min():.6f}, max = {pi.max():.6f}, "
          f"sum = {pi.sum():.12f}")

    # Detailed balance check
    db_checks = validate_detailed_balance(T_rev, pi)

    # Eigendecomposition
    ev_rev, vr_rev = eigendecomposition(T_rev, n_eigen)
    ts_rev = implied_timescales(ev_rev, lag, DT_PS)
    np.save(os.path.join(OUT, f"eigenvalues_rev_tau{lag}.npy"), ev_rev)
    np.save(os.path.join(OUT, f"eigenvectors_rev_tau{lag}.npy"), vr_rev)
    np.save(os.path.join(OUT, f"timescales_rev_tau{lag}.npy"), ts_rev)

    print(f"\n  Reversible T — eigenvalues & implied timescales:")
    print(f"  {'#':>3s}  {'λ':>12s}  {'ITS (ns)':>12s}")
    print(f"  {'-' * 3:>3s}  {'-' * 12:>12s}  {'-' * 12:>12s}")
    for i in range(min(n_eigen, 15)):
        ts_s = (f"{ts_rev[i]:.2f}" if np.isfinite(ts_rev[i])
                else ("∞" if ts_rev[i] == np.inf else "—"))
        print(f"  {i + 1:3d}  {ev_rev[i]:12.6f}  {ts_s:>12s}")

    # ── 5. Chapman-Kolmogorov test ──
    print(f"\n[5] Chapman-Kolmogorov self-consistency test …")
    # Load block labels (excluding block2)
    clust_dir = os.path.join(BASE, "clustering")
    block_labels = []
    for bn in BLOCKS:
        bl = np.load(os.path.join(clust_dir, f"{bn}_labels.npy")).astype(np.int32)
        block_labels.append(bl)

    ck = chapman_kolmogorov_test(block_labels, scc_map, full_to_scc,
                                 m, lag, ck_steps, DT_PS)
    if ck["n"]:
        np.savez(os.path.join(OUT, "ck_test_results.npz"),
                 n=ck["n"], lag_n=ck["lag_n"],
                 l2_norm=ck["l2_norm"], linf_norm=ck["linf_norm"])

    # ── 6. Remap block labels to SCC indices ──
    if restrict:
        print(f"\n[6] Remapping block labels to SCC state space …")
        for bn, bl in zip(BLOCKS, block_labels):
            bl_scc = full_to_scc[bl]
            n_mapped = np.sum(bl_scc >= 0)
            n_dropped = np.sum(bl_scc < 0)
            fpath = os.path.join(OUT, f"{bn}_labels_scc.npy")
            np.save(fpath, bl_scc)
            print(f"  {bn}: {n_mapped} mapped, {n_dropped} dropped (-1)")
    else:
        print(f"\n[6] No remapping needed (all states kept)")

    # ── 7. Diagnostic plots ──
    print(f"\n[7] Generating diagnostic plots …")
    plot_eigenvalue_spectrum(ev_rev, lag, OUT)
    plot_implied_timescales_table(ev_rev, lag, DT_PS, OUT)
    plot_stationary_distribution(pi, scc_map, OUT)
    plot_transition_matrix_heatmap(T_rev, OUT, "reversible")
    plot_transition_matrix_heatmap(T_nonrev, OUT, "non-reversible")
    if ck["n"]:
        plot_ck_test(ck, DT_PS, OUT)

    # ── 8. Summary ──
    print(f"\n[8] Writing summary …")
    summary_path = os.path.join(OUT, "transition_matrix_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Step 5 — Transition Probability Matrix Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Blocks used        : {', '.join(BLOCKS)}\n")
        f.write(f"Lag time           : {lag} frames "
                f"({lag * DT_PS / 1000:.2f} ns)\n")
        f.write(f"Full microstates   : {n_states}\n")
        f.write(f"SCC restriction    : {'yes' if restrict else 'no'}\n")
        f.write(f"Active states (SCC): {m}\n")
        if restrict and m < n_states:
            dropped = np.where(full_to_scc < 0)[0]
            f.write(f"Dropped states     : {dropped.tolist()}\n")
        f.write(f"\nCount matrix (all blocks aggregated, SCC):\n")
        f.write(f"  Total counts     : {C_scc.sum():,}\n")
        f.write(f"  Non-zero entries : {np.count_nonzero(C_scc):,}\n")
        f.write(f"  Self-transition  : "
                f"{np.trace(C_scc) / max(C_scc.sum(), 1):.1%}\n")

        f.write(f"\nNon-reversible T:\n")
        f.write(f"  Row-stochastic   : ✓\n")
        f.write(f"  Top-5 ITS (ns)   : ")
        top5 = [f"{ts_nonrev[i]:.1f}" if np.isfinite(ts_nonrev[i]) else "∞"
                for i in range(1, min(6, n_eigen))]
        f.write(", ".join(top5) + "\n")

        f.write(f"\nReversible T (symmetrised):\n")
        f.write(f"  Row-stochastic   : ✓\n")
        f.write(f"  Detailed balance : "
                f"{'✓' if db_checks['detailed_balance'] else '✗'}  "
                f"(max viol. = {db_checks['max_db_violation']:.2e})\n")
        f.write(f"  π range          : [{pi.min():.6f}, {pi.max():.6f}]\n")

        f.write(f"\nReversible eigenvalues & implied timescales (τ={lag}):\n")
        f.write(f"  {'#':>3s}  {'λ':>12s}  {'ITS (ns)':>12s}\n")
        for i in range(min(n_eigen, 20)):
            ts_s = (f"{ts_rev[i]:.2f}" if np.isfinite(ts_rev[i])
                    else ("inf" if ts_rev[i] == np.inf else "nan"))
            f.write(f"  {i + 1:3d}  {ev_rev[i]:12.6f}  {ts_s:>12s}\n")

        if ck["n"]:
            f.write(f"\nChapman-Kolmogorov test (T(nτ) vs T(τ)^n):\n")
            f.write(f"  {'n':>3s}  {'nτ (ns)':>8s}  {'L2':>10s}  {'L∞':>10s}\n")
            for i in range(len(ck["n"])):
                f.write(f"  {ck['n'][i]:3d}  "
                        f"{ck['lag_n'][i] * DT_PS / 1000:8.1f}  "
                        f"{ck['l2_norm'][i]:10.6f}  "
                        f"{ck['linf_norm'][i]:10.6f}\n")

        f.write(f"\nOutputs in {OUT}/:\n")
        f.write(f"  C_agg_tau{lag}.npy          — aggregated counts (full)\n")
        f.write(f"  C_scc_tau{lag}.npy              — SCC-restricted counts\n")
        f.write(f"  scc_state_map.npy              — SCC→original index map\n")
        f.write(f"  full_to_scc_map.npy            — original→SCC index map\n")
        f.write(f"  T_nonrev_tau{lag}.npy           — non-reversible T\n")
        f.write(f"  T_rev_tau{lag}.npy              — reversible T\n")
        f.write(f"  eigenvalues_rev_tau{lag}.npy    — reversible eigenvalues\n")
        f.write(f"  eigenvectors_rev_tau{lag}.npy   — reversible eigenvectors\n")
        f.write(f"  timescales_rev_tau{lag}.npy     — implied timescales\n")
        f.write(f"  stationary_dist_tau{lag}.npy    — π\n")
        for bn in BLOCKS:
            f.write(f"  {bn}_labels_scc.npy       — SCC-remapped labels\n")
        f.write(f"  ck_test_results.npz            — CK test data\n")
        f.write(f"\nPlots:\n")
        f.write(f"  eigenvalue_spectrum.png\n")
        f.write(f"  implied_timescales.png\n")
        f.write(f"  stationary_distribution.png\n")
        f.write(f"  T_heatmap_reversible.png\n")
        f.write(f"  T_heatmap_non-reversible.png\n")
        f.write(f"  ck_test.png\n")

    print(f"  Summary → {summary_path}")
    print(f"\n  Step 5 (transition matrices) complete.")


if __name__ == "__main__":
    main()
