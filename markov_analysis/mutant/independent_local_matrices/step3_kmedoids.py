#!/usr/bin/env python
"""
Step 3 — k-Medoids Microstate Discretisation
==============================================
MSM pipeline for Mutant Nav1.5 (Martini 3 CG).

Partitions the TICA projections from Step 2 into N discrete geometric
microstates using the FasterPAM k-medoids algorithm (Schubert & Rousseeuw,
J Stat Softw 2022).

k-Medoids is preferred over k-Means for MSM microstate assignment because
medoids are actual data points (real conformations), making them
physically interpretable as representative structures.

Pipeline:
  1. Load per-block TICA projections from Step 2.
  2. Optionally select first `--ntics` TICs for clustering.
  3. Compute the pairwise Euclidean distance matrix (condensed form).
  4. Cluster scan: run FasterPAM for a range of k values, recording
     (a) total dissimilarity (loss); (b) silhouette score.
  5. At the chosen k (default k=200), produce final cluster assignments
     for each block, medoid indices, and summary statistics.
  6. Save per-block label sequences (needed for Step 4 transition counting).
  7. Generate diagnostic plots (elbow, silhouette, population histogram,
     cluster map on TIC-1 / TIC-2).

Inputs:   tica/msm_block{1,2,3}_tica.npy   (from Step 2)
Outputs:  clustering/                        (all outputs)

Usage:
    conda activate bio_env
    python step3_kmedoids.py                       # defaults: k=200
    python step3_kmedoids.py --k 150               # specific k
    python step3_kmedoids.py --k 200 --scan        # run k-scan first
    python step3_kmedoids.py --ntics 10            # cluster on first 10 TICs
    python step3_kmedoids.py --scan-only           # k-scan only, no final fit
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
TICA_DIR = os.path.join(BASE, "tica")
BLOCKS   = ["msm_block1", "msm_block2", "msm_block3"]
DT_PS    = 100.0                         # trajectory stride (ps per frame)

# Defaults (overridable via CLI)
DEFAULT_K        = 200                   # number of microstates
DEFAULT_N_TICS   = None                  # None → use all available TICs
DEFAULT_SEED     = 42                    # random seed for reproducibility
DEFAULT_MAX_ITER = 300                   # max PAM iterations

# k-scan range
K_SCAN_DEFAULT = [25, 50, 75, 100, 150, 200, 250, 300, 400, 500]


# ────────────────────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────────────────────

def load_tica_blocks(tica_dir, block_names, n_tics=None):
    """Load per-block TICA projections, optionally truncating to n_tics."""
    data = []
    for bn in block_names:
        fpath = os.path.join(tica_dir, f"{bn}_tica.npy")
        X = np.load(fpath).astype(np.float64)
        if n_tics is not None:
            X = X[:, :n_tics]
        print(f"  Loaded {bn:12s}  shape {X.shape}")
        data.append(X)
    return data


# ────────────────────────────────────────────────────────────────
# DISTANCE MATRIX
# ────────────────────────────────────────────────────────────────

def compute_distance_matrix(X, metric="euclidean"):
    """Compute the full square pairwise distance matrix.

    Returns the square form (N × N) as required by the kmedoids package.
    """
    N = X.shape[0]
    mem_gb = N * N * 8 / 1e9  # float64, square
    print(f"  N = {N},  distance pairs = {N*(N-1)//2:,}")
    print(f"  Estimated memory (square, f64): {mem_gb:.2f} GB")

    t0 = time.time()
    D_cond = pdist(X, metric=metric)
    D_sq = squareform(D_cond)
    del D_cond
    dt = time.time() - t0
    print(f"  Distance matrix computed in {dt:.1f} s  shape {D_sq.shape}")
    return D_sq


# ────────────────────────────────────────────────────────────────
# K-MEDOIDS CLUSTERING
# ────────────────────────────────────────────────────────────────

def run_kmedoids(D_sq, k, max_iter=300, seed=42):
    """Run FasterPAM k-medoids on a square distance matrix.

    Parameters
    ----------
    D_sq : ndarray, shape (N, N)
        Square pairwise distance matrix.
    k : int
        Number of clusters.
    max_iter : int
        Maximum PAM iterations.
    seed : int
        Random seed.

    Returns
    -------
    result : KMedoidsResult
        .labels  — cluster assignment per point
        .medoids — indices of medoid points
        .loss    — total dissimilarity (sum of distances to medoids)
    """
    import kmedoids
    result = kmedoids.fasterpam(D_sq, k, max_iter=max_iter, random_state=seed)
    return result


def silhouette_from_square(D_sq, labels, sample_size=5000, seed=42):
    """Compute silhouette score; sub-samples if N is large for speed."""
    from sklearn.metrics import silhouette_score

    N = len(labels)
    if N <= sample_size:
        return silhouette_score(D_sq, labels, metric="precomputed")

    # Sub-sample for speed
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, sample_size, replace=False)
    return silhouette_score(D_sq[np.ix_(idx, idx)], labels[idx],
                            metric="precomputed")


# ────────────────────────────────────────────────────────────────
# K-SCAN
# ────────────────────────────────────────────────────────────────

def k_scan(D_sq, k_values, max_iter=300, seed=42, sample_sil=5000):
    """Run FasterPAM over a range of k values, recording loss & silhouette."""
    results = {
        "k_values":   [],
        "loss":       [],
        "silhouette": [],
        "n_iter":     [],
        "wall_time":  [],
    }

    N = D_sq.shape[0]

    for k in k_values:
        if k >= N:
            print(f"  k={k} ≥ N={N}, skipping")
            continue

        t0 = time.time()
        res = run_kmedoids(D_sq, k, max_iter=max_iter, seed=seed)
        wall = time.time() - t0

        sil = silhouette_from_square(D_sq, np.asarray(res.labels),
                                     sample_size=sample_sil, seed=seed)

        results["k_values"].append(k)
        results["loss"].append(res.loss)
        results["silhouette"].append(sil)
        results["n_iter"].append(res.n_iter)
        results["wall_time"].append(wall)

        print(f"  k = {k:4d}  |  loss = {res.loss:12.1f}  |  "
              f"silhouette = {sil:.4f}  |  iter = {res.n_iter:3d}  |  "
              f"time = {wall:.1f} s")

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


# ────────────────────────────────────────────────────────────────
# UTILITIES
# ────────────────────────────────────────────────────────────────

def split_labels_by_block(labels, block_sizes):
    """Split a combined label array back into per-block arrays."""
    out = []
    offset = 0
    for n in block_sizes:
        out.append(labels[offset:offset + n].copy())
        offset += n
    return out


def cluster_statistics(labels, D_sq, medoids, X_tica):
    """Compute per-cluster statistics.

    Returns a list of dicts, each with:
      - cluster_id, population, fraction,
      - medoid_index, medoid_coords (in TICA space),
      - mean_dist_to_medoid, max_dist_to_medoid.
    """
    N = len(labels)
    n_clusters = len(medoids)
    stats = []

    for c in range(n_clusters):
        mask = (labels == c)
        pop  = int(np.sum(mask))
        med  = medoids[c]

        if pop > 0:
            dists = D_sq[med, mask]
            mean_d = float(np.mean(dists))
            max_d  = float(np.max(dists))
        else:
            mean_d = 0.0
            max_d  = 0.0

        stats.append({
            "cluster_id":         c,
            "population":         pop,
            "fraction":           pop / N,
            "medoid_index":       int(med),
            "medoid_tic1":        float(X_tica[med, 0]) if X_tica.shape[1] > 0 else 0.0,
            "medoid_tic2":        float(X_tica[med, 1]) if X_tica.shape[1] > 1 else 0.0,
            "mean_dist_to_medoid": mean_d,
            "max_dist_to_medoid":  max_d,
        })

    return stats


# ────────────────────────────────────────────────────────────────
# PLOTTING
# ────────────────────────────────────────────────────────────────

def _get_plt():
    """Import matplotlib with Agg backend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  [WARN] matplotlib not available — skipping plots")
        return None


def plot_elbow(scan, outdir):
    """Elbow plot: total loss vs k."""
    plt = _get_plt()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scan["k_values"], scan["loss"], "o-",
            color="steelblue", linewidth=2, markersize=6)
    ax.set_xlabel("Number of clusters (k)", fontsize=12)
    ax.set_ylabel("Total dissimilarity (loss)", fontsize=12)
    ax.set_title("k-Medoids Elbow Plot — Mutant Nav1.5", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(outdir, "kmedoids_elbow.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved elbow plot → {fpath}")


def plot_silhouette_vs_k(scan, outdir):
    """Silhouette score vs k."""
    plt = _get_plt()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scan["k_values"], scan["silhouette"], "s-",
            color="darkorange", linewidth=2, markersize=6)
    ax.set_xlabel("Number of clusters (k)", fontsize=12)
    ax.set_ylabel("Silhouette score", fontsize=12)
    ax.set_title("Silhouette Score vs k — Mutant Nav1.5", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(outdir, "kmedoids_silhouette_vs_k.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved silhouette plot → {fpath}")


def plot_cluster_map(X_tica, labels, medoid_idx, outdir, filename="kmedoids_tic1_vs_tic2.png"):
    """Scatter of TIC-1 vs TIC-2 coloured by cluster assignment."""
    plt = _get_plt()
    if plt is None:
        return
    from matplotlib.colors import ListedColormap

    n_clusters = len(np.unique(labels))
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a colourmap with enough distinct colours
    cmap = plt.cm.get_cmap("tab20", min(n_clusters, 20))

    sc = ax.scatter(X_tica[:, 0], X_tica[:, 1],
                    c=labels, cmap=cmap, s=3, alpha=0.5,
                    rasterized=True)

    # Mark medoids
    ax.scatter(X_tica[medoid_idx, 0], X_tica[medoid_idx, 1],
               c="red", marker="x", s=30, linewidths=1.0,
               label="medoids", zorder=5, alpha=0.7)

    ax.set_xlabel("TIC-1", fontsize=12)
    ax.set_ylabel("TIC-2", fontsize=12)
    ax.set_title(f"k-Medoids Clustering (k={n_clusters}) — Mutant Nav1.5", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fpath = os.path.join(outdir, filename)
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved cluster map → {fpath}")


def plot_population_histogram(labels, outdir):
    """Histogram of cluster populations."""
    plt = _get_plt()
    if plt is None:
        return

    pops = np.bincount(labels)
    n_clusters = len(pops)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Sorted bar chart
    ax = axes[0]
    sorted_pops = np.sort(pops)[::-1]
    ax.bar(range(n_clusters), sorted_pops, color="steelblue", width=1.0,
           edgecolor="none")
    ax.set_xlabel("Cluster rank", fontsize=12)
    ax.set_ylabel("Population (frames)", fontsize=12)
    ax.set_title(f"Sorted Cluster Populations (k={n_clusters})", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    # (b) Population distribution
    ax = axes[1]
    ax.hist(pops, bins=50, color="darkorange", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(pops), color="red", linestyle="--", linewidth=1.5,
               label=f"mean = {np.mean(pops):.1f}")
    ax.axvline(np.median(pops), color="blue", linestyle="--", linewidth=1.5,
               label=f"median = {np.median(pops):.1f}")
    ax.set_xlabel("Population (frames)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Population Distribution", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(outdir, "kmedoids_population_histogram.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved population histogram → {fpath}")


def plot_cluster_map_3tics(X_tica, labels, medoid_idx, outdir):
    """TIC-1/2, TIC-1/3, TIC-2/3 coloured by cluster."""
    plt = _get_plt()
    if plt is None:
        return
    if X_tica.shape[1] < 3:
        return

    n_clusters = len(np.unique(labels))
    cmap = plt.cm.get_cmap("tab20", min(n_clusters, 20))

    pairs = [(0, 1, "TIC-1", "TIC-2"),
             (0, 2, "TIC-1", "TIC-3"),
             (1, 2, "TIC-2", "TIC-3")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (i, j, xl, yl) in zip(axes, pairs):
        ax.scatter(X_tica[:, i], X_tica[:, j],
                   c=labels, cmap=cmap, s=2, alpha=0.4, rasterized=True)
        ax.scatter(X_tica[medoid_idx, i], X_tica[medoid_idx, j],
                   c="red", marker="x", s=20, linewidths=0.8, alpha=0.6)
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(yl, fontsize=11)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"k-Medoids (k={n_clusters}) — Top-3 TIC Projections", fontsize=14)
    plt.tight_layout()
    fpath = os.path.join(outdir, "kmedoids_tic_pairs.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved TIC-pair plots → {fpath}")


def plot_block_timeseries(block_labels, block_names, dt_ps, outdir):
    """Time-series of cluster assignments per block."""
    plt = _get_plt()
    if plt is None:
        return

    n_blocks = len(block_labels)
    fig, axes = plt.subplots(n_blocks, 1, figsize=(14, 3 * n_blocks), sharex=False)
    if n_blocks == 1:
        axes = [axes]

    for ax, labels, bn in zip(axes, block_labels, block_names):
        t_ns = np.arange(len(labels)) * dt_ps / 1000.0
        ax.scatter(t_ns, labels, s=0.5, alpha=0.4, c="steelblue", rasterized=True)
        ax.set_ylabel("Cluster ID", fontsize=10)
        ax.set_title(bn, fontsize=11)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (ns)", fontsize=12)
    fig.suptitle("Microstate Time Series — Mutant Nav1.5", fontsize=14, y=1.01)
    plt.tight_layout()
    fpath = os.path.join(outdir, "kmedoids_block_timeseries.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved block time-series → {fpath}")


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 3 — k-Medoids microstate clustering (FasterPAM)")
    parser.add_argument("--k",        type=int,   default=DEFAULT_K,
                        help="Number of clusters (default: %(default)s)")
    parser.add_argument("--ntics",    type=int,   default=DEFAULT_N_TICS,
                        help="Number of TICs to use (default: all)")
    parser.add_argument("--seed",     type=int,   default=DEFAULT_SEED,
                        help="Random seed (default: %(default)s)")
    parser.add_argument("--max-iter", type=int,   default=DEFAULT_MAX_ITER,
                        help="Max PAM iterations (default: %(default)s)")
    parser.add_argument("--scan",     action="store_true",
                        help="Run k-scan before final fit")
    parser.add_argument("--scan-only", action="store_true",
                        help="Run k-scan only, no final fit")
    parser.add_argument("--scan-ks",  type=str,   default=None,
                        help="Comma-separated k values for scan "
                             "(default: 25,50,75,100,150,200,250,300,400,500)")
    args = parser.parse_args()

    k          = args.k
    n_tics     = args.ntics
    seed       = args.seed
    max_iter   = args.max_iter
    do_scan    = args.scan or args.scan_only
    scan_only  = args.scan_only

    if args.scan_ks is not None:
        k_scan_values = [int(x.strip()) for x in args.scan_ks.split(",")]
    else:
        k_scan_values = K_SCAN_DEFAULT

    OUT = os.path.join(BASE, "clustering")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 65)
    print("  Step 3 — k-Medoids Clustering  |  Mutant Nav1.5  (Martini 3 CG)")
    print("=" * 65)
    print(f"  Final k          : {k}")
    tic_str = str(n_tics) if n_tics is not None else "all"
    print(f"  TICs used        : {tic_str}")
    print(f"  Random seed      : {seed}")
    print(f"  Max iterations   : {max_iter}")
    print(f"  k-scan           : {'yes' if do_scan else 'no'}")
    if do_scan:
        print(f"  k-scan values    : {k_scan_values}")
    print(f"  Output dir       : {OUT}")

    # ── 1. Load TICA projections ──
    print("\n[1] Loading per-block TICA projections …")
    blocks = load_tica_blocks(TICA_DIR, BLOCKS, n_tics=n_tics)
    block_sizes = [X.shape[0] for X in blocks]
    X_all = np.vstack(blocks)
    N, d = X_all.shape
    print(f"  Combined: {N} frames × {d} TICs")

    # ── 2. Compute pairwise distance matrix ──
    print("\n[2] Computing pairwise Euclidean distance matrix …")
    D_sq = compute_distance_matrix(X_all)
    print(f"  Saved square distance matrix")

    # ── 3. k-scan (optional) ──
    if do_scan:
        print(f"\n[3] Running k-scan over {len(k_scan_values)} values …")
        scan = k_scan(D_sq, k_scan_values, max_iter=max_iter, seed=seed)

        np.savez(os.path.join(OUT, "k_scan_results.npz"),
                 k_values=scan["k_values"],
                 loss=scan["loss"],
                 silhouette=scan["silhouette"],
                 n_iter=scan["n_iter"],
                 wall_time=scan["wall_time"])

        plot_elbow(scan, OUT)
        plot_silhouette_vs_k(scan, OUT)

        # Print recommendation
        best_sil_idx = np.argmax(scan["silhouette"])
        print(f"\n  Best silhouette: k = {int(scan['k_values'][best_sil_idx])}  "
              f"(score = {scan['silhouette'][best_sil_idx]:.4f})")

        if scan_only:
            print("\n  Scan-only mode — skipping final fit.")
            print("  Step 3 (k-scan) complete.")
            return
    else:
        print("\n[3] k-scan skipped (use --scan to enable)")

    # ── 4. Final clustering at chosen k ──
    print(f"\n[4] Running FasterPAM with k = {k} …")
    t0 = time.time()
    result = run_kmedoids(D_sq, k, max_iter=max_iter, seed=seed)
    wall = time.time() - t0

    labels    = np.asarray(result.labels, dtype=np.int32)
    medoids   = np.asarray(result.medoids, dtype=np.int64)
    loss      = result.loss
    n_iter    = result.n_iter

    print(f"  Converged in {n_iter} iterations  ({wall:.1f} s)")
    print(f"  Total dissimilarity (loss) = {loss:.2f}")
    print(f"  Medoids: {len(medoids)} cluster centres")

    # Silhouette
    print(f"  Computing silhouette score …", end="")
    sil = silhouette_from_square(D_sq, labels, sample_size=5000, seed=seed)
    print(f"  {sil:.4f}")

    # ── 5. Per-block labels ──
    print(f"\n[5] Splitting labels into per-block arrays …")
    block_labels = split_labels_by_block(labels, block_sizes)
    for bn, bl in zip(BLOCKS, block_labels):
        fpath = os.path.join(OUT, f"{bn}_labels.npy")
        np.save(fpath, bl)
        n_unique = len(np.unique(bl))
        print(f"  {bn:12s}  →  {fpath}  "
              f"(N={len(bl)}, {n_unique} unique states)")

    # Combined labels
    np.save(os.path.join(OUT, "labels_all_blocks.npy"), labels)
    print(f"  All blocks   →  labels_all_blocks.npy  shape {labels.shape}")

    # Medoid information
    np.save(os.path.join(OUT, "medoid_indices.npy"), medoids)
    medoid_tica_coords = X_all[medoids]
    np.save(os.path.join(OUT, "medoid_tica_coords.npy"), medoid_tica_coords.astype(np.float32))
    print(f"  Medoid indices  →  medoid_indices.npy   shape {medoids.shape}")
    print(f"  Medoid coords   →  medoid_tica_coords.npy   shape {medoid_tica_coords.shape}")

    # ── 6. Cluster statistics ──
    print(f"\n[6] Computing cluster statistics …")
    pops = np.bincount(labels, minlength=k)
    nonempty = np.sum(pops > 0)
    empty    = k - nonempty

    print(f"  Non-empty clusters  : {nonempty}")
    print(f"  Empty clusters      : {empty}")
    print(f"  Min population      : {pops[pops > 0].min()}")
    print(f"  Max population      : {pops.max()}")
    print(f"  Mean population     : {pops[pops > 0].mean():.1f}")
    print(f"  Median population   : {np.median(pops[pops > 0]):.1f}")

    # ── 7. Diagnostic plots ──
    print(f"\n[7] Generating diagnostic plots …")
    plot_cluster_map(X_all, labels, medoids, OUT)
    plot_population_histogram(labels, OUT)
    plot_cluster_map_3tics(X_all, labels, medoids, OUT)
    plot_block_timeseries(block_labels, BLOCKS, DT_PS, OUT)

    # ── 8. Summary file ──
    print(f"\n[8] Writing summary …")
    summary_path = os.path.join(OUT, "clustering_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Step 3 — k-Medoids Clustering Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Algorithm          : FasterPAM (kmedoids package)\n")
        f.write(f"Number of clusters : {k}\n")
        f.write(f"TICs used          : {d} (of {n_tics if n_tics else 'all available'})\n")
        f.write(f"Total frames       : {N}\n")
        f.write(f"Distance metric    : Euclidean\n")
        f.write(f"Random seed        : {seed}\n")
        f.write(f"Max iterations     : {max_iter}\n")
        f.write(f"Converged in       : {n_iter} iterations ({wall:.1f} s)\n")
        f.write(f"Total dissimilarity: {loss:.4f}\n")
        f.write(f"Silhouette score   : {sil:.4f}\n\n")

        f.write(f"Block composition:\n")
        for bn, bl in zip(BLOCKS, block_labels):
            n_unique = len(np.unique(bl))
            f.write(f"  {bn}: {len(bl)} frames, {n_unique} unique states\n")

        f.write(f"\nCluster population statistics:\n")
        f.write(f"  Non-empty clusters  : {nonempty}\n")
        f.write(f"  Empty clusters      : {empty}\n")
        f.write(f"  Min population      : {pops[pops > 0].min()}\n")
        f.write(f"  Max population      : {pops.max()}\n")
        f.write(f"  Mean population     : {pops[pops > 0].mean():.1f}\n")
        f.write(f"  Median population   : {np.median(pops[pops > 0]):.1f}\n")
        f.write(f"  Std population      : {pops[pops > 0].std():.1f}\n")

        # Top-10 most populated
        top10 = np.argsort(pops)[::-1][:10]
        f.write(f"\nTop-10 most populated clusters:\n")
        f.write(f"  {'Cluster':>8s}  {'Pop':>6s}  {'Frac':>8s}  "
                f"{'Medoid':>8s}  {'TIC-1':>8s}  {'TIC-2':>8s}\n")
        for c in top10:
            f.write(f"  {c:8d}  {pops[c]:6d}  {pops[c]/N:8.4f}  "
                    f"{medoids[c]:8d}  {X_all[medoids[c], 0]:8.3f}  "
                    f"{X_all[medoids[c], 1]:8.3f}\n")

        f.write(f"\nOutputs in {OUT}/:\n")
        f.write(f"  labels_all_blocks.npy       — combined cluster labels ({N},)\n")
        for bn in BLOCKS:
            f.write(f"  {bn}_labels.npy        — per-block labels\n")
        f.write(f"  medoid_indices.npy          — medoid frame indices ({k},)\n")
        f.write(f"  medoid_tica_coords.npy      — medoid TICA coordinates ({k}, {d})\n")
        f.write(f"  (distance matrix computed in memory, not saved to disk)\n")
        if do_scan:
            f.write(f"  k_scan_results.npz          — k-scan data\n")

        f.write(f"\nPlots:\n")
        if do_scan:
            f.write(f"  kmedoids_elbow.png\n")
            f.write(f"  kmedoids_silhouette_vs_k.png\n")
        f.write(f"  kmedoids_tic1_vs_tic2.png\n")
        f.write(f"  kmedoids_population_histogram.png\n")
        f.write(f"  kmedoids_tic_pairs.png\n")
        f.write(f"  kmedoids_block_timeseries.png\n")

    print(f"  Summary → {summary_path}")
    print(f"\n  Step 3 (k-medoids clustering) complete.")


if __name__ == "__main__":
    main()
