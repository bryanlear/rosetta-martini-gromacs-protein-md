#!/usr/bin/env python
"""
Step 1 — Local Feature Projection & Discretization
=====================================================
MSM pipeline for Mutant Nav1.5 (Martini 3 CG, HREMC replica-00 trajectory).

Extracts raw Cartesian coordinates x(t) from three trajectory blocks and
transforms them into a set of internal collective variables:

  (A) BB pairwise distances        — inter-domain BB–BB pairs (stride-subsampled)
  (B) Native-contact fraction Q(t) — per-domain and cross-domain
  (C) Backbone pseudo-dihedrals    — consecutive BB quadruplets (φ-like)
  (D) Radius of gyration           — total + per-domain
  (E) Inter-domain centroid dists   — 6 unique DI↔DII … DIII↔DIV pairs

All features are saved as NumPy arrays (one .npy per block) and a combined
feature matrix is written for downstream TICA / MSM discretization.

Work directory: independent_local_matrices/
"""

import os
import sys
import time
import warnings
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PKG  = os.path.dirname(BASE)            # mutant_analysis_pkg
TOP  = os.path.join(PKG, "reference", "conf_replica_00.gro")
TRAJ_DIR = BASE                          # blocks are in independent_local_matrices/
OUT  = BASE                              # independent_local_matrices/

BLOCKS = ["msm_block1.xtc", "msm_block2.xtc", "msm_block3.xtc"]

# Nav1.5 CG domain boundaries (Martini residue id)
DOMAINS = {
    "DI":   (1, 450),
    "DII":  (451, 900),
    "DIII": (901, 1350),
    "DIV":  (1351, 2016),
}

# Native-contact cut-off (nm → Å for MDA; Martini BB ≈ 0.8 nm contact)
NATIVE_CUTOFF_A = 8.0      # 0.8 nm in Angstrom

# For pairwise distance feature: subsample every N-th BB bead
BB_STRIDE = 10              # keeps ~ 200 beads → ~20 k distance pairs

# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────
def dihedral_angle(p1, p2, p3, p4):
    """Vectorised dihedral angle (radians) for arrays of shape (N, 3)."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    # normalise
    n1_norm = np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-12
    n2_norm = np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-12
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12))
    x = np.sum(n1 * n2, axis=-1)
    y = np.sum(m1 * n2, axis=-1)
    return np.arctan2(y, x)


def compute_native_contacts(ref_positions, domain_indices, cutoff=NATIVE_CUTOFF_A):
    """Return list of (i, j) pairs that are within *cutoff* in the reference."""
    contacts = []
    n = len(ref_positions)
    dist_mat = squareform(pdist(ref_positions))
    for i in range(n):
        for j in range(i + 4, n):          # skip i, i±1, i±2, i±3 (bonded)
            if dist_mat[i, j] < cutoff:
                contacts.append((i, j))
    return contacts


def fraction_native_contacts(positions, contact_pairs, cutoff):
    """Fraction Q of native contacts formed at a given frame."""
    if len(contact_pairs) == 0:
        return 0.0
    dists = np.linalg.norm(
        positions[np.array([p[0] for p in contact_pairs])]
        - positions[np.array([p[1] for p in contact_pairs])],
        axis=1,
    )
    return np.mean(dists < cutoff * 1.2)     # 20 % tolerance


def radius_of_gyration(positions, masses=None):
    """Rg from positions (equal mass if masses=None)."""
    if masses is None:
        masses = np.ones(len(positions))
    com = np.average(positions, axis=0, weights=masses)
    dr = positions - com
    return np.sqrt(np.sum(masses * np.sum(dr**2, axis=1)) / masses.sum())


# ────────────────────────────────────────────────────────────
# MAIN FEATURE EXTRACTION
# ────────────────────────────────────────────────────────────
def extract_features(block_xtc, topology=TOP):
    """
    Extract all internal CVs from one trajectory block.

    Returns
    -------
    features : dict   with keys  'pairwise_dist', 'native_Q', 'dihedrals',
                                  'rg', 'interdomain_dist'
    Each value is np.ndarray  shape  (n_frames, n_features_of_that_type)
    """
    print(f"\n{'='*60}")
    print(f"  Processing: {os.path.basename(block_xtc)}")
    print(f"{'='*60}")
    t0 = time.time()

    u = mda.Universe(topology, block_xtc)
    n_frames = len(u.trajectory)
    print(f"  Frames: {n_frames}   dt = {u.trajectory.dt} ps")

    # ── Atom groups ──
    bb_all   = u.select_atoms("name BB")                     # 2016 beads
    bb_sub   = bb_all[::BB_STRIDE]                            # subsampled for pw-dist
    n_sub    = bb_sub.n_atoms
    n_pairs  = n_sub * (n_sub - 1) // 2

    # Domain atom groups (BB only)
    dom_bb = {}
    for dname, (r0, r1) in DOMAINS.items():
        dom_bb[dname] = u.select_atoms(f"name BB and resid {r0}:{r1}")

    print(f"  BB beads: {bb_all.n_atoms}  (subsampled {n_sub} → {n_pairs} pw-dist pairs)")
    for dname, ag in dom_bb.items():
        print(f"    {dname}: {ag.n_atoms} BB beads")

    # ── Reference structure (first frame) for native contacts ──
    u.trajectory[0]
    ref_bb_pos = bb_all.positions.copy()

    # Compute native-contact pairs from the reference frame
    # We compute contacts among all BB beads
    print("  Computing native-contact reference map …")
    native_pairs = compute_native_contacts(ref_bb_pos, None, NATIVE_CUTOFF_A)
    print(f"  Native contacts (all): {len(native_pairs)}")

    # Also per-domain and cross-domain contact lists
    dom_resid_mask = {}
    for dname, (r0, r1) in DOMAINS.items():
        dom_resid_mask[dname] = np.where(
            (bb_all.resids >= r0) & (bb_all.resids <= r1)
        )[0]

    # Cross-domain contact pairs: break native_pairs into categories
    dom_names = list(DOMAINS.keys())
    cross_contacts = {}    # e.g. ('DI','DII') → [(i,j), …]
    intra_contacts = {}    # e.g. 'DI' → [(i,j), …]
    for dname in dom_names:
        intra_contacts[dname] = []
    for dn1 in range(len(dom_names)):
        for dn2 in range(dn1 + 1, len(dom_names)):
            cross_contacts[(dom_names[dn1], dom_names[dn2])] = []

    set_masks = {dn: set(dom_resid_mask[dn].tolist()) for dn in dom_names}
    for (i, j) in native_pairs:
        assigned = False
        for dn in dom_names:
            if i in set_masks[dn] and j in set_masks[dn]:
                intra_contacts[dn].append((i, j))
                assigned = True
                break
        if not assigned:
            for dn1 in range(len(dom_names)):
                for dn2 in range(dn1 + 1, len(dom_names)):
                    if ((i in set_masks[dom_names[dn1]] and j in set_masks[dom_names[dn2]]) or
                        (j in set_masks[dom_names[dn1]] and i in set_masks[dom_names[dn2]])):
                        cross_contacts[(dom_names[dn1], dom_names[dn2])].append((i, j))
                        break

    n_Q = 1 + len(dom_names) + len(cross_contacts)   # total Q, per-domain Q, cross-domain Q
    print(f"  Q features: {n_Q}  (1 total + {len(dom_names)} intra + {len(cross_contacts)} cross)")

    # ── Pre-allocate output arrays ──
    feat_pw    = np.empty((n_frames, n_pairs), dtype=np.float32)
    feat_Q     = np.empty((n_frames, n_Q), dtype=np.float32)
    n_dihed    = bb_all.n_atoms - 3                               # quadruplets
    feat_dih   = np.empty((n_frames, n_dihed * 2), dtype=np.float32)  # sin, cos
    feat_rg    = np.empty((n_frames, 1 + len(DOMAINS)), dtype=np.float32)   # total + per-domain
    n_idist    = len(dom_names) * (len(dom_names) - 1) // 2
    feat_idist = np.empty((n_frames, n_idist), dtype=np.float32)

    # BB subsampled indices (into bb_all positions)
    sub_idx = np.arange(0, bb_all.n_atoms, BB_STRIDE)

    # ── Iterate frames ──
    report_every = max(1, n_frames // 10)
    for fi, ts in enumerate(u.trajectory):
        if fi % report_every == 0:
            elapsed = time.time() - t0
            print(f"    frame {fi:>6d}/{n_frames}  ({elapsed:.1f} s)")

        bb_pos = bb_all.positions  # (2016, 3)

        # (A) Pairwise distances — subsampled BB beads
        pos_sub = bb_pos[sub_idx]  # (n_sub, 3)
        feat_pw[fi] = pdist(pos_sub)

        # (B) Native-contact fractions
        col = 0
        feat_Q[fi, col] = fraction_native_contacts(bb_pos, native_pairs, NATIVE_CUTOFF_A)
        col += 1
        for dn in dom_names:
            feat_Q[fi, col] = fraction_native_contacts(bb_pos, intra_contacts[dn], NATIVE_CUTOFF_A)
            col += 1
        for key in cross_contacts:
            feat_Q[fi, col] = fraction_native_contacts(bb_pos, cross_contacts[key], NATIVE_CUTOFF_A)
            col += 1

        # (C) Backbone pseudo-dihedrals → sin/cos encoding
        p1 = bb_pos[:-3]
        p2 = bb_pos[1:-2]
        p3 = bb_pos[2:-1]
        p4 = bb_pos[3:]
        phi = dihedral_angle(p1, p2, p3, p4)      # (n_dihed,)
        feat_dih[fi, :n_dihed] = np.sin(phi)
        feat_dih[fi, n_dihed:] = np.cos(phi)

        # (D) Radius of gyration — total + per-domain
        feat_rg[fi, 0] = radius_of_gyration(bb_pos)
        for di, dname in enumerate(dom_names):
            dom_idx = dom_resid_mask[dname]
            feat_rg[fi, 1 + di] = radius_of_gyration(bb_pos[dom_idx])

        # (E) Inter-domain centroid distances
        centroids = {}
        for dname in dom_names:
            centroids[dname] = bb_pos[dom_resid_mask[dname]].mean(axis=0)
        col = 0
        for dn1 in range(len(dom_names)):
            for dn2 in range(dn1 + 1, len(dom_names)):
                feat_idist[fi, col] = np.linalg.norm(
                    centroids[dom_names[dn1]] - centroids[dom_names[dn2]]
                )
                col += 1

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f} s")

    features = {
        "pairwise_dist":   feat_pw,
        "native_Q":        feat_Q,
        "dihedrals":       feat_dih,
        "rg":              feat_rg,
        "interdomain_dist": feat_idist,
    }
    return features


def save_features(features, block_name, outdir):
    """Save each feature type as .npy and the combined matrix."""
    tag = os.path.splitext(block_name)[0]      # e.g. 'msm_block1'
    block_dir = os.path.join(outdir, tag)
    os.makedirs(block_dir, exist_ok=True)

    combined_list = []
    for key, arr in features.items():
        fpath = os.path.join(block_dir, f"{key}.npy")
        np.save(fpath, arr)
        print(f"    saved {key:20s}  shape {str(arr.shape):>20s}  →  {fpath}")
        combined_list.append(arr)

    combined = np.hstack(combined_list)
    cpath = os.path.join(block_dir, "features_combined.npy")
    np.save(cpath, combined)
    print(f"    combined feature matrix  shape {str(combined.shape):>20s}  →  {cpath}")
    return combined


# ────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)

    all_combined = []
    meta = {}

    for block in BLOCKS:
        xtc = os.path.join(TRAJ_DIR, block)
        if not os.path.exists(xtc):
            print(f"WARNING: {xtc} not found — skipping")
            continue

        feats = extract_features(xtc)
        combined = save_features(feats, block, OUT)
        all_combined.append(combined)

        tag = os.path.splitext(block)[0]
        meta[tag] = {k: v.shape for k, v in feats.items()}

    # ── Save concatenated matrix (all blocks) ──
    if all_combined:
        full = np.vstack(all_combined)
        full_path = os.path.join(OUT, "features_all_blocks.npy")
        np.save(full_path, full)
        print(f"\n  All-blocks matrix  shape {full.shape}  →  {full_path}")

    # ── Write summary ──
    summary_path = os.path.join(OUT, "feature_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Step 1 — Feature Extraction Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Topology:  {TOP}\n")
        f.write(f"Blocks:    {BLOCKS}\n")
        f.write(f"BB stride: {BB_STRIDE}\n")
        f.write(f"Native contact cutoff: {NATIVE_CUTOFF_A} Å\n\n")
        f.write("Domain boundaries:\n")
        for dname, (r0, r1) in DOMAINS.items():
            f.write(f"  {dname}: resid {r0}–{r1}\n")
        f.write("\n")
        for tag, shapes in meta.items():
            f.write(f"\n{tag}:\n")
            for key, shape in shapes.items():
                f.write(f"  {key:25s}  {str(shape)}\n")
        if all_combined:
            f.write(f"\nCombined (all blocks):  {full.shape}\n")
        f.write(f"\nFeature columns breakdown:\n")
        f.write(f"  pairwise_dist     = BB subsampled pw-distances (stride {BB_STRIDE})\n")
        f.write(f"  native_Q          = fraction of native contacts (total, intra-domain, cross-domain)\n")
        f.write(f"  dihedrals         = sin/cos of BB pseudo-dihedral angles\n")
        f.write(f"  rg                = radius of gyration (total + per-domain)\n")
        f.write(f"  interdomain_dist  = centroid-centroid distances between domains\n")
    print(f"\n  Summary written to {summary_path}")
    print("\n  Step 1 complete.")
