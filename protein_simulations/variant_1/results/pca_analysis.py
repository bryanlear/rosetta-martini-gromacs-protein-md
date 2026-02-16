#!/usr/bin/env python3
"""
PCA on concatenated WT + Mutant Cα trajectories in same eigenspace.

Workflow:
  1. Load both trajectories (GRO topology + XTC trajectory)
  2. Select Cα atoms (2016 atoms × 3 coords = 6048 features per frame)
  3. Align each frame to the WT first-frame reference (removes translation/rotation)
  4. Concatenate: 1001 WT + 1001 MT = 2002 frames
  5. PCA on the concatenated coordinate matrix
  6. Project both trajectories and plot
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import MDAnalysis as mda
from MDAnalysis.analysis import align
import os
import time

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = "/backmap_cg2at/backmap_charmm36/FINAL"
MT_GRO = f"{BASE}/production/first_frame.gro"
MT_XTC = f"{BASE}/production/production.xtc"
WT_GRO = f"{BASE}/production_for_WT/prod_wt.gro"  # 743194 atoms, matches WT trajectory
WT_XTC = f"{BASE}/production_for_WT/prod_wt.xtc"
OUT_DIR = "/PLOTS_analsysis_WT_mutant/plots_no_id"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.figsize': (10, 8), 'font.size': 12, 'axes.linewidth': 1.2,
    'lines.linewidth': 1.5, 'legend.fontsize': 11, 'axes.grid': True,
    'grid.alpha': 0.3, 'figure.dpi': 150,
})
WT_COLOR = '#2196F3'
MT_COLOR = '#F44336'

SELECTION = 'name CA'  # Cα atoms

# ── Step 1: Load universes ─────────────────────────────────────────────────
print("Loading trajectories...")
t0 = time.time()

u_wt = mda.Universe(WT_GRO, WT_XTC)
u_mt = mda.Universe(MT_GRO, MT_XTC)

ca_wt = u_wt.select_atoms(SELECTION)
ca_mt = u_mt.select_atoms(SELECTION)

n_atoms = ca_wt.n_atoms
n_features = n_atoms * 3
n_frames_wt = len(u_wt.trajectory)
n_frames_mt = len(u_mt.trajectory)

print(f"  WT: {n_frames_wt} frames, {ca_wt.n_atoms} Cα atoms")
print(f"  MT: {n_frames_mt} frames, {ca_mt.n_atoms} Cα atoms")
print(f"  Features per frame: {n_features} (= {n_atoms} atoms × 3 coords)")
print(f"  Loaded in {time.time()-t0:.1f}s")

# ── Step 2: Build reference for alignment ──────────────────────────────────
# Use WT first frame as reference for WT, MT first frame for MT
# (different total atom counts, but same # of Cα atoms)
ref_wt = mda.Universe(WT_GRO)
ref_mt = mda.Universe(MT_GRO)
ref_positions = ref_wt.select_atoms(SELECTION).positions.copy()

# ── Step 3: Extract aligned Cα coordinates ─────────────────────────────────
print("\nExtracting and aligning Cα coordinates...")
t0 = time.time()

def extract_aligned_coords(universe, ca_selection, ref_universe, sel_str=SELECTION):
    """Align each frame to reference and extract Cα coordinates."""
    coords = np.zeros((len(universe.trajectory), ca_selection.n_atoms * 3))
    # Align the full trajectory to reference
    aligner = align.AlignTraj(universe, ref_universe, select=sel_str, in_memory=False)
    aligner.run()
    # Now extract aligned coordinates
    for i, ts in enumerate(universe.trajectory):
        coords[i] = ca_selection.positions.flatten()
        if (i+1) % 200 == 0:
            print(f"    Frame {i+1}/{len(universe.trajectory)}")
    return coords

coords_wt = extract_aligned_coords(u_wt, ca_wt, ref_wt, SELECTION)
print(f"  WT done: shape {coords_wt.shape}")

coords_mt = extract_aligned_coords(u_mt, ca_mt, ref_mt, SELECTION)
print(f"  MT done: shape {coords_mt.shape}")

# Now align MT to WT reference frame via Cα superposition of mean structures
# so both share the same spatial orientation
from MDAnalysis.analysis.align import rotation_matrix
wt_mean = coords_wt.mean(axis=0).reshape(-1, 3)
mt_mean = coords_mt.mean(axis=0).reshape(-1, 3)
# Center both
wt_center = wt_mean.mean(axis=0)
mt_center = mt_mean.mean(axis=0)
wt_centered = wt_mean - wt_center
mt_centered = mt_mean - mt_center
R, rmsd_val = rotation_matrix(mt_centered, wt_centered)
# Apply rotation to all MT frames
for i in range(len(coords_mt)):
    frame = coords_mt[i].reshape(-1, 3)
    frame -= mt_center
    frame = frame @ R.T
    frame += wt_center
    coords_mt[i] = frame.flatten()
print(f"  MT aligned to WT reference (RMSD of means: {rmsd_val:.3f} nm)")

print(f"  Extracted in {time.time()-t0:.1f}s")

# ── Step 4: Concatenate ────────────────────────────────────────────────────
print("\nConcatenating WT + MT...")
coords_concat = np.vstack([coords_wt, coords_mt])
labels = np.array(['WT'] * n_frames_wt + ['Mutant'] * n_frames_mt)
print(f"  Concatenated matrix: {coords_concat.shape} (samples × features)")

# ── Step 5: PCA ────────────────────────────────────────────────────────────
print("\nRunning PCA...")
t0 = time.time()
n_components = 10
pca = PCA(n_components=n_components)
projections = pca.fit_transform(coords_concat)
print(f"  PCA done in {time.time()-t0:.1f}s")
print(f"  Explained variance ratios:")
for i in range(n_components):
    print(f"    PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}%")
print(f"  Cumulative (PC1-PC{n_components}): {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

# Split projections back
proj_wt = projections[:n_frames_wt]
proj_mt = projections[n_frames_wt:]
time_wt = np.arange(n_frames_wt) * 10 / 1000.0  # ps → ns
time_mt = np.arange(n_frames_mt) * 10 / 1000.0

# ── Step 6: Plots ──────────────────────────────────────────────────────────
print("\nGenerating plots...")

# --- Plot 1: PC1 vs PC2 scatter ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(proj_wt[:, 0], proj_wt[:, 1], c=WT_COLOR, alpha=0.4, s=15, label='WT', edgecolors='none')
ax.scatter(proj_mt[:, 0], proj_mt[:, 1], c=MT_COLOR, alpha=0.4, s=15, label='Mutant', edgecolors='none')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA: WT vs Mutant (Cα atoms, shared eigenspace)')
ax.legend(markerscale=3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '16_pca_pc1_vs_pc2.png'))
plt.close()
print("  [OK] PC1 vs PC2 scatter")

# --- Plot 2: PC1 vs PC2 with time coloring ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, proj, time_arr, label, cmap in [
    (axes[0], proj_wt, time_wt, 'WT', 'Blues'),
    (axes[1], proj_mt, time_mt, 'Mutant', 'Reds'),
]:
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=time_arr, cmap=cmap, s=15, alpha=0.6, edgecolors='none')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Time (ns)')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'{label}')
    # Set same axis limits for both
    ax.set_xlim(projections[:, 0].min()-0.5, projections[:, 0].max()+0.5)
    ax.set_ylim(projections[:, 1].min()-0.5, projections[:, 1].max()+0.5)
plt.suptitle('PCA Time Evolution (Cα, shared eigenspace)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '17_pca_time_evolution.png'))
plt.close()
print("  [OK] PC1 vs PC2 time evolution")

# --- Plot 3: PC projections along time ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(time_mt, proj_mt[:, i], color=MT_COLOR, alpha=0.7, linewidth=0.8, label='Mutant')
    ax.plot(time_wt, proj_wt[:, i], color=WT_COLOR, alpha=0.7, linewidth=0.8, label='WT')
    ax.set_ylabel(f'PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)')
    ax.legend(loc='upper right')
axes[-1].set_xlabel('Time (ns)')
axes[0].set_title('Principal Component Projections vs Time')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '18_pca_projections_vs_time.png'))
plt.close()
print("  [OK] PC projections vs time")

# --- Plot 4: Explained variance (scree plot) ---
fig, ax = plt.subplots(figsize=(8, 5))
pcs = np.arange(1, n_components+1)
ax.bar(pcs, pca.explained_variance_ratio_ * 100, color='#607D8B', alpha=0.8, edgecolor='black')
ax.plot(pcs, np.cumsum(pca.explained_variance_ratio_) * 100, 'o-', color='#FF5722', linewidth=2, label='Cumulative')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance (%)')
ax.set_title('PCA Scree Plot (Cα atoms)')
ax.set_xticks(pcs)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '19_pca_scree_plot.png'))
plt.close()
print("  [OK] Scree plot")

# --- Plot 5: PC1 vs PC3 scatter ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(proj_wt[:, 0], proj_wt[:, 2], c=WT_COLOR, alpha=0.4, s=15, label='WT', edgecolors='none')
ax.scatter(proj_mt[:, 0], proj_mt[:, 2], c=MT_COLOR, alpha=0.4, s=15, label='Mutant', edgecolors='none')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
ax.set_title('PCA: WT vs Mutant (PC1 vs PC3)')
ax.legend(markerscale=3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '20_pca_pc1_vs_pc3.png'))
plt.close()
print("  [OK] PC1 vs PC3 scatter")

# --- Plot 6: Density / histogram along PC1 ---
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(projections[:, 0].min(), projections[:, 0].max(), 60)
ax.hist(proj_wt[:, 0], bins=bins, color=WT_COLOR, alpha=0.6, density=True, label='WT')
ax.hist(proj_mt[:, 0], bins=bins, color=MT_COLOR, alpha=0.6, density=True, label='Mutant')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel('Density')
ax.set_title('PC1 Distribution: WT vs Mutant')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '21_pca_pc1_distribution.png'))
plt.close()
print("  [OK] PC1 distribution")

# --- Plot 7: 3D PCA (PC1 vs PC2 vs PC3) ---
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(proj_wt[:, 0], proj_wt[:, 1], proj_wt[:, 2],
           c=WT_COLOR, alpha=0.35, s=12, label='WT', edgecolors='none', depthshade=True)
ax.scatter(proj_mt[:, 0], proj_mt[:, 1], proj_mt[:, 2],
           c=MT_COLOR, alpha=0.35, s=12, label='Mutant', edgecolors='none', depthshade=True)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', labelpad=10)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', labelpad=10)
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', labelpad=10)
ax.set_title('3D PCA: WT vs Mutant (Cα atoms)', pad=15)
ax.legend(markerscale=3, loc='upper left')
ax.view_init(elev=25, azim=135)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '22_pca_3d_pc1_pc2_pc3.png'))
plt.close()
print("  [OK] 3D PCA (PC1 vs PC2 vs PC3)")

# --- Plot 7b: 3D PCA with time coloring ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8), subplot_kw={'projection': '3d'})
for ax, proj, time_arr, label, cmap in [
    (axes[0], proj_wt, time_wt, 'WT', 'Blues'),
    (axes[1], proj_mt, time_mt, 'Mutant', 'Reds'),
]:
    sc = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                    c=time_arr, cmap=cmap, s=12, alpha=0.5, edgecolors='none', depthshade=False)
    cb = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label('Time (ns)')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', labelpad=8)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', labelpad=8)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', labelpad=8)
    ax.set_title(label)
    ax.view_init(elev=25, azim=135)
    # Same axis limits for both
    ax.set_xlim(projections[:, 0].min()-0.5, projections[:, 0].max()+0.5)
    ax.set_ylim(projections[:, 1].min()-0.5, projections[:, 1].max()+0.5)
    ax.set_zlim(projections[:, 2].min()-0.5, projections[:, 2].max()+0.5)
plt.suptitle('3D PCA Time Evolution (Cα, shared eigenspace)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '23_pca_3d_time_evolution.png'))
plt.close()
print("  [OK] 3D PCA time evolution")

# ── Save projections for fast re-plotting ──────────────────────────────────
npz_path = os.path.join(OUT_DIR, 'pca_projections.npz')
np.savez(npz_path,
         proj_wt=proj_wt, proj_mt=proj_mt,
         time_wt=time_wt, time_mt=time_mt,
         explained_variance_ratio=pca.explained_variance_ratio_,
         n_atoms=n_atoms)
print(f"  [OK] Saved projections to pca_projections.npz")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\nAll PCA plots saved to: {OUT_DIR}")
print(f"\nSummary:")
print(f"  Selection: {SELECTION} ({n_atoms} atoms)")
print(f"  Frames: {n_frames_wt} WT + {n_frames_mt} MT = {n_frames_wt + n_frames_mt} total")
print(f"  Feature space: {n_features} dimensions ({n_atoms} × 3)")
print(f"  Reference: WT first frame")
print(f"  PC1+PC2 explained variance: {(pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1])*100:.1f}%")
