#!/usr/bin/env python3
"""
Fast PCA replot from saved projections — Blues/Reds time-gradient design.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import os

OUT_DIR = "/PLOTS_analsysis_WT_mutant/plots_no_id"
data = np.load(os.path.join(OUT_DIR, 'pca_projections.npz'))
proj_wt = data['proj_wt']
proj_mt = data['proj_mt']
time_wt = data['time_wt']
time_mt = data['time_mt']
evr = data['explained_variance_ratio']
n_atoms = int(data['n_atoms'])

# Combine for axis limits
proj_all = np.vstack([proj_wt, proj_mt])

plt.rcParams.update({
    'font.size': 12, 'axes.linewidth': 1.2,
    'lines.linewidth': 1.5, 'legend.fontsize': 11, 'axes.grid': True,
    'grid.alpha': 0.3, 'figure.dpi': 150,
})

WT_CMAP = 'Blues'
MT_CMAP = 'Reds'

def legend_handles():
    """Create custom legend handles for time-gradient colormaps."""
    return [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1565C0', markersize=10, label='WT'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#C62828', markersize=10, label='Mutant'),
    ]

print("Regenerating PCA plots with Blues/Reds time-gradient design...\n")

# --- Plot 16: PC1 vs PC2 scatter ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(proj_wt[:, 0], proj_wt[:, 1], c=time_wt, cmap=WT_CMAP, alpha=0.5, s=15, edgecolors='none')
sc = ax.scatter(proj_mt[:, 0], proj_mt[:, 1], c=time_mt, cmap=MT_CMAP, alpha=0.5, s=15, edgecolors='none')
ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}%)')
ax.set_title('PCA: WT vs Mutant (Cα atoms, shared eigenspace)')
ax.legend(handles=legend_handles(), markerscale=1.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '16_pca_pc1_vs_pc2.png'))
plt.close()
print("  [OK] 16 - PC1 vs PC2 scatter")

# --- Plot 17: PC1 vs PC2 with time coloring (side-by-side) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, proj, time_arr, label, cmap in [
    (axes[0], proj_wt, time_wt, 'WT', WT_CMAP),
    (axes[1], proj_mt, time_mt, 'Mutant', MT_CMAP),
]:
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=time_arr, cmap=cmap, s=15, alpha=0.6, edgecolors='none')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Time (ns)')
    ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}%)')
    ax.set_title(label)
    ax.set_xlim(proj_all[:, 0].min()-0.5, proj_all[:, 0].max()+0.5)
    ax.set_ylim(proj_all[:, 1].min()-0.5, proj_all[:, 1].max()+0.5)
plt.suptitle('PCA Time Evolution (Cα, shared eigenspace)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '17_pca_time_evolution.png'))
plt.close()
print("  [OK] 17 - PC1 vs PC2 time evolution")

# --- Plot 18: PC projections along time ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for i, ax in enumerate(axes):
    ax.scatter(time_mt, proj_mt[:, i], c=time_mt, cmap=MT_CMAP, alpha=0.5, s=4, edgecolors='none')
    ax.scatter(time_wt, proj_wt[:, i], c=time_wt, cmap=WT_CMAP, alpha=0.5, s=4, edgecolors='none')
    ax.set_ylabel(f'PC{i+1} ({evr[i]*100:.1f}%)')
    ax.legend(handles=legend_handles(), loc='upper right')
axes[-1].set_xlabel('Time (ns)')
axes[0].set_title('Principal Component Projections vs Time')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '18_pca_projections_vs_time.png'))
plt.close()
print("  [OK] 18 - PC projections vs time")

# --- Plot 19: Scree plot ---
n_components = len(evr)
fig, ax = plt.subplots(figsize=(8, 5))
pcs = np.arange(1, n_components+1)
ax.bar(pcs, evr * 100, color='#1565C0', alpha=0.8, edgecolor='#0D47A1')
ax.plot(pcs, np.cumsum(evr) * 100, 'o-', color='#C62828', linewidth=2, markersize=7, label='Cumulative')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance (%)')
ax.set_title('PCA Scree Plot (Cα atoms)')
ax.set_xticks(pcs)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '19_pca_scree_plot.png'))
plt.close()
print("  [OK] 19 - Scree plot")

# --- Plot 20: PC1 vs PC3 scatter ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(proj_wt[:, 0], proj_wt[:, 2], c=time_wt, cmap=WT_CMAP, alpha=0.5, s=15, edgecolors='none')
ax.scatter(proj_mt[:, 0], proj_mt[:, 2], c=time_mt, cmap=MT_CMAP, alpha=0.5, s=15, edgecolors='none')
ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)')
ax.set_ylabel(f'PC3 ({evr[2]*100:.1f}%)')
ax.set_title('PCA: WT vs Mutant (PC1 vs PC3)')
ax.legend(handles=legend_handles(), markerscale=1.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '20_pca_pc1_vs_pc3.png'))
plt.close()
print("  [OK] 20 - PC1 vs PC3 scatter")

# --- Plot 21: PC1 distribution ---
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(proj_all[:, 0].min(), proj_all[:, 0].max(), 60)
ax.hist(proj_wt[:, 0], bins=bins, color='#1565C0', alpha=0.6, density=True, label='WT')
ax.hist(proj_mt[:, 0], bins=bins, color='#C62828', alpha=0.6, density=True, label='Mutant')
ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)')
ax.set_ylabel('Density')
ax.set_title('PC1 Distribution: WT vs Mutant')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '21_pca_pc1_distribution.png'))
plt.close()
print("  [OK] 21 - PC1 distribution")

# --- Plot 22: 3D PCA scatter ---
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(proj_wt[:, 0], proj_wt[:, 1], proj_wt[:, 2],
           c=time_wt, cmap=WT_CMAP, alpha=0.4, s=12, edgecolors='none', depthshade=False)
ax.scatter(proj_mt[:, 0], proj_mt[:, 1], proj_mt[:, 2],
           c=time_mt, cmap=MT_CMAP, alpha=0.4, s=12, edgecolors='none', depthshade=False)
ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)', labelpad=10)
ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}%)', labelpad=10)
ax.set_zlabel(f'PC3 ({evr[2]*100:.1f}%)', labelpad=10)
ax.set_title('3D PCA: WT vs Mutant (Cα atoms)', pad=15)
ax.legend(handles=legend_handles(), markerscale=1.5, loc='upper left')
ax.view_init(elev=25, azim=135)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '22_pca_3d_pc1_pc2_pc3.png'))
plt.close()
print("  [OK] 22 - 3D PCA scatter")

# --- Plot 23: 3D PCA time evolution (side-by-side) ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8), subplot_kw={'projection': '3d'})
for ax, proj, time_arr, label, cmap in [
    (axes[0], proj_wt, time_wt, 'WT', WT_CMAP),
    (axes[1], proj_mt, time_mt, 'Mutant', MT_CMAP),
]:
    sc = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                    c=time_arr, cmap=cmap, s=12, alpha=0.5, edgecolors='none', depthshade=False)
    cb = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label('Time (ns)')
    ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)', labelpad=8)
    ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}%)', labelpad=8)
    ax.set_zlabel(f'PC3 ({evr[2]*100:.1f}%)', labelpad=8)
    ax.set_title(label)
    ax.view_init(elev=25, azim=135)
    ax.set_xlim(proj_all[:, 0].min()-0.5, proj_all[:, 0].max()+0.5)
    ax.set_ylim(proj_all[:, 1].min()-0.5, proj_all[:, 1].max()+0.5)
    ax.set_zlim(proj_all[:, 2].min()-0.5, proj_all[:, 2].max()+0.5)
plt.suptitle('3D PCA Time Evolution (Cα, shared eigenspace)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '23_pca_3d_time_evolution.png'))
plt.close()
print("  [OK] 23 - 3D PCA time evolution")

print("\nAll PCA plots updated.")
