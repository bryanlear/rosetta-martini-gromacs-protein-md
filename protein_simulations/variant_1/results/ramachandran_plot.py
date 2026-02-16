#!/usr/bin/env python3
"""
Ramachandran plot comparison: WT vs Mutant SCN5A production trajectories.
Computes backbone φ/ψ dihedrals and generates density-based Ramachandran plots.
Uses every 10th frame to keep memory manageable on large trajectories.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Ramachandran
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
MT_GRO = "/production.gro"
MT_XTC = "/production.xtc"

WT_GRO = "/prod_wt.gro"
WT_XTC = "/prod_wt.xtc"

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.figsize': (10, 6), 'font.size': 12, 'axes.linewidth': 1.2,
    'figure.dpi': 150,
})
WT_COLOR = '#2196F3'
MT_COLOR = '#F44336'

STRIDE = 10  # analyse every 10th frame (~1000 ps intervals for 2 fs dt, 200 ps write)


def compute_dihedrals(gro, xtc, label, stride=STRIDE):
    """Load trajectory and compute φ/ψ for protein backbone."""
    print(f"  Loading {label} universe: {gro}")
    u = mda.Universe(gro, xtc)
    protein = u.select_atoms("protein")
    print(f"    Protein atoms: {protein.n_atoms}, "
          f"Frames: {u.trajectory.n_frames}, Stride: {stride}")

    rama = Ramachandran(protein).run(step=stride, verbose=True)
    # rama.results.angles shape: (n_frames, n_residues, 2) → (phi, psi)
    angles = rama.results.angles
    phi = angles[:, :, 0].flatten()
    psi = angles[:, :, 1].flatten()
    print(f"    Total φ/ψ pairs: {len(phi):,}")
    return phi, psi


def plot_ramachandran_single(phi, psi, label, color, filename):
    """2D histogram Ramachandran for a single system."""
    fig, ax = plt.subplots(figsize=(7, 7))
    h = ax.hist2d(phi, psi, bins=180, range=[[-180, 180], [-180, 180]],
                  cmap='inferno', norm=LogNorm(), density=True)
    plt.colorbar(h[3], ax=ax, label='Log density')

    # secondary-structure reference regions
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.4)
    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.4)

    ax.set_xlabel(r'$\phi$ (degrees)')
    ax.set_ylabel(r'$\psi$ (degrees)')
    ax.set_title(f'Ramachandran Plot — {label}')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect('equal')
    fig.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_ramachandran_comparison(phi_wt, psi_wt, phi_mt, psi_mt):
    """Side-by-side Ramachandran comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, phi, psi, label, cmap in [
        (axes[0], phi_wt, psi_wt, 'Wild-Type', 'Blues'),
        (axes[1], phi_mt, psi_mt, 'Mutant',    'Reds'),
    ]:
        h = ax.hist2d(phi, psi, bins=180, range=[[-180, 180], [-180, 180]],
                      cmap=cmap, norm=LogNorm(), density=True)
        plt.colorbar(h[3], ax=ax, label='Log density', shrink=0.8)
        ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.4)
        ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.4)
        ax.set_xlabel(r'$\phi$ (degrees)')
        ax.set_ylabel(r'$\psi$ (degrees)')
        ax.set_title(f'{label}')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_aspect('equal')

    fig.suptitle('Ramachandran Plot — WT vs. Mutant', fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, '04_ramachandran_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_ramachandran_difference(phi_wt, psi_wt, phi_mt, psi_mt):
    """Difference map: Mutant density − WT density (highlights shifted populations)."""
    bins = 180
    r = [[-180, 180], [-180, 180]]

    H_wt, xedges, yedges = np.histogram2d(phi_wt, psi_wt, bins=bins, range=r, density=True)
    H_mt, _, _ = np.histogram2d(phi_mt, psi_mt, bins=bins, range=r, density=True)

    diff = H_mt - H_wt
    vmax = np.percentile(np.abs(diff), 99)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(diff.T, origin='lower', extent=[-180, 180, -180, 180],
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    plt.colorbar(im, ax=ax, label=r'$\Delta$ density (Mutant − WT)')
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.4)
    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.4)
    ax.set_xlabel(r'$\phi$ (degrees)')
    ax.set_ylabel(r'$\psi$ (degrees)')
    ax.set_title('Ramachandran Difference Map (Mutant − WT)')
    fig.tight_layout()
    out = os.path.join(OUT_DIR, '04b_ramachandran_difference.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    print("Computing Ramachandran plots …")

    print("\n[1/2] Wild-Type")
    phi_wt, psi_wt = compute_dihedrals(WT_GRO, WT_XTC, "WT")

    print("\n[2/2] Mutant")
    phi_mt, psi_mt = compute_dihedrals(MT_GRO, MT_XTC, "Mutant")

    print("\nGenerating plots …")
    plot_ramachandran_comparison(phi_wt, psi_wt, phi_mt, psi_mt)
    plot_ramachandran_difference(phi_wt, psi_wt, phi_mt, psi_mt)

    print("\nDone.")
