#!/bin/bash
# ============================================================
# submit_hremc.sh — HPC submission script for HREMC
# Adjust cluster scheduler (SLURM / PBS / SGE)
# ============================================================
#
# This is a SLURM template. Modify as needed for HPC system.

#SBATCH --job-name=hremc_scn5a
#SBATCH --partition=gpu            # Adjust partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48         # Enough for 24 replicas × 2 threads
#SBATCH --gres=gpu:1               # Optional: GPU acceleration
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=hremc_%j.out
#SBATCH --error=hremc_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@domain.com

# ============================================================
# Environment
# ============================================================
# module load gromacs/2024         # Uncomment for module system
# module load python/3.11
# source ~/venv/bin/activate       # Uncomment for virtualenv

set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "${WORKDIR}"

echo "============================================================"
echo " HREMC Job: ${SLURM_JOB_ID:-local}"
echo " Host:      $(hostname)"
echo " Partition: ${SLURM_JOB_PARTITION:-local}"
echo " Cores:     ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo " Started:   $(date)"
echo " Workdir:   ${WORKDIR}"
echo "============================================================"

# ============================================================
# Choose system: wt, mutant, or both
# ============================================================
SYSTEM="${1:-both}"

# ============================================================
# Step 1: Setup (if not already done)
# ============================================================
if [[ ! -d "wt/replica_00" ]] && [[ "${SYSTEM}" == "both" || "${SYSTEM}" == "wt" ]]; then
    echo "Running setup..."
    bash run_hremc.sh setup "${SYSTEM}"
fi

# ============================================================
# Step 2: Equilibration (if not already done)
# ============================================================
for s in wt mutant; do
    if [[ "${SYSTEM}" != "both" && "${SYSTEM}" != "${s}" ]]; then
        continue
    fi
    if [[ ! -f "${s}/replica_00/state.gro" ]]; then
        echo "Running equilibration for ${s}..."
        bash run_hremc.sh equilibrate "${s}"
    fi
done

# ============================================================
# Step 3: Production HREMC
# ============================================================
echo ""
echo "Starting production HREMC..."
echo ""

# Check if resuming
RESUME_FLAG=""
for s in wt mutant; do
    if [[ "${SYSTEM}" != "both" && "${SYSTEM}" != "${s}" ]]; then
        continue
    fi
    if [[ -f "${s}/checkpoints/checkpoint_latest.json" ]]; then
        RESUME_FLAG="--resume"
        echo "Resuming from checkpoint for ${s}..."
    fi
done

python3 hremc_engine.py --config config.yaml --system "${SYSTEM}" --production ${RESUME_FLAG}

echo ""
echo "============================================================"
echo " HREMC COMPLETE"
echo " Finished:  $(date)"
echo "============================================================"
