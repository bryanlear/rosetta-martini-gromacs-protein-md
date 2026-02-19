#!/bin/bash
# ============================================================
# run_hremc.sh — Master run script for HREMC
# Hamiltonian Replica Exchange Monte Carlo
# SCN5A Nav1.5 (WT & RxxxH mutant) — Martini 3 CG
# ============================================================
#
# Usage:
#   ./run_hremc.sh setup              # Generate replica directories
#   ./run_hremc.sh equilibrate [sys]   # EM + NVT + NPT for all replicas
#   ./run_hremc.sh production [sys]    # Run production HREMC cycles
#   ./run_hremc.sh resume [sys]        # Resume from checkpoint
#   ./run_hremc.sh status [sys]        # Check run status
#   ./run_hremc.sh clean [sys]         # Remove intermediate files
#
#   sys = wt | mutant | both (default: both)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

CONFIG="config.yaml"
PYTHON="python3"

# ============================================================
# Helpers
# ============================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

check_deps() {
    # Check Python
    if ! command -v ${PYTHON} &>/dev/null; then
        echo "ERROR: python3 not found"
        exit 1
    fi

    # Check PyYAML
    ${PYTHON} -c "import yaml" 2>/dev/null || {
        echo "ERROR: PyYAML not installed. Run: pip install pyyaml"
        exit 1
    }

    # Check GROMACS
    if ! command -v gmx &>/dev/null; then
        echo "ERROR: gmx (GROMACS) not found in PATH"
        exit 1
    fi

    log "Dependencies OK: python3, PyYAML, GROMACS $(gmx --version 2>&1 | head -1)"
}

# ============================================================
# Commands
# ============================================================
cmd_setup() {
    local sys="${1:-both}"
    log "Setting up HREMC replicas (system: ${sys})"
    ${PYTHON} setup_hremc.py --config ${CONFIG} --system "${sys}"
}

cmd_equilibrate() {
    local sys="${1:-both}"
    log "Running equilibration (system: ${sys})"
    ${PYTHON} hremc_engine.py --config ${CONFIG} --system "${sys}" --equilibrate
}

cmd_production() {
    local sys="${1:-both}"
    log "Running production HREMC (system: ${sys})"
    ${PYTHON} hremc_engine.py --config ${CONFIG} --system "${sys}" --production
}

cmd_resume() {
    local sys="${1:-both}"
    log "Resuming HREMC from checkpoint (system: ${sys})"
    ${PYTHON} hremc_engine.py --config ${CONFIG} --system "${sys}" --production --resume
}

cmd_status() {
    local sys="${1:-both}"
    log "Checking HREMC status (system: ${sys})"

    for s in wt mutant; do
        if [[ "${sys}" != "both" && "${sys}" != "${s}" ]]; then
            continue
        fi
        echo ""
        echo "=== ${s^^} ==="
        ckpt="${s}/checkpoints/checkpoint_latest.json"
        if [[ -f "${ckpt}" ]]; then
            cycle=$(${PYTHON} -c "import json; d=json.load(open('${ckpt}')); print(d['cycle'])")
            n_att=$(${PYTHON} -c "import json; d=json.load(open('${ckpt}')); print(d['total_exchanges_attempted'])")
            n_acc=$(${PYTHON} -c "import json; d=json.load(open('${ckpt}')); print(d['total_exchanges_accepted'])")
            echo "  Last cycle:        ${cycle}"
            echo "  Exchanges tried:   ${n_att}"
            echo "  Exchanges accepted:${n_acc}"
            if [[ ${n_att} -gt 0 ]]; then
                rate=$(${PYTHON} -c "print(f'{${n_acc}/${n_att}*100:.1f}%')")
                echo "  Acceptance rate:   ${rate}"
            fi
        else
            echo "  No checkpoint found. Run setup and equilibration first."
        fi

        # Count existing replica directories
        n_dirs=$(find "${s}" -maxdepth 1 -type d -name "replica_*" 2>/dev/null | wc -l | tr -d ' ')
        echo "  Replica dirs:      ${n_dirs}"

        if [[ -f "${s}/hremc_report.txt" ]]; then
            echo "  Report:            ${s}/hremc_report.txt"
        fi
    done
    echo ""
}

cmd_clean() {
    local sys="${1:-both}"
    log "Cleaning intermediate files (system: ${sys})"

    for s in wt mutant; do
        if [[ "${sys}" != "both" && "${sys}" != "${s}" ]]; then
            continue
        fi
        echo "Cleaning ${s}..."
        for rep_dir in ${s}/replica_*/; do
            if [[ -d "${rep_dir}" ]]; then
                # Remove old segment files but keep checkpointed ones and state.gro
                find "${rep_dir}" -name "seg_*.trr" -delete 2>/dev/null || true
                find "${rep_dir}" -name "seg_*.log" -delete 2>/dev/null || true
                find "${rep_dir}" -name "#*#" -delete 2>/dev/null || true
                find "${rep_dir}" -name "rerun_*" -delete 2>/dev/null || true
                find "${rep_dir}" -name "mdout.mdp" -delete 2>/dev/null || true
            fi
        done
        echo "  Done."
    done
}

# ============================================================
# Main dispatch
# ============================================================
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 {setup|equilibrate|production|resume|status|clean} [wt|mutant|both]"
    exit 1
fi

CMD="$1"
SYS="${2:-both}"

check_deps

case "${CMD}" in
    setup)       cmd_setup "${SYS}" ;;
    equilibrate) cmd_equilibrate "${SYS}" ;;
    production)  cmd_production "${SYS}" ;;
    resume)      cmd_resume "${SYS}" ;;
    status)      cmd_status "${SYS}" ;;
    clean)       cmd_clean "${SYS}" ;;
    *)
        echo "Unknown command: ${CMD}"
        echo "Usage: $0 {setup|equilibrate|production|resume|status|clean} [wt|mutant|both]"
        exit 1
        ;;
esac

log "Done."
