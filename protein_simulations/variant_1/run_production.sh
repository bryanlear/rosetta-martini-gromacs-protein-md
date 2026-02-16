#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Production MD — SCN5A RXXXH(100 ns)
# Parrinello-Rahman, semiisotropic, CHARMM36 all-atom
# ──────────────────────────────────────────────────────────────
set -euo pipefail

FINAL_DIR="/backmap_cg2at/backmap_charmm36/FINAL"
WORK_DIR="${FINAL_DIR}/production"
cd "${WORK_DIR}"

echo "=== [1/2] grompp — production ==="
gmx grompp \
  -f  production.mdp \
  -c  "${FINAL_DIR}/npt_unrest/npt_unrest.gro" \
  -t  "${FINAL_DIR}/npt_unrest/npt_unrest.cpt" \
  -p  "${FINAL_DIR}/topol_final.top" \
  -o  production.tpr \
  -maxwarn 0 \
  2>&1 | tee grompp.log

echo ""
echo "=== [2/2] mdrun — 100 ns production ==="
gmx mdrun \
  -deffnm production \
  -v \
  -ntomp 4 \
  -nb gpu \
  -pme gpu \
  -bonded gpu \
  -update gpu \
  -pin on \
  2>&1 | tee mdrun.log

echo ""
echo "=== Production MD complete ==="
