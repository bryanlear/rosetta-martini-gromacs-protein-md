#!/bin/bash
set -eo pipefail

WD="/backmap_cg2at/backmap_charmm36/FINAL/production_for_WT"
cd "$WD"

TPR="prod_wt.tpr"
XTC="prod_wt.xtc"
EDR="prod_wt.edr"
NDX="index_wt.ndx"

# --- 5. MSD (POPC) ---
echo "[5/9] MSD POPC..."
echo "13" | gmx msd -s $TPR -f $XTC -o msd_popc_wt.xvg -n $NDX -lateral z 2>&1 | tail -3

# --- 6. Energy extractions ---
echo "[6/9] Energy extractions..."
echo "Potential" | gmx energy -f $EDR -o potential_wt.xvg 2>&1 | tail -2
echo "Pressure" | gmx energy -f $EDR -o pressure_wt.xvg 2>&1 | tail -2
echo "Temperature" | gmx energy -f $EDR -o temperature_wt.xvg 2>&1 | tail -2
echo "Total-Energy" | gmx energy -f $EDR -o total_energy_wt.xvg 2>&1 | tail -2
echo "Density" | gmx energy -f $EDR -o density_wt.xvg 2>&1 | tail -2
echo "Box-X Box-Y" | gmx energy -f $EDR -o box_xy_wt.xvg 2>&1 | tail -2
echo "Box-X" | gmx energy -f $EDR -o box_vol_wt.xvg 2>&1 | tail -2

# --- 7. SCD order parameters (attempt) ---
echo "[7/9] SCD order parameters..."
echo "13" | gmx order -s $TPR -f $XTC -o scd_sn1_wt.xvg -n $NDX -od deuter_sn1_wt.xvg -d z 2>&1 | tail -3 || echo "  SCD failed (may need custom atom index)"

# --- 8. Thickness proxy ---
echo "[8/9] Bilayer thickness proxy (multi-group density)..."
echo -e "1\n13\n14\n15\n16\n20" | gmx density -s $TPR -f $XTC -o density_groups_wt.xvg -n $NDX -center -ng 5 2>&1 | tail -3 || echo "  Multi-group density skipped"

echo ""
echo "========================================="
echo "  Remaining WT Analyses Complete!"
echo "========================================="
echo "Output files:"
ls "$WD"/*_wt.xvg "$WD"/*_wt.dat 2>/dev/null
