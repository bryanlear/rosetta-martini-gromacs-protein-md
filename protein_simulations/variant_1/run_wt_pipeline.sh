#!/bin/bash
set -eo pipefail

MDRUN="gmx mdrun -ntmpi 1 -ntomp 4 -nb gpu -pme gpu -bonded gpu -update cpu"
LOG="pipeline_v2.log"

log() { echo "$(date): $1" | tee -a "$LOG"; }
check_output() {
    if [ ! -f "$1" ]; then
        log "ERROR: Expected output $1 not found. Aborting."
        exit 1
    fi
}

# Stages 1, 2a, 2b already completed

# ============================================================
# STAGE 3: NPT C-rescale, dt=1fs, 1ns (stable)
# ============================================================
log "=== STAGE 3: NPT C-rescale (1fs, 1ns) ==="

gmx grompp -f stage3_npt_crescale.mdp -c stage2b_npt.gro -p topol_wt.top \
    -o stage3_npt.tpr -r stage2b_npt.gro -t stage2b_npt.cpt -maxwarn 5 2>&1 | tee -a "$LOG"
check_output stage3_npt.tpr

$MDRUN -deffnm stage3_npt 2>&1 | tee -a "$LOG"
check_output stage3_npt.gro
log "Stage 3 complete"

# ============================================================
# STAGE 4: Production (dt=2fs, 10ns) - attempt 2fs first
# ============================================================
log "=== STAGE 4: Production (2fs, 10ns) ==="

cat > prod_wt.mdp << 'MDPEOF'
integrator              = md
dt                      = 0.002
nsteps                  = 5000000
nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstxout-compressed      = 5000
nstlog                  = 5000
nstenergy               = 5000
nstcalcenergy           = 100
continuation            = yes
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4
cutoff-scheme           = Verlet
nstlist                 = 40
rcoulomb                = 1.2
rvdw                    = 1.2
vdw-modifier            = Force-switch
rvdw-switch             = 1.0
coulombtype             = PME
pme_order               = 4
fourierspacing          = 0.12
tcoupl                  = V-rescale
tc-grps                 = System
tau_t                   = 1.0
ref_t                   = 310
pcoupl                  = C-rescale
pcoupltype              = semiisotropic
tau_p                   = 5.0
ref_p                   = 1.0 1.0
compressibility         = 4.5e-5 4.5e-5
pbc                     = xyz
DispCorr                = no
gen_vel                 = no
MDPEOF

gmx grompp -f prod_wt.mdp -c stage3_npt.gro -p topol_wt.top \
    -o prod_wt.tpr -t stage3_npt.cpt -maxwarn 5 2>&1 | tee -a "$LOG"
check_output prod_wt.tpr

# Try production at dt=2fs
$MDRUN -deffnm prod_wt 2>&1 | tee -a "$LOG"

if [ -f prod_wt.gro ]; then
    log "=== PRODUCTION COMPLETE (dt=2fs) ==="
else
    log "dt=2fs failed, falling back to dt=1fs production"
    
    cat > prod_wt_1fs.mdp << 'MDPEOF2'
integrator              = md
dt                      = 0.001
nsteps                  = 10000000
nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstxout-compressed      = 10000
nstlog                  = 10000
nstenergy               = 10000
nstcalcenergy           = 100
continuation            = yes
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4
cutoff-scheme           = Verlet
nstlist                 = 40
rcoulomb                = 1.2
rvdw                    = 1.2
vdw-modifier            = Force-switch
rvdw-switch             = 1.0
coulombtype             = PME
pme_order               = 4
fourierspacing          = 0.12
tcoupl                  = V-rescale
tc-grps                 = System
tau_t                   = 1.0
ref_t                   = 310
pcoupl                  = C-rescale
pcoupltype              = semiisotropic
tau_p                   = 5.0
ref_p                   = 1.0 1.0
compressibility         = 4.5e-5 4.5e-5
pbc                     = xyz
DispCorr                = no
gen_vel                 = no
MDPEOF2

    gmx grompp -f prod_wt_1fs.mdp -c stage3_npt.gro -p topol_wt.top \
        -o prod_wt.tpr -t stage3_npt.cpt -maxwarn 5 2>&1 | tee -a "$LOG"
    
    $MDRUN -deffnm prod_wt 2>&1 | tee -a "$LOG"
    check_output prod_wt.gro
    log "=== PRODUCTION COMPLETE (dt=1fs fallback) ==="
fi

log "All stages finished successfully!"
