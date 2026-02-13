# Equilibration MDP variables (Martini)

Source files:
- `martini_files/M3_Ionizable_Lipids-main/Case_studies/Quantifying_stalk_formation/mdps/eq_1.mdp`
- `martini_files/M3_Ionizable_Lipids-main/Case_studies/Quantifying_stalk_formation/mdps/eq_2.mdp`
- `martini_files/M3_Ionizable_Lipids-main/Case_studies/Quantifying_stalk_formation/mdps/eq_3.mdp`

Keys extracted (as requested): `integrator`, `dt`, `nsteps`, `tcoupl`, `tc-grps`, `tau-t`, `ref-t`, `pcoupl`, `pcoupltype`, `tau-p`, `ref-p`, `compressibility`, `refcoord-scaling` (file uses `refcoord_scaling`), and any `define =` line(s).

## eq_1.mdp

| Key | Value |
|---|---|
| integrator | `md` |
| dt | `0.002` |
| nsteps | `500000` |
| tcoupl | `v-rescale` |
| tc-grps | `Solute Solvent` |
| tau-t | `1.0 1.0` |
| ref-t | `310 310` |
| pcoupl | `no` (line: `Pcoupl = no ;Berendsen`) |
| pcoupltype | `isotropic` |
| tau-p | `4.0` |
| ref-p | `1.0 1.0` |
| compressibility | `3e-4 3e-4` |
| refcoord-scaling | `com` (line: `refcoord_scaling = com`) |
| define | `-DPOSRES -DBBREST` |

## eq_2.mdp

| Key | Value |
|---|---|
| integrator | `md` |
| dt | `0.01` |
| nsteps | `500000` |
| tcoupl | `v-rescale` |
| tc-grps | `Solute Solvent` |
| tau-t | `1.0 1.0` |
| ref-t | `310 310` |
| pcoupl | `Berendsen` |
| pcoupltype | `semiisotropic` |
| tau-p | `4.0` |
| ref-p | `1.0 1.0` |
| compressibility | `3e-4 3e-4` |
| refcoord-scaling | `com` (line: `refcoord_scaling = com`) |
| define | _(not set in file)_ |

## eq_3.mdp

| Key | Value |
|---|---|
| integrator | `md` |
| dt | `0.02` |
| nsteps | `500000` |
| tcoupl | `v-rescale` |
| tc-grps | `Solute Solvent` |
| tau-t | `1.0 1.0` |
| ref-t | `310 310` |
| pcoupl | `Berendsen` |
| pcoupltype | `semiisotropic` |
| tau-p | `4.0` |
| ref-p | `1.0 1.0` |
| compressibility | `3e-4 3e-4` |
| refcoord-scaling | `com` (line: `refcoord_scaling = com`) |
| define | _(not set in file)_ |
