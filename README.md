

## Project Overview

This repository contains a multi-scale computational workflow designed to substantiate the pathogenicity (or lack thereof) of variants classified as pathogenic by my own automated ACMG/AMP Bayesian variant classification pipeline: [VarClass](https://www.varclass.com). 

The main target is to provide a mechanistic basis for variants previously classified as pathogenic within the pipeline's Bayesian scoring framework.

## Methodology

### Atomistic Mutation
Initial protein structures (AlphaFold-predicted) are processed to introduce point mutations using Rosetta.

```bash
python mutation_analysis.py --pdb scn5a_af.pdb --position xxx --mutation His
```

## Multi-scale Coarse-Graining (Martini 3)

The resulting protein is mapped to a CG representation using `martinize2` to extend simulation timescales to the microsecond range.

* **Force Field**: Martini 3.0.0.1.

* **Parameters**: Elastic network enabled, automated disulfide bond assignment (0.24 nm constraint), and DSSP-based secondary structure assignment.

## Asymmetric Membrane Construction

An asymmetric lipid bilayer attempting to reflect a physiological cardiac membrane composition (SCN5A gene) was constructed using INSANE.

* **Upper Leaflet**: POPC:POPE:CHOL (6:2:4)

* **Lower Leaflet**: POPC:POPE:POPS:CHOL (4:3:1:4)

* **System Size**: 18.2×18.2×25.0 nm (~750,000 particles).

## Equilibration Strategy

The system underwent a multi-stage equilibration to ensure stability:

1. **EM**: Steepest descent to relax steric clashes
2. **NVT Ensemble**: $310K$ thermalization with postion restraints.
3. **NPT (Berendsen)**: Gradual pressure coupling to reach target density.

$$\mu = \left[ 1 - \frac{\beta \Delta t}{\tau_p} (P_{ref} - P) \right]^{1/3}$$
4. **NPT (Parrinello-Rahman)**: Pressure coupling for production-ready tranjectories:
$$\frac{d^2 \mathbf{b}}{dt^2} = V \mathbf{b}^{-T} \mathbf{W}^{-1} (\mathbf{P} - \mathbf{P}_{ref})$$

$$\mathbf{W}^{-1} = \frac{4 \pi^2 \beta}{3 \tau_p^2 L}$$

## Back-mapping (CG2AT)

To recover atomic resolution for high fidelity interaction analysis, CGs are back-mapped to **CHARMM36** force field.

* Align atomistic fragments to CG bead centers of mass:
$$\mathbf{R}_i = \frac{\sum_{j \in I} m_j \mathbf{r}_j}{\sum_{j \in I} m_j}$$

* EM and position-restrained MD to resolve clashed:
$$\nabla V(\mathbf{r}_1, \mathbf{r}_2, \dots, \mathbf{r}_N) = 0$$