# Frozen bioRxiv Figure 2 inputs

Selected against Figure 2, page 6 of `2026_10_01_biorxiv_DPR.pdf`.
Both runs use seed 13, beta2=20, beta3=0.001, magnification=-0.05,
2,000 prototypes and 2,000 iterations. Primary beta1=0.00035;
panel B comparator beta1=0.002. Inputs, initialization and original trainer
hashes agree between runs. Training and rendering use the individual-V1 surface.

Each directory contains only the saved boundary, prototypes, initialization,
final checkpoint and original run metadata with portable asset paths.
`freeze.json` records file hashes and original manifest hashes. Historical
source paths in metadata are provenance, not rendering dependencies.
`partition-reference.npz` is the eight-area partition from the saved seed-13
export, with no forced apex cut. The full sweep and search code are excluded.

Run `python scripts/verify_figure2_freeze.py`, then
`python fig2/compose_figure.py` in the locked environment.
No training is needed. Full training parity is not yet verified.
