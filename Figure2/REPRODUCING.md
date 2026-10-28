# Reproduce bioRxiv Figure 2

The target is Figure 2 on page 6 of `2026_10_01_biorxiv_DPR.pdf`.
From the lab repository root enter `Figure2/`, then run:

```sh
uv sync --locked --group notebooks
uv run --locked --group notebooks python scripts/verify_figure2_freeze.py
uv run --locked --group notebooks python fig2/compose_figure.py
uv run --locked --group notebooks python -m unittest discover -s tests -v
```

The composition is exported as `fig2/output/full/fig2.png`, `.pdf` and `.svg`.
The default uses the frozen seed-13 primary and matched comparator:
beta1=0.00035 and 0.002 respectively, beta2=20, beta3=0.001,
magnification=-0.05, 2,000 prototypes, 2,000 iterations, and an 80 x 8 grid.
Both use the same individual-V1 training/rendering surface and synthetic-rebalanced
boundary. Panel E uses the saved eight-area recipe, without a forced apex cut.

`scripts/make_figure2_manuscript.py` applies the manuscript layout to the shared
renderer. It retains the original panel D interpolation. Later periodic-ribbon
and explanatory-layout revisions are not used. Arial is preferred; when absent,
Liberation Sans is used as a metric-compatible fallback. Exact typography and
PDF bytes can therefore differ from the typeset manuscript. The manuscript
caption says nine areas, whereas its displayed panel E and these results show eight.

## What is reproduced

Rendering uses included final checkpoints, not a new optimization or search.
The tests compare the computed partition and field-sign map with the archived
seed-13 export and check that frozen checkpoint/geometry drift is rejected.
The two saved runs share initialization, boundary, prototypes and trainer hashes.
The generic model runner remains available for experiments; it is not a verified
bit-for-bit retraining recipe for these historical runs. Historical trainer hashes
are provenance and do not assert that the current generic trainer is identical.
Full retraining parity remains unfinished.

## Export provenance and remaining work

`EXPORT_PROVENANCE.json` records the development HEAD, per-file SHA-256 hashes,
and whether each exported file matched that commit. No private Git history,
parameter-search drivers, or bulk sweep results are copied.
Only the two final selected runs are included.

Finalize citation, licensing and data redistribution review, and the planned
variability/sensitivity work before treating this draft as the final paper release.
