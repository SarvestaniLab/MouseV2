# Mouse V2 Figure 2 publication draft

This review snapshot contains a constructive model of visual-field
reversals outside mouse V1 and Figure 2 from the October 1 bioRxiv draft, using its selected seed-13 runs.
It is not yet a complete manuscript release. The Figure 2 model example and
comparison are [frozen](data/publication/figure2/README.md); full Figure 2
rendering is included; seed variability and parameter sensitivity analysis remain pending.

## Run

From the lab repository root, enter `Figure2/` first. This folder
has its own Python environment and lockfile, independent of the other analyses.

```sh
cd Figure2
uv sync --locked --group notebooks
uv run --locked --group notebooks python scripts/check_environment.py
python scripts/verify_figure2_freeze.py
uv run --locked --group notebooks python fig2/compose_figure.py
uv run --locked --group notebooks python -m unittest discover -s tests -v
```

Figure 2 renders from the frozen checkpoints. See [REPRODUCING.md](REPRODUCING.md) for the exact saved-run settings and
remaining retraining limitations; the generic runner defaults are development settings. The notebooks dependency group supplies plotting dependencies required
by the runner. See [reproduction instructions](REPRODUCING.md),
[data availability](DATA.md), and [limitations](KNOWN_LIMITATIONS.md).

The model is adapted from Yu et al. (2020), *A twisted visual field map in the
primate dorsomedial cortex predicted by topographic continuity*, Science
Advances 6, eaaz8673. Author metadata and the software license must be finalized
before the final release; this draft does not grant a new software or data license.
