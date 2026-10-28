# Reproducing the draft snapshot

Use Python 3.8 and the supplied uv.lock. From the lab repository root, run
`cd Figure2` before the commands below. If already in this directory,
do not change directories again.
Create the environment with `uv sync --locked --group notebooks`.

## Train the selected model

The frozen primary example and its panel B comparator use this explicit recipe:

```sh
uv run --locked --group notebooks python scripts/run_mouse_v2.py \
  --seed 4 --beta1 0.00014 0.001 --beta2 10 --beta3 0.0001 \
  --iterations 1500 --prototype-count 2000 --map-h 80 --map-w 8 \
  --magnification -0.5 --boundary-retinotopy synthetic-waters-supported \
  --surface-svg data/waters2019/derived/WatersV1_NormalConditionalMedian_borders.svg \
  --sigma 1.2 --save-interval 0 --output-root results/figure2-retraining
```

For an execution check, change `--iterations 1500` to `--iterations 2` and use
`--output-root results/figure2-smoke`. The runner refuses to overwrite a nonempty
seed directory. Full retraining parity with the frozen checkpoints is unverified.
The saved checkpoints remain authoritative. The generic runner's defaults are
not this recipe. Available boundary modes are limited to included assets.

## Render Figure 2

```sh
python scripts/verify_figure2_freeze.py
uv run --locked --group notebooks python fig2/compose_figure.py
```

This renders the frozen maps to SVG, PDF, and PNG under `fig2/output/full/`.
Panel F needs three included Zhuang MAT arrays in `data/zhuang2017/`, the Zhuang
border tracing, and the Waters geometry calibration manifest. Training uses the
conditional-normal surface; rendering preserves the current individual-V1
surface. This difference still needs scientific review.

## Checks and provenance

Run `uv run --locked --group notebooks python -m unittest discover -s tests -v`.
The included existing tests check model energy and gradient equivalence and
cortical geometry. They are a subset of the development suite and do not
validate all manuscript claims.

`EXPORT_PROVENANCE.json` records SHA-256 hashes, the development HEAD, and
whether each exported file matched that commit. It describes working-tree
contents, so HEAD alone is not a complete identifier when any file differs.
The export manifest is preserved separately. No development Git history is copied.

## Still required for the paper

Verify full retraining and resolve the training/rendering geometry distinction.
Finish the seed-versus-mouse variability and parameter sensitivity analyses,
and finalize citation, licensing, and data redistribution.
The included historical Zhuang arrays require source/permission review before
public release. This is a locally tested draft.
