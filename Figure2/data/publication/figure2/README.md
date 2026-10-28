# Frozen Figure 2 model example

Frozen on 2026-10-21 from the saved runs selected by `fig2/common.py`.
The primary publication example is the saved seed-4 map used by panels C-F.
Panel B retains its paired high-smoothness comparison.

| Setting | Primary | Panel B comparison |
| --- | --- | --- |
| beta1 | 0.00014 | 0.001 |
| beta2 | 10 | 10 |
| beta3 | 0.0001 | 0.0001 |
| Seed | 4 | 4 |
| Final iteration | 1500 | 1500 |
| Prototype count | 2000 | 2000 |
| Prototype density c | -0.5 | -0.5 |
| Grid | 80 rows by 8 outward columns | Same |
| Boundary | synthetic-waters-supported | Same |

`freeze.json` records hashes for each selected checkpoint, initial map, prototype
array, boundary array, reconstructed run manifest, and relevant geometry/input
files. The copied checkpoint and array bytes are unchanged from the saved
Figure 2 runs. Run `python scripts/verify_figure2_freeze.py` from the project
root to check them. Figure 2's standard panel/composition entry points also
verify the freeze when using either frozen default run.

## Geometry distinction

The saved runs were trained using
`WatersV1_NormalConditionalMedian_borders.svg`. At freeze time the Figure 2
code inherited the individual-V1 median renderer default from `mouse_surface`.
The renderer is now explicitly pinned to
`WatersV1_IndividualV1Median_borders.svg` to preserve that current behavior.
The earlier Figure 2 README incorrectly described the rendering default as
conditional-normal. Training and rendering geometry are recorded separately;
whether the final paper should use the same geometry for both remains a review
item. This freeze does not silently reinterpret the saved training run.

## Metadata and retraining

The original Figure 2 run directories lack the `run-manifest.json` files
required by the current renderer. The supplied manifests are reconstructed
from their parent `sweep.json` files, with source hashes and an explicit
reconstruction label. Grid dimensions are consistent with the 640-node
checkpoints and the renderer's 80-row reshape.

The following command explicitly specifies the recorded training settings for
a future reproduction attempt using the current implementation:

```sh
uv run --locked --group notebooks python scripts/run_mouse_v2.py \
  --seed 4 --beta1 0.00014 0.001 --beta2 10 --beta3 0.0001 \
  --iterations 1500 --prototype-count 2000 --map-h 80 --map-w 8 \
  --magnification -0.5 --boundary-retinotopy synthetic-waters-supported \
  --surface-svg data/waters2019/derived/WatersV1_NormalConditionalMedian_borders.svg \
  --sigma 1.2 --save-interval 0 --output-root results/figure2-retraining
```

Full retraining has not been verified against the frozen checkpoints, and the
historical training source revision is not established. The source commit in
`freeze.json` identifies this selection context, not necessarily the code that
originally trained the maps. The saved outputs are the authoritative frozen
example. The generic runner retains its development defaults; use all explicit
options above rather than `--seed 4` alone.

The curated package includes the complete composition renderer and the three
Zhuang arrays it requires. Their redistribution review remains pending.
The example arose through exploratory selection; robustness still requires the
planned seed and parameter sensitivity analyses.
