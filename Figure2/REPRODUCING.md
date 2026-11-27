# Reproducing Figure 2

Follow the two environment setups in [README.md](README.md). The complete
machine-readable contract is [REPRODUCTION_RECIPE.json](REPRODUCTION_RECIPE.json).
`python reproduce.py --show-recipe` prints it without installing dependencies.

## Exact figure reconstruction

Python 3.12 and every version in `requirements-render.txt` are required.
This includes `fonttools==4.59.0`: newer font subsetting changes embedded PDF
bytes even when the panel drawing commands are unchanged.
`reproduce.py` checks the package manifest, copies only frozen inputs to a
clean temporary directory, removes prior generated panels, renders A/B, C and
D/E, then assembles the figure. It rejects mismatched PDF or PNG hashes.
Committed inputs and reference artifacts are never overwritten, including when
an unsafe `--output-dir` is supplied.

Reference PDF SHA-256:
`19281a4338937a72534838a06b5b0762ef8d6906a35e1780bc6024b50a39e97b`

Successful outputs are `reconstruction/figure2.pdf`, `.png`, `.svg` and
`verification.json`. SVG internal IDs may differ; SVG byte identity is not
required. `verify_package.py` checks input checksums without rendering.

## Selected training run

Run `family_05_seed_2026111604` uses seed **2026111604**, b1=**1.45e-5**,
b2=**100**, b3=**0.03**, 2,000 visual-field prototypes, and a 160 by 16
trainable surrounding-cortex grid. The frozen cortical surface and fixed V1
boundary are supplied in `training/training_inputs.npz`.

The optimizer performs 3,000 updates with eta0=0.1, momentum=0.8,
eta_decay=0.9933221725495059, kappa0=44, and
kappa_decay=0.001533921712758013. Checkpoints are recorded every 25 updates.
`training/training_config.json` stores the run and schedule.

`training/model.py` contains initialization, graph construction and the training
loop; `training/optimizer.py` contains the float64 GPU stencil and session
configuration. The elastic-net objective and RMSProp implementation are in
`elastictopy/`. The package provides the selected run and its reconstruction
workflow.

`replay_learning.py` requires TensorFlow 2.17.1 and a compatible NVIDIA GPU.
Use Python 3.11 with `requirements-training.txt` on Linux or WSL2. The NumPy and
SciPy training pins define the public replay environment; the archived receipt
records TensorFlow 2.17.1 but does not record those two package versions.
Outputs are new `maps.npz`, `trajectory.npz`, `training_config.json`, and
`replay-verification.json` under `reconstruction/learning_replay/`.
The full new trajectory and its configuration are saved even when checkpoint
comparison fails. The initial state, step 2500, step 3000 and final endpoint must match
`fit_sources/hva_maps.npz` bit for bit. A backend mismatch fails explicitly;
newly computed maps and the comparison receipt remain available for inspection.
The archived successful replay receipt is `training/replay-verification.json`;
it records an earlier replay, not a new validation on this host.

## CPU V1 fit

`training/v1.py` uses the same seed, loss coefficients and 3,000-update schedule
on a nonperiodic 41 by 41 harmonic grid, with 39 by 39 trainable interior nodes.
It uses the training environment but does not require a GPU. The 160 original
cortical perimeter samples are in `training/v1_boundary_cortex.npy` in SVG
coordinates. Keep these exact samples: reconstructing them through the HVA mesh
changes coordinates by about 1e-14 and can change the optimized endpoint.
The corner placement is selected using cortical cell orientation alone (offset
33 for this input). Initial RFs use the shared prototypes and seeded noise.

```sh
.venv-training/bin/python training/v1.py --output-dir results/v1
```

This writes `fit.npz` and `verification.json`. The receipt reports endpoint RMS,
maximum error and exactness against the archive. Completion alone does not
assert endpoint parity. Full CPU replay with the original perimeter matched the
archived endpoint and recorded loss history exactly in the validated environment
(Python 3.11, NumPy 1.26.4, SciPy 1.14.1, TensorFlow 2.17.1).

## From learning to panels

```sh
.venv/bin/python build_figure_data.py --v1-fit results/v1/fit.npz --hva-maps reconstruction/learning_replay/maps.npz --render --output-dir results/rebuilt
```

Run the builder in the pinned Python 3.12 rendering environment. It recomputes
field signs, the selected eight-parcel segmentation, injection memberships,
inverse V1 registration and light display smoothing. It consumes the final HVA
checkpoint; checkpoint-history validation remains the responsibility of
`replay_learning.py`. Inputs must have the selected grid shapes and V1 boundary.

Omitting a fit argument explicitly uses that archived endpoint. With both
omitted, the builder checks regenerated arrays against the archive at absolute
tolerance 1e-9 (zero relative tolerance), including exact discrete labels and
memberships. With supplied fits it reports comparisons without requiring parity.
It saves regenerated data, selected metadata and a verification receipt; add
`--render` for PDF, PNG and SVG. Existing output directories are rejected.
The builder uses the recorded segmentation parameters and injection sites.

`reproduce.py --learning-maps reconstruction/learning_replay/maps.npz` remains
the separate strict check: verify archived checkpoints, then render the frozen
arrays and require exact PDF/PNG hashes. The numerical methods build does not
promise byte-identical rendering. `EXPORT_PROVENANCE.json` records extraction
sources and `package_manifest.json` protects shipped inputs.
