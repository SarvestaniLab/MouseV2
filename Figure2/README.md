# Figure 2: selected retinotopy model

Train the selected seed **2026111604** and reconstruct the updated five-panel
Figure 2. This compact package contains the single-run trainer, frozen inputs
and checkpoints, downstream analysis methods, and the exact-PDF renderer.

## Reconstruct the figure

From the repository root:

```sh
cd Figure2
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-render.txt
.venv/bin/python reproduce.py
```

On Windows use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.
Outputs appear in `reconstruction/`. The command builds all panels in a clean
temporary directory and requires the new PDF and PNG to match the frozen hashes.
The committed `figure2.pdf` remains untouched.

## Train the selected model

Training uses a separate Python 3.11 environment and a compatible NVIDIA GPU:

```sh
python3.11 -m venv .venv-training
.venv-training/bin/python -m pip install -r requirements-training.txt
.venv-training/bin/python replay_learning.py
.venv/bin/python reproduce.py --learning-maps reconstruction/learning_replay/maps.npz
```

The replay initializes from the specified seed, performs all 3,000 updates, and
writes new checkpoints to `reconstruction/learning_replay/`. It requires the
initial state and saved checkpoints to match the archived run exactly. Rendering
uses the frozen endpoint; after a successful replay this is the same endpoint
just trained. GPU/backend differences can prevent bitwise training parity.

## Rebuild the methods and figure

Using the environments above, train V1 on the CPU and rebuild all downstream
analysis with the new V1 and HVA endpoints:

```sh
.venv-training/bin/python training/v1.py --output-dir results/v1
.venv/bin/python build_figure_data.py --v1-fit results/v1/fit.npz --hva-maps reconstruction/learning_replay/maps.npz --render --output-dir results/rebuilt
```

Without a GPU, rebuild segmentation, injection mapping and registration from
the archived endpoints:

```sh
.venv/bin/python build_figure_data.py --render --output-dir results/reference-methods
```

Each build writes its own data and verification receipt. `--render` adds PDF,
PNG and SVG. Choose a new output directory for each run. These methods builds
report numerical differences from the archive; `reproduce.py` remains the
strict byte-exact check. Full HVA GPU replay has not been validated on this host.

See [METHODS.md](METHODS.md) for the model equations and code map,
and [REPRODUCING.md](REPRODUCING.md) for the schedule, verification scope and
inputs, and [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for scientific limits.
This is a review draft; final licensing and attribution remain pending.
