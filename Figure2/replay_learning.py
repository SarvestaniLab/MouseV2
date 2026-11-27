"""Optional exact GPU replay of the frozen Figure 2 learning run.

Frozen inputs and reference checkpoints are never overwritten. New replay outputs go under
reconstruction/learning_replay unless --output-dir is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


ROOT = Path(__file__).resolve().parent


def array_sha256(array: np.ndarray) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reconstruction" / "learning_replay",
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise SystemExit(
            "Learning replay requires Python 3.11; rendering uses Python 3.12."
        )
    output = args.output_dir.resolve()
    from reproduce import verify_package, validate_output_dir

    validate_output_dir(output)
    verify_package()

    required = {"numpy": "1.26.4", "scipy": "1.14.1", "tensorflow": "2.17.1"}
    for package, expected in required.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "MISSING"
        if actual != expected:
            raise SystemExit(
                f"Learning requires {package}=={expected}; found {actual}. "
                "Install requirements-training.txt in a Python 3.11 environment."
            )
    # Set deterministic TensorFlow/thread configuration before importing NumPy.
    from training import model
    import numpy as np

    archive = ROOT / "training"
    config = json.loads((archive / "training_config.json").read_text())
    run = config["run"]
    tf = model.bench.tf
    if tf.__version__ != "2.17.1":
        raise SystemExit("Learning replay requires TensorFlow 2.17.1.")
    if not tf.config.list_physical_devices("GPU"):
        raise SystemExit(
            "Learning replay requires a compatible GPU/TensorFlow backend."
        )
    output.mkdir(parents=True, exist_ok=True)
    # An interrupted or failed rerun must not leave an earlier success receipt.
    (output / "replay-verification.json").unlink(missing_ok=True)
    inputs = np.load(archive / "training_inputs.npz")
    original = np.load(ROOT / "fit_sources/hva_maps.npz")
    figure = np.load(ROOT / "data/figure2_data.npz")
    figure_endpoint = figure["rf_surround"][:, 1:].transpose(1, 0, 2).reshape(-1, 2)
    np.testing.assert_array_equal(original["final"], figure_endpoint)
    initial = model.seeded_initial(inputs["targets"], run["seed"])
    np.testing.assert_array_equal(initial, original["initial"])
    plan = {
        "schedule": config["schedule"],
        "snapshots": list(range(25, config["schedule"]["steps"] + 1, 25)),
        "mesh_svg": str(ROOT / "fit_sources/expanded_footprint.svg"),
    }
    print(
        json.dumps({"run": run, "schedule": config["schedule"]}, indent=2), flush=True
    )
    graph = model.build_graph(
        plan, inputs["surface"], inputs["targets"], inputs["boundary"]
    )
    with tf.Session(config=model.bench.config()) as session:
        snapshots, seconds, partial_costs = model.train_one(
            session, graph, run, plan, initial
        )
    comparisons = {
        key: {
            "exact": bool(np.array_equal(original[key], snapshots[key])),
            "max_abs": float(np.max(abs(original[key] - snapshots[key]))),
            "sha256": array_sha256(snapshots[key]),
        }
        for key in original.files
    }
    metadata = {
        "config": dict(run, steps=plan["schedule"]["steps"]),
        "comparisons": comparisons,
        "all_saved_checkpoints_bitwise_equal": all(
            value["exact"] for value in comparisons.values()
        ),
        "tensorflow": tf.__version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "seconds": seconds,
        "partial_costs": [float(value) for value in partial_costs],
        "checkpoint_interval": 25,
        "schedule": config["schedule"],
        "backend": "original expanded-mesh GPU float64 graph",
        "temporal_interpolation": False,
        "reference_checkpoints_untouched": True,
    }
    np.savez_compressed(output / "maps.npz", **snapshots)
    np.savez_compressed(
        output / "trajectory.npz",
        step_0=initial,
        **{key: value for key, value in snapshots.items() if key.startswith("step_")},
    )
    shutil.copy2(archive / "training_config.json", output / "training_config.json")
    metadata["status"] = (
        "PASS" if metadata["all_saved_checkpoints_bitwise_equal"] else "FAIL"
    )
    (output / "replay-verification.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if metadata["status"] != "PASS":
        raise SystemExit(
            "Training completed, but checkpoints differ from the reference; "
            "see replay-verification.json. New maps and trajectory are preserved."
        )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
