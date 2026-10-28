"""Verify the selected Figure 2 saved model inputs without running training."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root = Path(root)
    freeze = json.loads((root / "data/publication/figure2/freeze.json").read_text())
    for name, record in freeze["files"].items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Frozen Figure 2 input changed: " + name)
    for run in freeze["runs"].values():
        directory = root / run["directory"]
        if {p.name for p in directory.glob("y-*.data")} != {run["final_checkpoint"]}:
            raise ValueError("Frozen run must contain exactly its selected checkpoint")
    return freeze


if __name__ == "__main__":
    result = verify()
    print("Verified {} frozen Figure 2 inputs; seed {}, iteration {}.".format(
        len(result["files"]), result["runs"]["primary"]["seed"],
        result["runs"]["primary"]["iterations"]))
