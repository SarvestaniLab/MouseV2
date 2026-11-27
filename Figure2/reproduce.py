"""Reconstruct Figure 2 in a clean room and require an exact PDF hash match."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
RECIPE_PATH = ROOT / "REPRODUCTION_RECIPE.json"
MANIFEST_PATH = ROOT / "package_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_package() -> None:
    manifest = load_json(MANIFEST_PATH)
    failures = []
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            failures.append(f"missing: {entry['path']}")
            continue
        actual = sha256(path)
        if actual != entry["sha256"]:
            failures.append(
                f"checksum: {entry['path']} (expected {entry['sha256']}, got {actual})"
            )
    if failures:
        raise SystemExit("Package verification failed:\n  " + "\n  ".join(failures))


def verify_environment(recipe: dict) -> dict[str, str]:
    required_python = tuple(recipe["render_pipeline"]["python_major_minor"])
    if sys.version_info[:2] != required_python:
        raise SystemExit(
            f"Exact rendering requires Python {required_python[0]}.{required_python[1]}; "
            f"running {sys.version_info.major}.{sys.version_info.minor}. "
            "Create the environment from requirements-render.txt."
        )
    actual = {}
    mismatches = []
    for distribution, expected in recipe["render_pipeline"]["distributions"].items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "MISSING"
        actual[distribution] = version
        if version != expected:
            mismatches.append(f"{distribution}: expected {expected}, got {version}")
    if mismatches:
        raise SystemExit(
            "Exact rendering requires the pinned environment:\n  "
            + "\n  ".join(mismatches)
            + "\nInstall requirements-render.txt in a clean Python 3.12 environment."
        )
    return actual


def clean_generated_outputs(work_root: Path) -> None:
    relative_paths = [
        "figure2_compositions/clean_export/panels_AB_redesigned.pdf",
        "figure2_compositions/clean_export/panels_AB_redesigned.png",
        "figure2_compositions/clean_export/panels_AB_redesigned.svg",
        "figure2_compositions/clean_export/panels_DE_clean.pdf",
        "figure2_compositions/clean_export/panels_DE_clean.svg",
        "figure2_compositions/clean_export/figure2_clean.pdf",
        "figure2_compositions/clean_export/figure2_clean.png",
        "figure2_compositions/clean_export/figure2_clean.svg",
        "ribbon_local/panel_C_dense.pdf",
        "ribbon_local/panel_C_dense.png",
        "ribbon_local/panel_C_dense.svg",
        "ribbon_local/panel_C_dense_notes.txt",
    ]
    for relative in relative_paths:
        path = work_root / relative
        if path.exists():
            path.unlink()


def validate_output_dir(output_dir: Path) -> None:
    """Reject destinations that could overwrite any committed package input."""
    output_dir = output_dir.resolve()
    if ROOT == output_dir or ROOT.is_relative_to(output_dir):
        raise SystemExit("Output directory must not contain the package root.")
    for name in (
        "figure2.pdf",
        "figure2.png",
        "figure2.svg",
        "verification.json",
        "maps.npz",
        "trajectory.npz",
        "training_config.json",
        "replay-verification.json",
    ):
        if (output_dir / name).is_symlink():
            raise SystemExit(f"Output file is a symbolic link: {output_dir / name}")
    for entry in load_json(MANIFEST_PATH)["files"]:
        source = (ROOT / entry["path"]).resolve()
        if source == output_dir or source.is_relative_to(output_dir):
            raise SystemExit(
                f"Output directory contains a package input: {entry['path']}"
            )


def verify_learning_maps(path: Path) -> dict:
    """Require new training checkpoints to be the selected figure's exact run."""
    import numpy as np

    with np.load(path, allow_pickle=False) as learned, np.load(
        ROOT / "fit_sources/hva_maps.npz", allow_pickle=False
    ) as reference, np.load(
        ROOT / "data/figure2_data.npz", allow_pickle=False
    ) as figure:
        for key in reference.files:
            if key not in learned or not np.array_equal(learned[key], reference[key]):
                raise SystemExit(f"Learning checkpoint mismatch: {key} in {path}")
        endpoint = figure["rf_surround"][:, 1:].transpose(1, 0, 2).reshape(-1, 2)
        if not np.array_equal(learned["final"], endpoint):
            raise SystemExit("Learned endpoint does not match the figure arrays.")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "checkpoints_exact": True,
    }


def reconstruct(output_dir: Path, learning_maps: Path | None = None) -> dict:
    validate_output_dir(output_dir)
    recipe = load_json(RECIPE_PATH)
    verify_package()
    versions = verify_environment(recipe)
    learning = (
        verify_learning_maps(learning_maps) if learning_maps is not None else None
    )
    reference = ROOT / recipe["target"]["reference_pdf"]
    expected_pdf = recipe["target"]["sha256"]
    if sha256(reference) != expected_pdf:
        raise SystemExit("The committed reference PDF does not match the recipe hash.")

    with tempfile.TemporaryDirectory(prefix="figure2-exact-") as temporary:
        work_root = Path(temporary) / "package"
        # Copy only frozen package inputs, excluding environments and previous outputs.
        for entry in load_json(MANIFEST_PATH)["files"]:
            destination = work_root / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / entry["path"], destination)
        clean_generated_outputs(work_root)
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        environment["MPLCONFIGDIR"] = str(work_root / "mpl_cache")
        for name in recipe["render_pipeline"]["entrypoints"]:
            subprocess.run(
                [sys.executable, str(work_root / "ribbon_local" / name)],
                cwd=work_root,
                env=environment,
                check=True,
            )

        generated_root = work_root / "figure2_compositions" / "clean_export"
        generated_pdf = generated_root / "figure2_clean.pdf"
        generated_png = generated_root / "figure2_clean.png"
        generated_svg = generated_root / "figure2_clean.svg"
        actual_pdf = sha256(generated_pdf)
        if actual_pdf != expected_pdf:
            raise SystemExit(
                "Reconstruction completed but the PDF is not byte-exact:\n"
                f"  expected {expected_pdf}\n  actual   {actual_pdf}"
            )
        expected_png = recipe["render_pipeline"]["expected_png_sha256"]
        actual_png = sha256(generated_png)
        if actual_png != expected_png:
            raise SystemExit(
                "PDF matched, but the rendered PNG did not match the frozen PNG hash:\n"
                f"  expected {expected_png}\n  actual   {actual_png}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_pdf, output_dir / "figure2.pdf")
        shutil.copy2(generated_png, output_dir / "figure2.png")
        shutil.copy2(generated_svg, output_dir / "figure2.svg")

    result = {
        "status": "PASS",
        "byte_exact_pdf": True,
        "seed": recipe["learning_pipeline"]["seed"],
        "run_id": recipe["learning_pipeline"]["run_id"],
        "reference_pdf": recipe["target"]["reference_pdf"],
        "reconstructed_pdf": str((output_dir / "figure2.pdf").resolve()),
        "pdf_sha256": expected_pdf,
        "byte_exact_png": True,
        "png_sha256": recipe["render_pipeline"]["expected_png_sha256"],
        "svg_byte_exactness_required": False,
        "environment": {"python": sys.version.split()[0], **versions},
        "verified_learning_maps": learning,
    }
    (output_dir / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct the committed Figure 2 and require an exact PDF hash."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reconstruction",
        help="Destination for newly reconstructed outputs (default: package/reconstruction).",
    )
    parser.add_argument(
        "--show-recipe",
        action="store_true",
        help="Print the frozen seed, learning schedule, and rendering recipe without rendering.",
    )
    parser.add_argument(
        "--learning-maps",
        type=Path,
        help="Verify checkpoints from replay_learning.py before rendering the frozen figure.",
    )
    args = parser.parse_args()
    recipe = load_json(RECIPE_PATH)
    if args.show_recipe:
        print(json.dumps(recipe, indent=2))
        return
    print(
        f"Figure 2 exact reconstruction: seed={recipe['learning_pipeline']['seed']} "
        f"run={recipe['learning_pipeline']['run_id']}",
        flush=True,
    )
    print(
        "Learning: b1={b1}, b2={b2}, b3={b3}, steps={steps}, eta0={eta0}, "
        "momentum={momentum}".format(
            **recipe["learning_pipeline"]["parameters"],
            **recipe["learning_pipeline"]["schedule"],
        ),
        flush=True,
    )
    result = reconstruct(args.output_dir.resolve(), args.learning_maps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
