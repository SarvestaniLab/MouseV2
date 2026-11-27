"""Recompute Figure 2 analysis from fit endpoints and optionally render it.

Defaults explicitly reuse the archived V1/HVA endpoints, but recompute every
partition, injection membership, field, and registration array. Pass --v1-fit
and --hva-maps to use newly trained fits. This is numerical reproduction;
reproduce.py remains the separate byte-exact archived-figure check.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from reproduce import (
    ROOT,
    load_json,
    sha256,
    validate_output_dir,
    verify_environment,
    verify_package,
)


def compare(value, reference):
    import numpy as np

    if value.shape != reference.shape:
        return {"shape_matches": False, "exact": False, "within_1e_9": False}
    finite = np.isfinite(value) & np.isfinite(reference)
    error = (
        float(
            np.max(abs(value[finite].astype(float) - reference[finite].astype(float)))
        )
        if finite.any()
        else 0.0
    )
    return dict(
        exact=bool(np.array_equal(value, reference, equal_nan=True)),
        max_abs=error,
        within_1e_9=bool(
            np.allclose(value, reference, atol=1e-9, rtol=0, equal_nan=True)
        ),
    )


def build(output_dir, v1_fit=None, hva_maps=None, render=False):
    validate_output_dir(output_dir)
    if output_dir.exists():
        raise SystemExit(
            "Choose a new output directory; existing results are preserved."
        )
    verify_package()
    recipe = load_json(ROOT / "REPRODUCTION_RECIPE.json")
    versions = verify_environment(recipe)
    import numpy as np
    import matplotlib.tri as mtri
    from analysis.geometry import (
        triangles,
        interpolate_sheet,
        fields,
        smooth_field_sign,
        inverse_map,
        register_ribbon,
        smooth_ribbon,
    )
    from analysis.partition import normalize_visual, garrett_area_parcellation

    reference = np.load(ROOT / "data/figure2_data.npz", allow_pickle=False)
    inputs = np.load(ROOT / "training/training_inputs.npz", allow_pickle=False)
    meta = load_json(ROOT / "data/figure2_shared_model.json")
    if v1_fit is None:
        cortex_v1, rf_v1 = reference["cortex_v1"], reference["rf_v1"]
    else:
        with np.load(v1_fit, allow_pickle=False) as fit:
            cortex_v1 = fit["cortical_grid"] * [1, -1]
            rf_v1 = fit["visual_grid"]
            np.testing.assert_array_equal(fit["targets"], inputs["targets"])
    if cortex_v1.shape != (41, 41, 2) or rf_v1.shape != cortex_v1.shape:
        raise ValueError("V1 fit must contain 41 x 41 cortical and visual grids.")
    if not np.isfinite(cortex_v1).all() or not np.isfinite(rf_v1).all():
        raise ValueError("Nonfinite V1 coordinates")
    # Validate the fixed RF border by matching cortical perimeter positions.
    boundary_mask = np.ones((41, 41), bool)
    boundary_mask[1:-1, 1:-1] = False
    border = inputs["surface"][:, 0] * [1, -1]
    boundary_indices = np.argmin(
        np.sum((cortex_v1[boundary_mask, None] - border[None]) ** 2, axis=-1), axis=1
    )
    np.testing.assert_allclose(
        cortex_v1[boundary_mask], border[boundary_indices], atol=1e-10, rtol=0
    )
    np.testing.assert_array_equal(
        rf_v1[boundary_mask], inputs["boundary"][boundary_indices]
    )
    hva_path = hva_maps or ROOT / "fit_sources/hva_maps.npz"
    with np.load(hva_path, allow_pickle=False) as fit:
        final = fit["final"]
    if final.shape != (2560, 2) or not np.isfinite(final).all():
        raise ValueError("HVA final checkpoint must contain 2560 finite RF pairs.")
    rf_surround = np.concatenate(
        (inputs["boundary"][:, None], final.reshape(16, 160, 2).transpose(1, 0, 2)),
        axis=1,
    )
    cortex_surround = inputs["surface"] * [1, -1]
    arrays = dict(
        cortex_v1=cortex_v1,
        rf_v1=rf_v1,
        cortex_surround=cortex_surround,
        rf_surround=rf_surround,
    )
    units = meta["partition_parameters"]["surface_units_per_mm"]
    print("Rebuilding fields and segmentation…", flush=True)
    for name, cyclic in (("v1", False), ("surround", True)):
        cortex = interpolate_sheet(arrays["cortex_" + name], cyclic=cyclic, linear=True)
        rf = interpolate_sheet(arrays["rf_" + name], cyclic=cyclic)
        raw = fields(rf, cortex, cyclic)[2]
        arrays["panel_b_cortex_" + name] = cortex
        arrays["panel_b_rf_" + name] = rf
        arrays["field_sign_" + name + "_raw"] = raw
        arrays["field_sign_" + name + "_display"] = smooth_field_sign(
            raw, cortex, triangles(rf.shape[:2], cyclic), units, sigma_mm=0.075
        )
    partition = garrett_area_parcellation(
        normalize_visual(rf_surround, np.isfinite(rf_surround[:, 1:]).all(2)),
        surface=cortex_surround,
        **meta["partition_parameters"],
    )
    arrays["labels"] = partition["labels"]
    sites = np.asarray(meta["injection_centers"])
    inverse, hits, membership, ambiguous = inverse_map(
        rf_surround.reshape(-1, 2),
        rf_v1,
        cortex_v1,
        triangles(rf_v1.shape[:2]),
        sites,
        meta["injection_radius_mm"] * units,
    )
    arrays.update(
        inverse_v1=inverse, inverse_triangle_hits=hits, injection_membership=membership
    )
    # Sites are prescribed cortical locations; RF labels follow the chosen V1 fit.
    tri = mtri.Triangulation(*cortex_v1.reshape(-1, 2).T, triangles(rf_v1.shape[:2]))
    site_rf = np.column_stack(
        [
            mtri.LinearTriInterpolator(tri, rf_v1[..., k].ravel())(*sites.T).filled(
                np.nan
            )
            for k in range(2)
        ]
    )
    if not np.isfinite(site_rf).all():
        raise ValueError("Injection sites fall outside the V1 fit")
    meta.update(
        injection_rf=site_rf.tolist(),
        parcels=len(partition["centers"]),
        parcel_signs=partition["centers"][:, 2].tolist(),
        target_counts=membership.reshape(160, 17, -1)[:, 1:].sum((0, 1)).tolist(),
        inverse_missing=int((hits == 0).sum()),
        inverse_multiple_triangles=int((hits > 1).sum()),
        inverse_ambiguous_distinct_preimages=int(ambiguous.sum()),
    )
    print("Rebuilding inverse registration…", flush=True)
    arrays.update(register_ribbon(rf_surround, cortex_surround, cortex_v1, rf_v1))
    light = smooth_ribbon(arrays["panel_c_registered"], arrays["panel_c_height"])
    checks = {key: compare(value, reference[key]) for key, value in arrays.items()}
    frozen_partition = np.load(ROOT / "data/panel_B_partition_field.npz")
    partition_arrays = {
        key: partition[key] for key in ("labels", "field_sign", "raw_field_sign")
    }
    partition_arrays["cortex"] = cortex_surround
    partition_checks = {
        key: compare(value, frozen_partition[key])
        for key, value in partition_arrays.items()
    }
    light_check = compare(
        light, np.load(ROOT / "ribbon_local/ribbon_smoothing_variants.npz")["light"]
    )
    numerical_match = all(
        c["within_1e_9"]
        for c in [*checks.values(), *partition_checks.values(), light_check]
    )
    if v1_fit is None and hva_maps is None and not numerical_match:
        raise ValueError(
            "Analysis from reference fits failed the 1e-9 numerical regression check."
        )
    receipt = dict(
        status="COMPLETED",
        reference_arrays_match=numerical_match,
        byte_exact_pdf_required=False,
        fits={
            "v1": {
                "path": str(v1_fit or ROOT / "data/figure2_data.npz"),
                "sha256": sha256(v1_fit or ROOT / "data/figure2_data.npz"),
                "default_archive": v1_fit is None,
            },
            "hva": {
                "path": str(hva_path),
                "sha256": sha256(hva_path),
                "default_archive": hva_maps is None,
            },
        },
        array_checks=checks,
        partition_checks=partition_checks,
        ribbon_check=light_check,
        environment={"python": sys.version.split()[0], **versions},
    )
    # Stage the entire build so failures leave no partially published output.
    with tempfile.TemporaryDirectory(prefix="figure2-methods-") as temporary:
        work = Path(temporary)
        (work / "data").mkdir()
        (work / "ribbon_local").mkdir()
        np.savez_compressed(work / "data/figure2_data.npz", **arrays)
        np.savez_compressed(
            work / "data/panel_B_partition_field.npz", **partition_arrays
        )
        (work / "data/figure2_shared_model.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )
        np.savez_compressed(
            work / "ribbon_local/ribbon_smoothing_variants.npz", light=light
        )
        if render:
            for path in (ROOT / "ribbon_local").glob("*.py"):
                shutil.copy2(path, work / "ribbon_local" / path.name)
            shutil.copy2(
                ROOT / "ribbon_local/cumulative_height_audit.json",
                work / "ribbon_local/cumulative_height_audit.json",
            )
            environment = dict(
                os.environ, PYTHONHASHSEED="0", MPLCONFIGDIR=str(work / "mpl_cache")
            )
            for entry in recipe["render_pipeline"]["entrypoints"]:
                subprocess.run(
                    [sys.executable, str(work / "ribbon_local" / entry)],
                    cwd=work,
                    env=environment,
                    check=True,
                )
            generated = work / "figure2_compositions/clean_export"
            receipt["pdf_sha256"] = sha256(generated / "figure2_clean.pdf")
        output_dir.mkdir(parents=True)
        shutil.copytree(work / "data", output_dir / "data")
        shutil.copy2(
            work / "ribbon_local/ribbon_smoothing_variants.npz",
            output_dir / "ribbon_smoothing_variants.npz",
        )
        if render:
            for suffix in ("pdf", "png", "svg"):
                shutil.copy2(
                    generated / f"figure2_clean.{suffix}",
                    output_dir / f"figure2.{suffix}",
                )
        (output_dir / "verification.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
    print(
        f"Completed: {output_dir}\nReference arrays match within 1e-9: {numerical_match}"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rebuilt")
    parser.add_argument(
        "--v1-fit",
        type=Path,
        help="fit.npz from training/v1.py; default: archived V1 endpoint",
    )
    parser.add_argument(
        "--hva-maps",
        type=Path,
        help="maps.npz from replay_learning.py; default: archived HVA endpoint",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render PDF, PNG, SVG after rebuilding data",
    )
    args = parser.parse_args()
    build(args.output_dir.resolve(), args.v1_fit, args.hva_maps, args.render)


if __name__ == "__main__":
    main()
