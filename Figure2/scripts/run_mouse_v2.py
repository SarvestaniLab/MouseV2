"""Run the canonical mouse V2 model, optionally over several beta1 values."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import tensorflow.compat.v1 as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "elastictopy"))

from elastic_net_mouse import (
    DEFAULT_MOUSE_BETA1,
    DEFAULT_MOUSE_BETA2,
    DEFAULT_MOUSE_BETA3,
    optimize,
)
from legacy_plotting import matlab_polar_maps, prepend_v1_boundary, reshape_model_map
from mouse_surface import (
    DEFAULT_BORDER_SVG,
    generate_mouse_surface_grid,
    resample_closed_values,
)
from vf import (
    LANDMARK_ALTITUDE_OFFSET_DEG,
    generate_polygon_landmarks,
    points_inside_polygon,
    required_spiral_radius,
)


DEFAULT_BETA1 = (DEFAULT_MOUSE_BETA1,)
# Provisional pooled Waters HVA primitive-density fit at center (0°, +20°).
DEFAULT_MAGNIFICATION = -0.05
DEFAULT_BOUNDARY_RETINOTOPY = "synthetic-rebalanced"
DEFAULT_MAGNIFICATION_EXPONENT = DEFAULT_MAGNIFICATION
REBALANCED_SYNTHETIC_BOUNDARY = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "MouseV1_synthetic_boundary_Waters_row0_anchored_80.csv"
)
SUPPORTED_SYNTHETIC_BOUNDARY = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "MouseV1_synthetic_boundary_Waters_supported_80.csv"
)
WATERS_RETINOTOPY_PATH = (
    PROJECT_ROOT
    / "data"
    / "waters2019"
    / "derived"
    / "waters_v1_border_retinotopy.npz"
)


def final_checkpoint(run_dir):
    paths = list(Path(run_dir).glob("y-*.data"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    return max(paths, key=lambda path: int(path.stem.rsplit("-", 1)[-1]))


def load_boundary(mode, map_h):
    if mode in ("synthetic-rebalanced", "synthetic-waters-supported"):
        path = (
            REBALANCED_SYNTHETIC_BOUNDARY
            if mode == "synthetic-rebalanced"
            else SUPPORTED_SYNTHETIC_BOUNDARY
        )
        table = np.genfromtxt(path, delimiter=",", names=True)
        values = np.column_stack([table["azimuth_deg"], table["elevation_deg"]])
        # The CSV contains 80 distinct cyclic rows, whereas the Waters NPZ
        # includes a repeated endpoint. Close only for periodic interpolation.
        closed = np.vstack([values, values[0]])
        source = np.linspace(0.0, 1.0, len(closed))
        target = np.linspace(0.0, 1.0, map_h, endpoint=False)
        return np.column_stack([
            np.interp(target, source, closed[:, axis])
            for axis in range(closed.shape[1])
        ])
    with np.load(WATERS_RETINOTOPY_PATH) as arrays:
        if mode == "waters-raw":
            azimuth = arrays["model_azimuth_raw_deg"].copy()
            altitude = arrays["model_altitude_raw_deg"].copy()
        elif mode == "waters-extended":
            azimuth = arrays["model_azimuth_extended_deg"].copy()
            altitude = arrays["model_altitude_extended_deg"].copy()
        else:
            raise ValueError(f"Unknown boundary-retinotopy mode: {mode}")
    return resample_closed_values(np.column_stack([azimuth, altitude]), map_h)


def boundary_asset_path(mode):
    """Return the checked-in source asset for a file-backed boundary mode."""
    if mode == "synthetic-rebalanced":
        return REBALANCED_SYNTHETIC_BOUNDARY
    if mode == "synthetic-waters-supported":
        return SUPPORTED_SYNTHETIC_BOUNDARY
    if mode in ("waters-raw", "waters-extended"):
        return WATERS_RETINOTOPY_PATH
    raise ValueError(f"Boundary mode has no file-backed asset: {mode}")


def make_shared_inputs(
    seed,
    map_h,
    map_w,
    prototype_count,
    boundary_retinotopy=DEFAULT_BOUNDARY_RETINOTOPY,
    magnification=DEFAULT_MAGNIFICATION,
):
    np.random.seed(seed)
    boundary = load_boundary(boundary_retinotopy, map_h)
    radial_limit = max(
        95.0,
        float(np.ceil(required_spiral_radius(boundary))),
    )
    visual_field = generate_polygon_landmarks(
        magnification, 0.05, radial_limit, prototype_count, boundary
    )
    np.random.seed(seed)
    indices = np.random.randint(visual_field.shape[0], size=map_h * map_w)
    noise = np.random.normal(0.0, 1.0, (map_h * map_w, 2))
    initial_map = visual_field[indices] + noise
    return visual_field, boundary, initial_map


def evaluate_run(checkpoint, visual_field, boundary, map_h, sigma):
    retinotopy = reshape_model_map(np.loadtxt(checkpoint, delimiter=","), map_h)
    full = prepend_v1_boundary(retinotopy, boundary)
    maps = matlab_polar_maps(full, sigma=sigma)

    along = np.linalg.norm(np.diff(retinotopy, axis=0), axis=2).ravel()
    across = np.linalg.norm(np.diff(retinotopy, axis=1), axis=2).ravel()
    neighbor_distances = np.concatenate((along, across))
    along_rms = float(np.sqrt(np.mean(along**2)))
    outward_rms = float(np.sqrt(np.mean(across**2)))
    coverage = cKDTree(retinotopy.reshape(-1, 2)).query(visual_field)[0]
    boundary_error = np.linalg.norm(retinotopy[:, 0] - boundary, axis=1)

    field_sign = maps["field_sign_blurred"]
    horizontal_crossings = field_sign[:, :-1] * field_sign[:, 1:] < 0
    vertical_crossings = field_sign[:-1] * field_sign[1:] < 0
    crossing_fraction = (
        horizontal_crossings.sum() + vertical_crossings.sum()
    ) / (horizontal_crossings.size + vertical_crossings.size)
    vector_along = full[1:, :-1] - full[:-1, :-1]
    vector_across = full[:-1, 1:] - full[:-1, :-1]
    cell_cross = (
        vector_along[:, :, 0] * vector_across[:, :, 1]
        - vector_along[:, :, 1] * vector_across[:, :, 0]
    )
    trained_inside = points_inside_polygon(retinotopy.reshape(-1, 2), boundary)
    prototype_inside = points_inside_polygon(visual_field, boundary)
    return {
        "altitude_min": float(retinotopy[:, :, 1].min()),
        "altitude_max": float(retinotopy[:, :, 1].max()),
        "boundary_rmse_deg": float(np.sqrt(np.mean(boundary_error**2))),
        "coverage_mean_deg": float(coverage.mean()),
        "coverage_p95_deg": float(np.percentile(coverage, 95)),
        "field_sign_crossing_fraction": float(crossing_fraction),
        "field_sign_positive_fraction": float(np.mean(field_sign > 0)),
        "cell_orientation_negative_fraction": float(np.mean(cell_cross < 0)),
        "cell_orientation_near_zero_fraction": float(
            np.mean(np.abs(cell_cross) < 1.0)
        ),
        "outer_seam_gap_deg": float(np.linalg.norm(full[0, -1] - full[-1, -1])),
        "trained_node_outside_boundary_fraction": float(np.mean(~trained_inside)),
        "prototype_outside_boundary_fraction": float(np.mean(~prototype_inside)),
        "along_neighbor_p95_deg": float(np.percentile(along, 95)),
        "outward_neighbor_p95_deg": float(np.percentile(across, 95)),
        "along_neighbor_rms_deg": along_rms,
        "outward_neighbor_rms_deg": outward_rms,
        "directional_neighbor_rms_log_ratio": float(
            abs(np.log(along_rms / outward_rms))
        ),
        "neighbor_mean_deg": float(neighbor_distances.mean()),
        "neighbor_rms_deg": float(np.sqrt(np.mean(neighbor_distances**2))),
        "retinotopy": retinotopy,
        "field_sign": field_sign,
    }


def save_summary_figure(records, surface, output, beta2, beta3):
    count = len(records)
    map_columns = min(3, count)
    map_rows = int(np.ceil(count / map_columns))
    figure = plt.figure(
        figsize=(4.0 * map_columns, 3.0 * map_rows + 6.0),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(2, 1, height_ratios=(map_rows, 2.0))
    map_grid = grid[0].subgridspec(map_rows, map_columns)
    for column, record in enumerate(records):
        axis = figure.add_subplot(
            map_grid[column // map_columns, column % map_columns]
        )
        axis.pcolormesh(
            surface[:, :, 0],
            -surface[:, :, 1],
            record["field_sign"],
            shading="gouraud",
            cmap="RdBu",
            vmin=-1,
            vmax=1,
            rasterized=True,
        )
        axis.set_title(f"β1={record['beta1']:.2g}")
        axis.set_aspect("equal")
        axis.set_axis_off()

    beta1 = np.asarray([record["beta1"] for record in records])
    metric_specs = (
        ("neighbor_rms_deg", "Neighbor RMS", "degrees"),
        ("coverage_p95_deg", "Prototype coverage p95", "degrees"),
        ("field_sign_crossing_fraction", "Field-sign crossings", "fraction"),
        ("cell_orientation_near_zero_fraction", "Near-zero RF cells", "fraction"),
        ("outer_seam_gap_deg", "Outer seam gap", "degrees"),
        ("boundary_rmse_deg", "Boundary RMSE", "degrees"),
    )
    metric_grid = grid[1].subgridspec(2, 3)
    for column, (name, title, ylabel) in enumerate(metric_specs):
        axis = figure.add_subplot(metric_grid[column // 3, column % 3])
        values = [record[name] for record in records]
        axis.plot(beta1, values, "o-", color="#345995")
        axis.set(title=title, xlabel="β1", ylabel=ylabel)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        axis.grid(color="0.9", linewidth=0.6)
    figure.suptitle(
        f"Mouse V2 model — fixed β2={beta2:g}, β3={beta3:g}"
    )
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--beta1", type=float, nargs="+", default=DEFAULT_BETA1)
    parser.add_argument("--beta2", type=float, default=DEFAULT_MOUSE_BETA2)
    parser.add_argument("--beta3", type=float, default=DEFAULT_MOUSE_BETA3)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="Checkpoint interval; use a positive value to retain learning frames",
    )
    parser.add_argument("--map-h", type=int, default=80)
    parser.add_argument("--map-w", type=int, default=8)
    parser.add_argument("--prototype-count", type=int, default=2000)
    parser.add_argument(
        "--magnification",
        type=float,
        default=DEFAULT_MAGNIFICATION,
        help="Radial-density parameter c for the spiral visual-field prototypes",
    )
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument(
        "--surface-svg", type=Path, default=DEFAULT_BORDER_SVG,
        help=("Cortical inner/outer border SVG (defaults to Waters individual-V1 median); "
              "use data/waters2019/derived/WatersV1_RegisteredMouseMedian_borders.svg "
              "for the common-origin candidate, or "
              "data/waters2019/derived/WatersV1_IndividualV1Median_borders.svg "
              "for the individual-V1 thickness candidate"),
    )
    parser.add_argument(
        "--boundary-retinotopy",
        choices=tuple(mode for mode in (
            "synthetic-rebalanced", "synthetic-waters-supported", "waters-raw", "waters-extended"
        ) if boundary_asset_path(mode).is_file()),
        default=DEFAULT_BOUNDARY_RETINOTOPY,
        help=(
            "RF coordinates constraining the V1 border; default uses the "
            "Wagor-calibrated, row-40/row-0 rebalanced synthetic boundary"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "mouse_v2",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tf.disable_v2_behavior()
    output_root = args.output_root.resolve() / f"seed-{args.seed:03d}"
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty sweep: {output_root}")
    output_root.mkdir(parents=True)

    visual_field, boundary, initial_map = make_shared_inputs(
        args.seed,
        args.map_h,
        args.map_w,
        args.prototype_count,
        boundary_retinotopy=args.boundary_retinotopy,
        magnification=args.magnification,
    )
    np.savetxt(output_root / "vf.csv", visual_field, fmt="%.12g", delimiter=",")
    np.savetxt(output_root / "boundary.csv", boundary, fmt="%.12g", delimiter=",")
    np.savetxt(output_root / "init.csv", initial_map, fmt="%.12g", delimiter=",")

    records = []
    for beta1 in args.beta1:
        run_dir = output_root / f"b1-{beta1:.3e}"
        run_dir.mkdir()
        for name, values in (
            ("vf.csv", visual_field),
            ("boundary.csv", boundary),
            ("init.csv", initial_map),
        ):
            np.savetxt(run_dir / name, values, fmt="%.12g", delimiter=",")

        tf.reset_default_graph()
        tf.set_random_seed(args.seed)
        y = tf.get_variable("y", dtype=tf.float64, initializer=initial_map)
        started = time.time()
        optimize(
            visual_field,
            boundary,
            beta1,
            args.beta2,
            args.beta3,
            args.map_h,
            args.map_w,
            y,
            k0=30.0,
            kr=0.003,
            eta0=0.1,
            outdir=run_dir,
            save_interval=args.save_interval,
            n=args.iterations,
            quiet_save=True,
            report_iterations=False,
            log_iterations=False,
            surface_svg=args.surface_svg,
        )
        checkpoint = final_checkpoint(run_dir)
        metrics = evaluate_run(
            checkpoint, visual_field, boundary, args.map_h, args.sigma
        )
        record = {
            "beta1": float(beta1),
            "beta2": args.beta2,
            "beta3": args.beta3,
            "checkpoint": checkpoint.name,
            "elapsed_seconds": time.time() - started,
            **metrics,
        }
        records.append(record)
        serializable = {
            key: value
            for key, value in record.items()
            if key not in ("retinotopy", "field_sign")
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(serializable))

    metric_names = [
        key
        for key in records[0]
        if key not in ("retinotopy", "field_sign")
    ]
    with (output_root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_names)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in metric_names})

    surface = generate_mouse_surface_grid(
        svg_path=args.surface_svg, map_h=args.map_h, map_w=args.map_w
    )
    save_summary_figure(
        records,
        surface,
        output_root / "summary.png",
        args.beta2,
        args.beta3,
    )
    manifest = {
        "seed": args.seed,
        "beta1": list(args.beta1),
        "beta2": args.beta2,
        "beta3": args.beta3,
        "iterations": args.iterations,
        "save_interval": args.save_interval,
        "prototype_count": args.prototype_count,
        "magnification": args.magnification,
        "visual_field": (
            "spiral prototypes clipped to the row-rebalanced synthetic V1 boundary"
            if args.boundary_retinotopy == "synthetic-rebalanced"
            else "spiral prototypes clipped to the selected closed V1 RF boundary"
        ),
        "prototype_sampling_center_deg": [0.0, LANDMARK_ALTITUDE_OFFSET_DEG],
        "smoothness_mask": "inverse squared Euclidean cortical-node distance",
        "cyclic_along": True,
        "surface_svg": str(args.surface_svg.resolve()),
        "boundary_retinotopy": args.boundary_retinotopy,
    }
    (output_root / "sweep.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output_root / "summary.png")


if __name__ == "__main__":
    main()
