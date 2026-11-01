"""Create the compact Figure 3F/G input from the Han et al. MATLAB assets.

This is a Python port of the data-collection and affine-registration portions
of ``plot_all_cell_smooth_tuning_map_4.m``. It is intentionally separate from
the plotting script because most users should consume the checked-in compact
CSV rather than the large upstream MATLAB workspace.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, map_coordinates

from registration import LEGACY_PARAMETERS, unpack_affine


HERE = Path(__file__).resolve().parent

# The common-map image size used for each mouse before the second alignment.
COMMON_IMAGE_SIZE = {
    "BV011": 500, "BV017": 400, "BV023": 400, "BV024": 400,
    "BV026": 400, "EV011": 500, "EV020": 500, "EV025": 500,
    "XH215": 500, "XH218": 400,
}

# transform_factor_contour_rev after the historical -2 row / +66 column
# adjustment. Entries are (di, dj, inverse scale 1, inverse scale 2, angle).
MOUSE_TO_COMMON_TRANSFORM = {
    "BV011": (138.0, 252.0, 1 / 0.82, 1 / 0.80, 1.0),
    "BV017": (145.0, 402.0, 1 / 1.22, 1 / 1.22, -4.0),
    "BV023": (276.0, 365.0, 1 / 0.82, 1 / 0.90, 2.0),
    "BV024": (223.0, 330.0, 1 / 0.86, 1 / 0.87, -4.0),
    "BV026": (235.0, 320.0, 1 / 0.91, 1 / 0.96, 1.0),
    "EV011": (61.0, 245.0, 1 / 0.74, 1 / 0.84, 10.0),
    "EV020": (90.0, 206.0, 1 / 0.80, 1 / 0.87, 2.0),
    "EV025": (91.0, 272.0, 1 / 0.73, 1 / 0.85, 2.0),
    "XH215": (143.0, 266.0, 1 / 0.73, 1 / 0.87, 1.0),
    "XH218": (144.0, 241.0, 1 / 0.75, 1 / 0.74, -4.0),
}


def _matlab_numbers(text):
    return tuple(float(value) for value in re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text
    ))


def load_session_transforms(script_path: Path):
    """Parse the final transform assignment in each MATLAB case block."""

    text = script_path.read_text(encoding="utf-8", errors="replace")
    cases = list(re.finditer(r"case\s+'([^']+)'", text))
    transforms = {}
    for index, match in enumerate(cases):
        end = cases[index + 1].start() if index + 1 < len(cases) else len(text)
        block = text[match.end():end]
        assignments = re.findall(r"transform_factor\s*=\s*\[([^\]]+)\]", block)
        if not assignments:
            continue
        values = _matlab_numbers(assignments[-1])
        if len(values) != 5:
            raise ValueError(f"Could not parse transform for {match.group(1)}")
        transforms[match.group(1)] = values
    return transforms


def _stack(records, key, dtype=float):
    return np.asarray([record[key] for record in records], dtype=dtype)


def _transform_cells(roi, first_transform, second_transform, second_size):
    """Apply the two historical MATLAB coordinate transforms."""

    i0 = roi[:, 0]
    j0 = roi[:, 1]
    first_center_i = first_center_j = None
    # The source background need not be square, so the caller supplies its
    # shape by attaching it to the transform tuple.
    di, dj, scale_i, scale_j, angle, height, width = first_transform
    first_center_i = height / 2.0
    first_center_j = width / 2.0
    i2 = (i0 - first_center_i) * scale_i
    j2 = (j0 - first_center_j) * scale_j
    theta = np.deg2rad(angle)
    i3 = j2 * np.cos(theta) - i2 * np.sin(theta) + di
    j3 = j2 * np.sin(theta) + i2 * np.cos(theta) + dj

    di, dj, inverse_scale_1, inverse_scale_2, angle = second_transform
    # The MATLAB script deliberately swaps entries 3/4 at this point.
    scale_i = inverse_scale_2
    scale_j = inverse_scale_1
    i0 = j3
    j0 = i3
    center = second_size / 2.0
    i2 = (i0 - center) * scale_i
    j2 = (j0 - center) * scale_j
    theta = np.deg2rad(angle)
    common_i = j2 * np.cos(theta) - i2 * np.sin(theta) + di
    common_j = j2 * np.sin(theta) + i2 * np.cos(theta) + dj
    return common_i, common_j


def _warp_zhuang_map(values, output_shape, parameters=LEGACY_PARAMETERS):
    """Match ``imwarp(values, affine2d(T), OutputView=...)`` for the saved T."""

    # MATLAB affine2d uses [x y 1] * T. This is the final transform assigned
    # in the historical registration script.
    linear, translation = unpack_affine(parameters)
    inverse = np.linalg.inv(linear.T)
    out_rows, out_columns = output_shape
    out_y, out_x = np.meshgrid(
        np.arange(1.0, out_rows + 1.0),
        np.arange(1.0, out_columns + 1.0),
        indexing="ij",
    )
    source_xy = inverse @ np.vstack((
        (out_x - translation[0]).ravel(),
        (out_y - translation[1]).ravel(),
    ))
    coordinates = np.vstack((source_xy[1] - 1.0, source_xy[0] - 1.0))
    warped = map_coordinates(
        values, coordinates, order=3, mode="constant", cval=np.nan,
        # scipy's global spline prefilter propagates the source map's NaN
        # exterior through the whole image. MATLAB's cubic imwarp does not;
        # evaluating the local cubic kernel directly preserves that behavior.
        prefilter=False,
    )
    return warped.reshape(output_shape)


def load_registration_inputs(
    source_root: Path,
    zhuang_root: Path | None = None,
    include_registration_columns: bool = True,
):
    """Load common-map cell positions, Han retinotopy, and Zhuang maps."""

    retinotopy_root = (
        source_root / "visualize cell location and tuning"
        / "Han 2022 NatComm data retinotopy"
    )
    dataset = loadmat(
        retinotopy_root / "dataset_meta.mat", simplify_cells=True
    )["dataset"]
    records = dataset["data"]
    roi_sessions = loadmat(
        retinotopy_root / "s_roi_location_all_8areas.mat", simplify_cells=True
    )["s_roi_location_all"]
    mouse_maps = loadmat(
        retinotopy_root / "s_retinomaps.mat", simplify_cells=True
    )["s_retinomaps"]
    session_transforms = load_session_transforms(
        retinotopy_root / "code_dependency" / "data_script_spatial_organization.m"
    )
    measured_azimuth = measured_elevation = None
    if include_registration_columns:
        measured_retinotopy = loadmat(
            retinotopy_root / "Cellrets.mat", simplify_cells=True
        )
        measured_azimuth = np.asarray(
            measured_retinotopy["vec_az_circ"], dtype=float
        ).reshape(-1)
        # The historical comparison reverses the stored elevation sign. The
        # additional display-only factor of 0.8 is unnecessary because objective
        # fitting estimates a global linear value calibration for each dimension.
        measured_elevation = -np.asarray(
            measured_retinotopy["vec_el_circ"], dtype=float
        ).reshape(-1)

    # The published MATLAB structure uses NaN for some unassigned cells.
    area_id = _stack(records, "area_id", float)
    animal_id = _stack(records, "animal_id", float)
    reliable = _stack(records, "is_reliable", bool)
    session_id = _stack(records, "idx_session", float)
    roi = np.vstack([record["roi_location_scaled"] for record in records])
    parameters = np.vstack([record["para_fit"] for record in records])
    sf_octaves = np.log2(parameters[:, 1] / 0.02)
    tf_octaves = np.log2(parameters[:, 3] / 0.5)
    session_names = [str(value).strip() for value in dataset["session_name"]]
    roi_session_names = [str(value["animal"]).strip() for value in roi_sessions]

    collected = []
    for mouse_record in mouse_maps:
        mouse = str(mouse_record["animal"]).strip()
        for session in (name for name in session_names if mouse in name):
            roi_session_index = roi_session_names.index(session)
            selected = np.flatnonzero(
                reliable & (session_id == roi_session_index + 1)
                & (area_id >= 1) & (area_id <= 8)
            )
            if selected.size == 0:
                continue
            transform = session_transforms[session]
            background_shape = roi_sessions[roi_session_index]["fig_bk_scaled"].shape
            first = (*transform, *background_shape)
            common_i, common_j = _transform_cells(
                roi[selected], first, MOUSE_TO_COMMON_TRANSFORM[mouse],
                COMMON_IMAGE_SIZE[mouse],
            )
            for offset, source_index in enumerate(selected):
                collected.append((
                    source_index, area_id[source_index], animal_id[source_index],
                    common_i[offset], common_j[offset],
                    (measured_azimuth[source_index]
                     if measured_azimuth is not None else np.nan),
                    (measured_elevation[source_index]
                     if measured_elevation is not None else np.nan),
                    sf_octaves[source_index], tf_octaves[source_index],
                ))

    aligned = pd.DataFrame.from_records(collected, columns=(
        "source_cell_index", "area_id", "animal_id", "common_i", "common_j",
        "han_1p_azimuth", "han_1p_elevation",
        "spatial_frequency_octaves", "temporal_frequency_octaves",
    ))

    retino_image = np.asarray(Image.open(
        retinotopy_root / "spatial organization" / "retino_map.png"
    ))
    output_shape = retino_image.shape[:2]
    zhuang_root = retinotopy_root if zhuang_root is None else zhuang_root
    altitude_source = loadmat(
        zhuang_root / "ZhuangFig9_panel_c_altitudes_interpolated.mat",
        simplify_cells=True,
    )["panel_c_altitudes_interpolated"]
    azimuth_source = loadmat(
        zhuang_root / "ZhuangFig9_panel_d_azimuth_interpolated.mat",
        simplify_cells=True,
    )["panel_d_azimuth_interpolated"]
    altitude_source = gaussian_filter(
        altitude_source.astype(float), sigma=3.0, mode="nearest", truncate=2.0
    )
    azimuth_source = gaussian_filter(
        azimuth_source.astype(float), sigma=3.0, mode="nearest", truncate=2.0
    )
    area_names = np.asarray(dataset["Area_name"], dtype=str)
    aligned["area"] = area_names[aligned["area_id"].to_numpy(dtype=int) - 1]
    return aligned, altitude_source, azimuth_source, output_shape


def extract(
    source_root: Path,
    zhuang_root: Path | None = None,
    registration_parameters=LEGACY_PARAMETERS,
    include_registration_columns: bool = False,
) -> pd.DataFrame:
    """Extract the compact cell table under a specified global affine."""

    aligned, altitude_source, azimuth_source, output_shape = (
        load_registration_inputs(
            source_root, zhuang_root, include_registration_columns
        )
    )
    altitude_map = _warp_zhuang_map(
        altitude_source, output_shape, registration_parameters
    )
    azimuth_map = _warp_zhuang_map(
        azimuth_source, output_shape, registration_parameters
    )

    # MATLAB round is half-away-from-zero; all valid coordinates are positive.
    rows = np.floor(aligned["common_i"].to_numpy() + 0.5).astype(int) - 1
    columns = np.floor(aligned["common_j"].to_numpy() + 0.5).astype(int) - 1
    valid = (
        (rows >= 0) & (rows < output_shape[0])
        & (columns >= 0) & (columns < output_shape[1])
    )
    elevation = np.full(len(aligned), np.nan)
    azimuth = np.full(len(aligned), np.nan)
    elevation[valid] = altitude_map[rows[valid], columns[valid]]
    azimuth[valid] = azimuth_map[rows[valid], columns[valid]]
    elevation = np.clip(elevation, -40.0, 40.0)
    azimuth = np.clip(azimuth, 0.0, 100.0)

    aligned["azimuth_deg"] = azimuth
    aligned["elevation_deg"] = elevation
    aligned["eccentricity_deg"] = np.hypot(azimuth, elevation - 20.0)
    columns = [
        "source_cell_index", "area", "animal_id", "azimuth_deg",
        "elevation_deg", "eccentricity_deg", "spatial_frequency_octaves",
        "temporal_frequency_octaves",
    ]
    if include_registration_columns:
        columns.extend((
            "common_i", "common_j", "han_1p_azimuth", "han_1p_elevation"
        ))
    return aligned[columns]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path, required=True,
        help="Directory containing dataset_meta.mat and the retinotopy asset folder.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "data" / "han2022_figure3fg.csv",
    )
    parser.add_argument(
        "--zhuang-root", type=Path,
        help=("Directory containing the two ZhuangFig9 interpolated MAT files. "
              "Defaults to the Han retinotopy asset directory."),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frame = extract(args.source_root, zhuang_root=args.zhuang_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, float_format="%.10g")
    print(f"Wrote {len(frame)} cells to {args.output}")
    print(f"Cells with finite registered retinotopy: {frame.eccentricity_deg.notna().sum()}")


if __name__ == "__main__":
    main()
