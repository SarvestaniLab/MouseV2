"""Warp the published Zhuang V1 retinotopy into Figure 2 model coordinates."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.io import loadmat
from scipy.ndimage import binary_erosion, label, map_coordinates
from scipy.spatial import cKDTree

from mouse_surface import load_svg_paths


def _transform(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return np.column_stack((points, np.ones(len(points)))) @ affine.T


def _ordered_mask_boundary(mask: np.ndarray) -> np.ndarray:
    figure, axis = plt.subplots()
    contour_set = axis.contour(mask.astype(float), levels=[0.5])
    paths = [path.vertices for path in contour_set.collections[0].get_paths()]
    result = max(paths, key=len)
    plt.close(figure)
    return result


def _resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if not np.allclose(points[0], points[-1]):
        points = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    return np.column_stack(
        [np.interp(targets, cumulative, points[:, axis]) for axis in range(2)]
    )


def build_warped_zhuang_v1(
    project_root: Path,
    surface: np.ndarray,
    boundary_rf: np.ndarray,
) -> dict[str, np.ndarray | float | str]:
    """Return Zhuang samples warped to the model V1 border and RF boundary."""
    data_root = (
        project_root
        / "MATLAB"
        / "HanBoninLabCode"
        / "Han 2022 NatComm data retinotopy"
    )
    # The curated package keeps only the three required arrays in a compact folder.
    compact_root = project_root / "data" / "zhuang2017"
    if compact_root.is_dir():
        data_root = compact_root
    azimuth = loadmat(
        data_root / "ZhuangFig9_panel_d_azimuth_interpolated.mat"
    )["panel_d_azimuth_interpolated"]
    elevation = loadmat(
        data_root / "ZhuangFig9_panel_c_altitudes_interpolated.mat"
    )["panel_c_altitudes_interpolated"]
    border_mask = loadmat(data_root / "ZhuangFig9_panel_d_border_mask.mat")[
        "panel_d_border_mask"
    ] > 0.5

    regions, count = label(~border_mask)
    candidates = []
    for index in range(1, count + 1):
        region = regions == index
        if (
            region[0].any()
            or region[-1].any()
            or region[:, 0].any()
            or region[:, -1].any()
        ):
            continue
        candidates.append((int(region.sum()), region))
    if not candidates:
        raise ValueError("Could not identify the closed Zhuang V1 mask")
    _, v1_mask = max(candidates, key=lambda item: item[0])

    edge = v1_mask & ~binary_erosion(v1_mask)
    rows, columns = np.nonzero(edge)
    source_border = np.column_stack((columns, rows)).astype(float)
    source_border = source_border[:: max(1, len(source_border) // 1500)]
    svg_border = load_svg_paths(
        project_root / "data" / "ZhuangTracing9_borders.svg", step=0.001
    )[1]
    svg_border = svg_border[:: max(1, len(svg_border) // 1500)]

    source_min, source_max = source_border.min(axis=0), source_border.max(axis=0)
    target_min, target_max = svg_border.min(axis=0), svg_border.max(axis=0)
    scale = (target_max - target_min) / (source_max - source_min)
    affine = np.array(
        [
            [scale[0], 0.0, target_min[0] - scale[0] * source_min[0]],
            [0.0, scale[1], target_min[1] - scale[1] * source_min[1]],
        ]
    )
    for _ in range(20):
        transformed = _transform(source_border, affine)
        distances, indices = cKDTree(svg_border).query(transformed)
        keep = distances <= np.percentile(distances, 85)
        design = np.column_stack(
            (source_border[keep], np.ones(np.count_nonzero(keep)))
        )
        solution, *_ = np.linalg.lstsq(
            design, svg_border[indices[keep]], rcond=None
        )
        affine = solution.T

    source_pixels = _resample_closed(_ordered_mask_boundary(v1_mask), len(surface))
    source_cortex = _transform(source_pixels, affine)
    source_cortex[:, 1] *= -1
    target_cortex = surface[:, 0]

    best_error = np.inf
    best_pixels = None
    best_cortex = None
    for reverse in (False, True):
        candidate_pixels = source_pixels[::-1] if reverse else source_pixels
        candidate_cortex = source_cortex[::-1] if reverse else source_cortex
        for shift in range(len(candidate_cortex)):
            shifted = np.roll(candidate_cortex, shift, axis=0)
            error = np.mean(np.sum((shifted - target_cortex) ** 2, axis=1))
            if error < best_error:
                best_error = error
                best_pixels = np.roll(candidate_pixels, shift, axis=0)
                best_cortex = shifted

    source_control_rf = np.column_stack(
        (
            map_coordinates(
                azimuth,
                [best_pixels[:, 1], best_pixels[:, 0]],
                order=1,
                mode="nearest",
            ),
            map_coordinates(
                elevation,
                [best_pixels[:, 1], best_pixels[:, 0]],
                order=1,
                mode="nearest",
            ),
        )
    )
    cortical_delta = RBFInterpolator(
        best_cortex,
        target_cortex - best_cortex,
        kernel="thin_plate_spline",
        smoothing=1e-6,
        degree=1,
    )
    retinotopy_delta = RBFInterpolator(
        best_cortex,
        boundary_rf - source_control_rf,
        kernel="thin_plate_spline",
        smoothing=1e-5,
        degree=1,
    )

    def warp(sample_rows: np.ndarray, sample_columns: np.ndarray) -> np.ndarray:
        source_xy = _transform(
            np.column_stack((sample_columns, sample_rows)), affine
        )
        source_xy[:, 1] *= -1
        source_rf = np.column_stack(
            (azimuth[sample_rows, sample_columns], elevation[sample_rows, sample_columns])
        )
        values = np.column_stack(
            (
                source_xy + cortical_delta(source_xy),
                source_rf + retinotopy_delta(source_xy),
            )
        )
        return values[np.all(np.isfinite(values), axis=1)]

    sample_mask = binary_erosion(v1_mask, iterations=2)
    dense_rows, dense_columns = np.nonzero(sample_mask)
    dense_keep = (dense_rows % 2 == 0) & (dense_columns % 2 == 0)
    sample_keep = (dense_rows % 4 == 0) & (dense_columns % 4 == 0)
    return {
        "dense": warp(dense_rows[dense_keep], dense_columns[dense_keep]),
        "samples": warp(dense_rows[sample_keep], dense_columns[sample_keep]),
        "source": (
            "Zhuang et al. 2017 Figure 9 C-D, warped to the Waters/model "
            "cortical and visual-field boundaries"
        ),
    }


__all__ = ["build_warped_zhuang_v1"]
