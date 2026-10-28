"""Analysis helpers ported from the final legacy MATLAB plotting workflow."""

from pathlib import Path
import json
import heapq

import numpy as np
from scipy.ndimage import (
    binary_closing, distance_transform_edt, gaussian_filter, label,
)
from scipy.cluster.vq import kmeans2
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from mouse_surface import generate_mouse_surface_grid


LEGACY_TRACE_UNITS_TO_MM = 0.5 / 1.4340


def load_checkpoint(run_dir, checkpoint=None):
    """Load a requested checkpoint, or the numerically latest checkpoint."""
    run_dir = Path(run_dir)
    if checkpoint is None:
        candidates = list(run_dir.glob("y-*.data"))
        if not candidates:
            raise FileNotFoundError(f"No y-*.data checkpoints found in {run_dir}")
        checkpoint_path = max(
            candidates, key=lambda path: int(path.stem.rsplit("-", 1)[-1])
        )
    else:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = run_dir / checkpoint_path
    return checkpoint_path, np.loadtxt(checkpoint_path, delimiter=",")


def list_checkpoints(run_dir):
    """Return checkpoints sorted by the final iteration number."""
    return sorted(
        Path(run_dir).glob("y-*.data"),
        key=lambda path: int(path.stem.rsplit("-", 1)[-1]),
    )


def reshape_model_map(values, map_h=80):
    """Convert the flattened model order to (along, across, coordinate)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected an N x 2 map, received {values.shape}")
    if values.shape[0] % map_h:
        raise ValueError(f"Map size {values.shape[0]} is not divisible by h={map_h}")
    map_w = values.shape[0] // map_h
    return values.reshape(map_w, map_h, 2).transpose(1, 0, 2)


def prepend_v1_boundary(retinotopy, boundary):
    """Add the V1/V2 border as column zero, matching the MATLAB arrays."""
    retinotopy = np.asarray(retinotopy, dtype=np.float64)
    boundary = np.asarray(boundary, dtype=np.float64)
    if retinotopy.ndim != 3 or retinotopy.shape[2] != 2:
        raise ValueError("retinotopy must have shape (along, across, 2)")
    if boundary.shape != (retinotopy.shape[0], 2):
        raise ValueError(
            f"boundary must have shape {(retinotopy.shape[0], 2)}, "
            f"received {boundary.shape}"
        )
    boundary = boundary.copy()
    boundary[:, 0] = np.maximum(boundary[:, 0], 0.0)
    return np.concatenate((boundary[:, None, :], retinotopy), axis=1)


def matlab_polar_maps(
    retinotopy, sigma=1.2, cyclic_along=True,
    v1_field_sign=-1.0, v1_border_field_sign=0.0,
):
    """Compute visual angle, eccentricity, and field sign as in MATLAB.

    MATLAB's ``gradient`` returns the column derivative first. NumPy returns
    axis derivatives in array order, so the names below make that conversion
    explicit before reproducing the original cross/dot-product expression.
    """
    retinotopy = np.asarray(retinotopy, dtype=np.float64)
    azimuth = np.maximum(retinotopy[:, :, 0], 0.0)
    altitude = retinotopy[:, :, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        # MATLAB used atand(y/x), including NaN at an exact (0, 0).
        visual_angle = np.degrees(np.arctan(altitude / azimuth))
    eccentricity = np.hypot(altitude, azimuth)

    if cyclic_along:
        dva_dalong = 0.5 * (
            np.roll(visual_angle, -1, axis=0) - np.roll(visual_angle, 1, axis=0)
        )
        decc_dalong = 0.5 * (
            np.roll(eccentricity, -1, axis=0) - np.roll(eccentricity, 1, axis=0)
        )
        dva_dacross = np.gradient(visual_angle, axis=1)
        decc_dacross = np.gradient(eccentricity, axis=1)
    else:
        dva_dalong, dva_dacross = np.gradient(visual_angle)
        decc_dalong, decc_dacross = np.gradient(eccentricity)
    numerator = dva_dalong * decc_dacross - dva_dacross * decc_dalong
    denominator = dva_dacross * decc_dacross + dva_dalong * decc_dalong
    field_angle = np.arctan2(numerator, denominator)
    field_sign = np.sin(field_angle)
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if v1_field_sign is not None:
        field_sign = field_sign.copy()
        field_sign[:, 0] = float(v1_border_field_sign)
        field_sign_blurred = v1_conditioned_field_sign_blur(
            field_sign, sigma=sigma, cyclic_along=cyclic_along,
            v1_field_sign=v1_field_sign,
            border_field_sign=v1_border_field_sign,
        )
    elif sigma == 0.0:
        field_sign_blurred = field_sign.copy()
    else:
        # imgaussfilt defaults: replicate padding and radius ceil(2*sigma).
        field_sign_blurred = matlab_gaussian_filter(
            field_sign, sigma, cyclic_along=cyclic_along
        )
    return {
        "azimuth": azimuth,
        "altitude": altitude,
        "visual_angle": visual_angle,
        "eccentricity": eccentricity,
        "field_sign": field_sign,
        "field_sign_blurred": field_sign_blurred,
    }


def v1_conditioned_field_sign_blur(
    field_sign, sigma=1.2, cyclic_along=True,
    v1_field_sign=-1.0, border_field_sign=0.0,
):
    """Blur an HVA field-sign sheet with a virtual negative-sign V1 interior.

    Column zero is the V1/HVA border and remains neutral. Virtual columns on
    its inward side fill the Gaussian support at ``v1_field_sign`` so values
    just outside V1 are smoothed against V1 rather than replicated edge data.
    """
    values = np.asarray(field_sign, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("field_sign must be a non-empty 2D sheet")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    conditioned = values.copy()
    conditioned[:, 0] = float(border_field_sign)
    if sigma == 0.0:
        return conditioned
    radius = int(np.ceil(2.0 * float(sigma)))
    interior = np.full(
        (values.shape[0], radius), float(v1_field_sign), dtype=np.float64,
    )
    extended = np.concatenate((interior, conditioned), axis=1)
    blurred = matlab_gaussian_filter(
        extended, sigma=sigma, cyclic_along=cyclic_along,
    )[:, radius:]
    blurred[:, 0] = float(border_field_sign)
    return blurred


def matlab_gaussian_filter(values, sigma=1.2, cyclic_along=False):
    """Match ``imgaussfilt`` replicate padding and default kernel radius."""
    values = np.asarray(values, dtype=np.float64)
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0.0:
        return values.copy()
    matlab_truncate = np.ceil(2.0 * sigma) / sigma
    mode = ("wrap", "nearest") if cyclic_along and values.ndim == 2 else "nearest"
    return gaussian_filter(values, sigma=sigma, mode=mode, truncate=matlab_truncate)


def synthetic_polar_surface(map_h=80, map_w=8):
    """Reproduce the beveled polar sheet used in MATLAB Figures 2, 6, and 8."""
    n = map_h - 1
    bevel = (n + 1) // 3
    bevel_offset = 0.05
    theta = np.linspace(0.0, 2.0 * np.pi * 5.75 / 6.0, n + 1)
    radii = np.linspace(3.0 / np.pi, 6.0 / np.pi, map_w + 1)
    theta_grid, radius_grid = np.meshgrid(theta, radii, indexing="ij")
    radius_grid = radius_grid.copy()
    for row in range(bevel):
        factor = (6.0 / np.pi - 3.0 / np.pi) * (
            (1.0 - bevel_offset + row + 1) / bevel
        )
        radius_grid[row] = 3.0 / np.pi + factor * (
            radius_grid[row] - 3.0 / np.pi
        )
    for row in range(n, n - bevel, -1):
        factor = (6.0 / np.pi - 3.0 / np.pi) * (
            (1.0 - bevel_offset + n + 1 - (row + 1)) / bevel
        )
        radius_grid[row] = 3.0 / np.pi + factor * (
            radius_grid[row] - 3.0 / np.pi
        )
    x = radius_grid * np.cos(theta_grid - np.pi / 4.0)
    y = radius_grid * np.sin(theta_grid - np.pi / 4.0)
    return np.stack((np.flipud(x), np.flipud(y)), axis=-1)


def cortical_magnification(
    retinotopy, boundary, surface=None, trace_units_to_mm=LEGACY_TRACE_UNITS_TO_MM
):
    """Compute the legacy local cortical-area/visual-area magnification."""
    retinotopy = np.asarray(retinotopy, dtype=np.float64)
    map_h, map_w, _ = retinotopy.shape
    if surface is None:
        surface = generate_mouse_surface_grid(map_h=map_h, map_w=map_w)
    full_retinotopy = prepend_v1_boundary(retinotopy, boundary)

    cortical_along = np.linalg.norm(
        np.roll(surface[:, :map_w], -1, axis=0) - surface[:, :map_w], axis=2
    )
    cortical_along_average = 0.5 * (
        np.roll(cortical_along, 1, axis=0) + cortical_along
    )
    cortical_across = np.linalg.norm(surface[:, 1:] - surface[:, :-1], axis=2)
    cortical_across_padded = np.column_stack(
        (cortical_across, cortical_across[:, -1])
    )
    cortical_across_average = 0.5 * (
        cortical_across_padded[:, :-1] + cortical_across_padded[:, 1:]
    )
    cortical_area = (
        cortical_along_average
        * cortical_across_average
        * trace_units_to_mm**2
    )

    visual_along = np.linalg.norm(
        np.roll(retinotopy, -1, axis=0) - retinotopy, axis=2
    )
    visual_along_average = 0.5 * (
        np.roll(visual_along, 1, axis=0) + visual_along
    )
    visual_across = np.linalg.norm(
        full_retinotopy[:, 1:] - full_retinotopy[:, :-1], axis=2
    )
    visual_across_padded = np.column_stack((visual_across, visual_across[:, -1]))
    visual_across_average = 0.5 * (
        visual_across_padded[:, :-1] + visual_across_padded[:, 1:]
    )
    visual_area = visual_along_average * visual_across_average
    with np.errstate(divide="ignore", invalid="ignore"):
        magnification = cortical_area / visual_area
    return {
        "cortical_area": cortical_area,
        "visual_area": visual_area,
        "magnification": magnification,
        "magnification_blurred": matlab_gaussian_filter(magnification, 1.2),
    }


def field_sign_parcellation(
    field_sign_blurred, cluster_count=7, random_seed=0,
    balanced_sign=False, fill_v1_border=False,
):
    """Reproduce the exploratory spatial field-sign parcellation.

    When ``fill_v1_border`` is true, column zero is excluded from the fit and
    inherits the label of column one.  This lets a neutral field-sign boundary
    remain zero for smoothing without creating an artificial positive-sign
    ring or leaving a gap between the fitted areas and V1.
    """
    field_sign_blurred = np.asarray(field_sign_blurred, dtype=np.float64)
    if fill_v1_border and field_sign_blurred.shape[1] < 2:
        raise ValueError("fill_v1_border requires at least two sheet columns")
    polar_surface = synthetic_polar_surface(
        map_h=field_sign_blurred.shape[0], map_w=field_sign_blurred.shape[1] - 1
    )
    threshold_sign = np.where(field_sign_blurred >= 0.0, 1.0, -1.0)
    spatial_features = polar_surface.reshape(-1, 2)
    fit_mask = np.ones(field_sign_blurred.shape, dtype=bool)
    if fill_v1_border:
        fit_mask[:, 0] = False
    fit_mask_flat = fit_mask.ravel()
    if balanced_sign:
        if cluster_count % 2:
            raise ValueError("balanced_sign requires an even cluster_count")
        clusters_per_sign = cluster_count // 2
        labels = np.empty(field_sign_blurred.size, dtype=np.int64)
        center_groups = []
        for group_index, sign_value in enumerate((1.0, -1.0)):
            sign_mask = (threshold_sign.ravel() == sign_value) & fit_mask_flat
            if np.count_nonzero(sign_mask) < clusters_per_sign:
                raise ValueError(
                    f"Not enough samples for {clusters_per_sign} clusters "
                    f"with field sign {sign_value:+.0f}"
                )
            spatial_centers, sign_labels = kmeans2(
                spatial_features[sign_mask], clusters_per_sign,
                iter=50, minit="++", seed=random_seed + group_index,
            )
            label_offset = group_index * clusters_per_sign
            labels[sign_mask] = sign_labels + label_offset
            center_groups.append(
                np.column_stack(
                    (spatial_centers, np.full(clusters_per_sign, sign_value))
                )
            )
        centers = np.vstack(center_groups)
    else:
        features = np.column_stack(
            (spatial_features, threshold_sign.ravel())
        )
        centers, fit_labels = kmeans2(
            features[fit_mask_flat], cluster_count,
            iter=50, minit="++", seed=random_seed,
        )
        labels = np.full(field_sign_blurred.size, -1, dtype=np.int64)
        labels[fit_mask_flat] = fit_labels
    labels = labels.reshape(field_sign_blurred.shape)
    if fill_v1_border:
        labels[:, 0] = labels[:, 1]
    return {
        "labels": labels,
        "centers": centers,
        "polar_surface": polar_surface,
        "threshold_sign": threshold_sign,
        "fill_v1_border": bool(fill_v1_border),
    }


def garrett_field_sign(
    retinotopy, sigma=3.0, cyclic_along=True, surface=None,
):
    """Compute Garrett et al.'s visual-field sign from azimuth and altitude.

    Unlike :func:`matlab_polar_maps`, this uses the gradients of the two
    retinotopic coordinate maps directly, matching Garrett et al. (2014),
    Eq. 1 and the released ``getAreaBorders.m`` implementation.
    """
    retinotopy = np.asarray(retinotopy, dtype=np.float64)
    if retinotopy.ndim != 3 or retinotopy.shape[2] != 2:
        raise ValueError("retinotopy must have shape (along, across, 2)")
    azimuth = retinotopy[:, :, 0]
    altitude = retinotopy[:, :, 1]
    if surface is not None:
        surface = np.asarray(surface, dtype=np.float64)
        if surface.shape != retinotopy.shape:
            raise ValueError("surface and retinotopy must have the same shape")
        azimuth_gradient = np.zeros_like(retinotopy)
        altitude_gradient = np.zeros_like(retinotopy)
        for row, column in np.ndindex(azimuth.shape):
            neighbors = list(_grid_neighbors(
                row, column, azimuth.shape, cyclic_along=cyclic_along
            ))
            displacement = np.asarray([
                surface[neighbor] - surface[row, column]
                for neighbor in neighbors
            ])
            azimuth_delta = np.asarray([
                azimuth[neighbor] - azimuth[row, column]
                for neighbor in neighbors
            ])
            altitude_delta = np.asarray([
                altitude[neighbor] - altitude[row, column]
                for neighbor in neighbors
            ])
            azimuth_gradient[row, column] = np.linalg.lstsq(
                displacement, azimuth_delta, rcond=None
            )[0]
            altitude_gradient[row, column] = np.linalg.lstsq(
                displacement, altitude_delta, rcond=None
            )[0]
        dazimuth_dalong = azimuth_gradient[:, :, 0]
        dazimuth_dacross = azimuth_gradient[:, :, 1]
        daltitude_dalong = altitude_gradient[:, :, 0]
        daltitude_dacross = altitude_gradient[:, :, 1]
    else:
        if cyclic_along:
            dazimuth_dalong = 0.5 * (
                np.roll(azimuth, -1, axis=0) - np.roll(azimuth, 1, axis=0)
            )
            daltitude_dalong = 0.5 * (
                np.roll(altitude, -1, axis=0) - np.roll(altitude, 1, axis=0)
            )
        else:
            dazimuth_dalong = np.gradient(azimuth, axis=0)
            daltitude_dalong = np.gradient(altitude, axis=0)
        dazimuth_dacross = np.gradient(azimuth, axis=1)
        daltitude_dacross = np.gradient(altitude, axis=1)
    cross = (
        dazimuth_dalong * daltitude_dacross
        - dazimuth_dacross * daltitude_dalong
    )
    dot = (
        dazimuth_dalong * daltitude_dalong
        + dazimuth_dacross * daltitude_dacross
    )
    field_sign = np.sin(np.arctan2(cross, dot))
    return matlab_gaussian_filter(
        field_sign, sigma=sigma, cyclic_along=cyclic_along
    )


def _cyclic_components(mask, cyclic_along=True):
    """Label 8-connected components, joining the first and last rows."""
    mask = np.asarray(mask, dtype=bool)
    structure = np.ones((3, 3), dtype=np.uint8)
    raw, count = label(mask, structure=structure)
    if cyclic_along:
        parents = np.arange(count + 1)
        def root(value):
            while parents[value] != value:
                parents[value] = parents[parents[value]]
                value = parents[value]
            return value
        for column in range(mask.shape[1]):
            for other in range(max(0, column - 1), min(mask.shape[1], column + 2)):
                first, last = raw[0, column], raw[-1, other]
                if first and last:
                    parents[root(last)] = root(first)
        raw = np.asarray([root(i) for i in range(count + 1)])[raw]
    output = np.full(mask.shape, -1, dtype=np.int64)
    for new_label, old_label in enumerate(np.unique(raw[mask])):
        output[raw == old_label] = new_label
    return output


def _fill_unassigned(labels, cyclic_along=True):
    """Extend patch cores through low-field-sign boundary pixels."""
    labels = np.asarray(labels, dtype=np.int64)
    if not np.any(labels >= 0):
        raise ValueError("No suprathreshold field-sign patches were found")
    if cyclic_along:
        tiled = np.concatenate((labels, labels, labels), axis=0)
        _, indices = distance_transform_edt(
            tiled < 0, return_distances=True, return_indices=True
        )
        nearest = tiled[tuple(indices)]
        return nearest[labels.shape[0]:2 * labels.shape[0]]
    _, indices = distance_transform_edt(
        labels < 0, return_distances=True, return_indices=True
    )
    return labels[tuple(indices)]


def _relabel(labels):
    output = np.full_like(labels, -1, dtype=np.int64)
    for new_label, old_label in enumerate(np.unique(labels[labels >= 0])):
        output[labels == old_label] = new_label
    return output


def _patch_signs(labels, field_sign):
    return np.asarray([
        np.sign(np.median(field_sign[labels == patch])) or 1.0
        for patch in np.unique(labels[labels >= 0])
    ])


def _triangle_visual_coverage_ratio(
    mask, azimuth, altitude, pixel_size=1.0, close_iterations=0,
    bounds=None, return_occupancy=False,
):
    """Estimate Garrett's integrated/unique visual-coverage ratio.

    The absolute areas of mapped grid triangles estimate integrated coverage;
    rasterizing those triangles in visual coordinates estimates their union.
    A folded (redundant) map therefore has a ratio greater than one.
    """
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    triangles = []
    for row in range(height):
        next_row = (row + 1) % height
        for column in range(width - 1):
            corners = (
                (row, column), (next_row, column),
                (next_row, column + 1), (row, column + 1),
            )
            for indices in ((0, 1, 2), (0, 2, 3)):
                nodes = [corners[index] for index in indices]
                if sum(mask[node] for node in nodes) < 2:
                    continue
                triangles.append(np.asarray([
                    (azimuth[node], altitude[node]) for node in nodes
                ], dtype=np.float64))
    if bounds is None:
        finite = np.isfinite(azimuth) & np.isfinite(altitude)
        x0 = np.floor(azimuth[finite].min() / pixel_size) * pixel_size
        y0 = np.floor(altitude[finite].min() / pixel_size) * pixel_size
        x1 = np.ceil(azimuth[finite].max() / pixel_size) * pixel_size
        y1 = np.ceil(altitude[finite].max() / pixel_size) * pixel_size
    else:
        x0, x1, y0, y1 = (float(value) for value in bounds)
    x_grid = np.arange(x0, x1 + 0.5 * pixel_size, pixel_size)
    y_grid = np.arange(y0, y1 + 0.5 * pixel_size, pixel_size)
    occupied = np.zeros((len(y_grid), len(x_grid)), dtype=bool)
    if not triangles:
        return (1.0, occupied) if return_occupancy else 1.0

    integrated_area = 0.0
    for triangle in triangles:
        edge1 = triangle[1] - triangle[0]
        edge2 = triangle[2] - triangle[0]
        twice_area = abs(np.cross(edge1, edge2))
        integrated_area += 0.5 * twice_area
        if twice_area <= 1e-12:
            continue
        x_indices = np.flatnonzero(
            (x_grid >= triangle[:, 0].min())
            & (x_grid <= triangle[:, 0].max())
        )
        y_indices = np.flatnonzero(
            (y_grid >= triangle[:, 1].min())
            & (y_grid <= triangle[:, 1].max())
        )
        xx, yy = np.meshgrid(x_grid[x_indices], y_grid[y_indices])
        points = np.column_stack((xx.ravel(), yy.ravel()))
        denominator = np.cross(edge1, edge2)
        relative = points - triangle[0]
        u = np.cross(relative, edge2) / denominator
        v = np.cross(edge1, relative) / denominator
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
        occupied[np.ix_(y_indices, x_indices)] |= inside.reshape(xx.shape)
    if close_iterations and close_iterations > 0:
        occupied = binary_closing(
            occupied, iterations=int(close_iterations), border_value=0,
        )
    unique_area = occupied.sum() * pixel_size**2
    if unique_area == 0.0:
        ratio = 1.0
    else:
        ratio = max(1.0, integrated_area / unique_area)
    return (ratio, occupied) if return_occupancy else ratio


def _visual_overlap_ratios(
    first_mask, second_mask, azimuth, altitude, pixel_size=0.5,
    close_iterations=15, bounds=None,
):
    """Return overlap divided by each patch's unique visual coverage."""
    _, first = _triangle_visual_coverage_ratio(
        first_mask, azimuth, altitude, pixel_size=pixel_size,
        close_iterations=close_iterations, bounds=bounds,
        return_occupancy=True,
    )
    _, second = _triangle_visual_coverage_ratio(
        second_mask, azimuth, altitude, pixel_size=pixel_size,
        close_iterations=close_iterations, bounds=bounds,
        return_occupancy=True,
    )
    overlap = float(np.count_nonzero(first & second))
    first_area = float(np.count_nonzero(first))
    second_area = float(np.count_nonzero(second))
    return (
        overlap / first_area if first_area else 0.0,
        overlap / second_area if second_area else 0.0,
    )


def _grid_neighbors(row, column, shape, cyclic_along=True):
    height, width = shape
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == column_offset == 0:
                continue
            new_row = row + row_offset
            if cyclic_along:
                new_row %= height
            elif not 0 <= new_row < height:
                continue
            new_column = column + column_offset
            if 0 <= new_column < width:
                yield new_row, new_column


def _mesh_geodesic_distances(surface, cyclic_along=True):
    """Return all-pairs shortest-path distances over the cortical mesh."""
    surface = np.asarray(surface, dtype=np.float64)
    shape = surface.shape[:2]
    node_count = int(np.prod(shape))
    rows = []
    columns = []
    weights = []
    for row, column in np.ndindex(shape):
        source = np.ravel_multi_index((row, column), shape)
        for neighbor in _grid_neighbors(
            row, column, shape, cyclic_along=cyclic_along
        ):
            target = np.ravel_multi_index(neighbor, shape)
            rows.append(source)
            columns.append(target)
            weights.append(float(np.linalg.norm(
                surface[neighbor] - surface[row, column]
            )))
    graph = csr_matrix((weights, (rows, columns)), shape=(node_count, node_count))
    return dijkstra(graph, directed=False)


def _geodesic_gaussian(values, distances, sigma):
    """Smooth nodal values with a normalized Gaussian of geodesic distance."""
    values = np.asarray(values, dtype=np.float64)
    if sigma is None or sigma <= 0.0:
        return values.copy()
    weights = np.exp(-0.5 * (distances / float(sigma)) ** 2)
    weights[distances > 3.0 * float(sigma)] = 0.0
    weights /= weights.sum(axis=1, keepdims=True)
    return (weights @ values.ravel()).reshape(values.shape)


def _geodesic_gaussian_with_v1(
    values, surface_mm, sigma, cyclic_along=True,
    v1_field_sign=-1.0, border_field_sign=0.0,
):
    """Geodesically blur a sheet after extending its inner edge into V1."""
    values = np.asarray(values, dtype=np.float64).copy()
    surface_mm = np.asarray(surface_mm, dtype=np.float64)
    if values.shape != surface_mm.shape[:2] or values.shape[1] < 2:
        raise ValueError("V1-conditioned smoothing requires a matching sheet")
    values[:, 0] = float(border_field_sign)
    if sigma is None or sigma <= 0.0:
        return values

    outward = surface_mm[:, 1] - surface_mm[:, 0]
    spacing = np.linalg.norm(outward, axis=1)
    if np.any(spacing <= 0.0):
        raise ValueError("V1 border and first HVA column must not overlap")
    unit_outward = outward / spacing[:, None]
    reference_spacing = float(np.median(spacing))
    interior_columns = max(1, int(np.ceil(3.0 * sigma / reference_spacing)))
    inward_distances = reference_spacing * np.arange(
        interior_columns, 0, -1, dtype=np.float64,
    )
    interior_surface = (
        surface_mm[:, None, 0, :]
        - unit_outward[:, None, :] * inward_distances[None, :, None]
    )
    extended_surface = np.concatenate((interior_surface, surface_mm), axis=1)
    interior_values = np.full(
        (values.shape[0], interior_columns), float(v1_field_sign),
        dtype=np.float64,
    )
    extended_values = np.concatenate((interior_values, values), axis=1)
    extended_distances = _mesh_geodesic_distances(
        extended_surface, cyclic_along=cyclic_along,
    )
    blurred = _geodesic_gaussian(
        extended_values, extended_distances, sigma,
    )[:, interior_columns:]
    blurred[:, 0] = float(border_field_sign)
    return blurred


def _mesh_nodal_areas(surface, cyclic_along=True):
    """Allocate one third of each cortical triangle's area to its vertices."""
    surface = np.asarray(surface, dtype=np.float64)
    shape = surface.shape[:2]
    areas = np.zeros(shape, dtype=np.float64)
    row_limit = shape[0] if cyclic_along else shape[0] - 1
    for row in range(row_limit):
        next_row = (row + 1) % shape[0]
        for column in range(shape[1] - 1):
            for triangle in (
                ((row, column), (next_row, column), (next_row, column + 1)),
                ((row, column), (next_row, column + 1), (row, column + 1)),
            ):
                points = np.asarray([surface[node] for node in triangle])
                if points.shape[1] == 2:
                    triangle_area = 0.5 * abs(np.cross(
                        points[1] - points[0], points[2] - points[0]
                    ))
                else:
                    triangle_area = 0.5 * np.linalg.norm(np.cross(
                        points[1] - points[0], points[2] - points[0]
                    ))
                for node in triangle:
                    areas[node] += triangle_area / 3.0
    return areas


def _fill_unassigned_geodesic(labels, distances, target_mask=None):
    """Extend patch cores using nearest cortical-mesh geodesic distance."""
    labels = np.asarray(labels, dtype=np.int64)
    if target_mask is None:
        target_mask = np.ones(labels.shape, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    assigned = (labels.ravel() >= 0) & target_mask.ravel()
    if not np.any(assigned):
        raise ValueError("No suprathreshold field-sign patches were found")
    assigned_indices = np.flatnonzero(assigned)
    target_indices = np.flatnonzero(target_mask.ravel())
    nearest = assigned_indices[np.argmin(
        distances[np.ix_(target_indices, assigned_indices)], axis=1,
    )]
    output = np.full(labels.size, -1, dtype=np.int64)
    output[target_indices] = labels.ravel()[nearest]
    return output.reshape(labels.shape)


def _graph_binary_morphology(mask, distances, radius, operation):
    """Apply a binary morphology operation within a geodesic radius."""
    mask = np.asarray(mask, dtype=bool)
    if radius is None or radius <= 0.0:
        return mask.copy()
    neighborhood = distances <= float(radius) + 1e-12
    flat = mask.ravel()

    def dilate(values):
        return np.any(neighborhood & values[None, :], axis=1)

    def erode(values):
        return np.all(~neighborhood | values[None, :], axis=1)

    if operation == "dilation":
        result = dilate(flat)
    elif operation == "erosion":
        result = erode(flat)
    elif operation == "closing":
        result = erode(dilate(flat))
    elif operation == "opening":
        result = dilate(erode(flat))
    else:
        raise ValueError(f"Unknown graph morphology operation: {operation}")
    return result.reshape(mask.shape)


def _eccentricity_minima(
    mask, azimuth, altitude, cyclic_along=True, distances=None,
    smoothing_sigma=None, bin_step=5.0, row_cut=None,
):
    """Find Garrett-style 5-degree-binned regional eccentricity minima."""
    center_azimuth = np.mean(azimuth[mask])
    center_altitude = np.mean(altitude[mask])
    azimuth_radians = np.radians(azimuth - center_azimuth)
    altitude_radians = np.radians(altitude - center_altitude)
    eccentricity = np.degrees(np.arctan(np.sqrt(
        np.tan(azimuth_radians) ** 2
        + np.tan(altitude_radians) ** 2 / np.cos(azimuth_radians) ** 2
    )))
    if distances is not None and smoothing_sigma is not None:
        eccentricity = _geodesic_gaussian(
            eccentricity, distances, smoothing_sigma,
        )
    cut_levels = np.arange(
        np.nanmin(eccentricity[mask]) - float(bin_step),
        np.nanmax(eccentricity[mask]) + 2.0 * float(bin_step),
        float(bin_step),
    )
    components = np.full(mask.shape, -1, dtype=np.int64)
    for cut_level in cut_levels:
        below = mask & (eccentricity <= cut_level)
        components = (_components_with_row_cut(below, row_cut) if row_cut is not None
                      else _cyclic_components(below, cyclic_along=cyclic_along))
        if len(np.unique(components[components >= 0])) > 1:
            break
    seeds = []
    for component in np.unique(components[components >= 0]):
        locations = np.argwhere(components == component)
        best = locations[np.argmin(eccentricity[components == component])]
        seeds.append((eccentricity[tuple(best)], int(best[0]), int(best[1])))
    seeds.sort()
    return eccentricity, [(row, column) for _, row, column in seeds[:4]]


def _watershed_patch(mask, eccentricity, seeds, cyclic_along=True, row_cut=None):
    """Marker-controlled watershed using a deterministic priority flood."""
    assignments = np.full(mask.shape, -1, dtype=np.int64)
    queue = []
    for seed_label, (row, column) in enumerate(seeds):
        assignments[row, column] = seed_label
        heapq.heappush(queue, (eccentricity[row, column], seed_label, row, column))
    while queue:
        flood_level, seed_label, row, column = heapq.heappop(queue)
        if assignments[row, column] != seed_label:
            continue
        for new_row, new_column in _grid_neighbors(
            row, column, mask.shape, cyclic_along=cyclic_along
        ):
            if row_cut is not None and row != new_row and {row, new_row} == {row_cut, (row_cut - 1) % mask.shape[0]}:
                continue
            if not mask[new_row, new_column]:
                continue
            new_level = max(flood_level, eccentricity[new_row, new_column])
            if assignments[new_row, new_column] < 0:
                assignments[new_row, new_column] = seed_label
                heapq.heappush(
                    queue, (new_level, seed_label, new_row, new_column)
                )
    return assignments


def _adjacent_pairs(labels, cyclic_along=True):
    pairs = set()
    for row, column in np.ndindex(labels.shape):
        current = labels[row, column]
        for neighbor in _grid_neighbors(
            row, column, labels.shape, cyclic_along=cyclic_along
        ):
            other = labels[neighbor]
            if current >= 0 and other >= 0 and current != other:
                pairs.add(tuple(sorted((int(current), int(other)))))
    return sorted(pairs)


def _fill_sign_constrained(labels, distances, field_sign, patch_signs, threshold, target_mask):
    """Fill missing nodes from same-sign seeds outside a neutral deadband."""
    output = np.where(target_mask, labels, -1).copy()
    assigned = np.flatnonzero(((labels >= 0) & target_mask).ravel())
    source_signs = np.asarray([patch_signs[int(labels.ravel()[i])] for i in assigned])
    missing = target_mask & (labels < 0)
    for sign in (-1, 0, 1):
        mask = missing & ((field_sign < -threshold) if sign == -1 else
                          (field_sign > threshold) if sign == 1 else
                          (np.abs(field_sign) <= threshold))
        targets = np.flatnonzero(mask.ravel())
        if not len(targets):
            continue
        sources = assigned if sign == 0 else assigned[source_signs == sign]
        if not len(sources):
            raise ValueError("No surviving core of required sign for constrained assignment")
        nearest = sources[np.argmin(distances[np.ix_(targets, sources)], axis=1)]
        output.ravel()[targets] = labels.ravel()[nearest]
    return output


def _fill_from_core_centroids(labels, surface, areas, target_mask):
    """Assign non-core nodes by Euclidean distance to area-weighted core centers."""
    labels = np.asarray(labels)
    ids = np.unique(labels[(labels >= 0) & target_mask])
    if not len(ids):
        raise ValueError("No surviving cores for centroid assignment")
    centers = np.asarray([np.average(surface[(labels == pid) & target_mask], axis=0,
        weights=areas[(labels == pid) & target_mask]) for pid in ids])
    output = np.where(target_mask, labels, -1).copy()
    missing = target_mask & (labels < 0)
    squared = np.sum((surface[missing, None, :] - centers[None, :, :]) ** 2, axis=2)
    output[missing] = ids[np.argmin(squared, axis=1)]
    return output, centers, ids


def _components_with_row_cut(mask, row_cut):
    """Connected components with edges before row_cut removed, including diagonals."""
    shifted = np.roll(mask, -int(row_cut), axis=0)
    return np.roll(_cyclic_components(shifted, cyclic_along=False), int(row_cut), axis=0)


def garrett_area_parcellation(
    retinotopy, field_sign_sigma=2.4, threshold_sd=0.75,
    redundancy_threshold=1.01, min_patch_pixels=4, cyclic_along=True,
    surface=None, field_sign_sigma_mm=None, field_sign_threshold=None,
    surface_units_per_mm=1.0, min_patch_area_mm2=None,
    min_patch_nodes=None, cleanup_radius_mm=0.0,
    dilation_radius_mm=0.0, eccentricity_sigma_mm=None,
    eccentricity_bin_step_deg=5.0, visual_space_pixel_size_deg=1.0,
    visual_space_close_iterations=0, merge_overlap_threshold=None,
    max_split_passes=2, fixed_field_sign_mask=None,
    fixed_field_sign_value=-1.0,
    v1_interior_field_sign=None, v1_border_field_sign=0.0,
    forced_row_cut=None,
    assignment_method="nearest_core",
    assignment_sign_threshold=None,
    post_cut_split=True,
):
    """Split a retinotopic sheet using Garrett et al. criteria 1--3.

    Criterion 4 identifies consistently located areas across animals and is
    intentionally not applied to this single simulated map. ``threshold_sd``
    defaults to 0.75 because the released MATLAB code computes 1.5 SD and
    thresholds at half that value. Supplying ``field_sign_sigma_mm`` enables
    physical mode: ``surface`` coordinates are converted with
    ``surface_units_per_mm``, smoothing and nearest-core assignment use mesh
    geodesic distances, and optional component thresholds use cortical area.
    ``fixed_field_sign_mask`` supplies a Dirichlet-like field-sign boundary:
    those nodes participate in smoothing at ``fixed_field_sign_value`` but are
    excluded from candidate-area cores, fills, splits, merges, and area totals.
    ``v1_interior_field_sign`` adds virtual tissue inward of column zero during
    physical smoothing; column zero is held at ``v1_border_field_sign`` and is
    excluded from candidate areas.
    """
    retinotopy = np.asarray(retinotopy, dtype=np.float64)
    physical_mode = field_sign_sigma_mm is not None
    if assignment_sign_threshold is not None:
        if not physical_mode or assignment_method != "nearest_core":
            raise ValueError("Sign-constrained assignment requires physical nearest_core mode")
        if not np.isfinite(assignment_sign_threshold) or not 0 <= assignment_sign_threshold <= 1:
            raise ValueError("assignment_sign_threshold must be between 0 and 1")
    if assignment_method not in ("nearest_core", "core_centroid"):
        raise ValueError("Unknown assignment_method")
    if assignment_method == "core_centroid" and not physical_mode:
        raise ValueError("core_centroid requires physical cortical geometry")
    if fixed_field_sign_mask is None:
        fixed_field_sign_mask = np.zeros(retinotopy.shape[:2], dtype=bool)
    fixed_field_sign_mask = np.asarray(fixed_field_sign_mask, dtype=bool)
    if fixed_field_sign_mask.shape != retinotopy.shape[:2]:
        raise ValueError("fixed_field_sign_mask must match the map shape")
    if np.any(fixed_field_sign_mask) and not physical_mode:
        raise ValueError("a fixed field-sign mask requires physical mode")
    v1_border_mask = np.zeros(retinotopy.shape[:2], dtype=bool)
    if v1_interior_field_sign is not None:
        if not physical_mode:
            raise ValueError("V1-conditioned smoothing requires physical mode")
        v1_border_mask[:, 0] = True
    active_mask = ~(fixed_field_sign_mask | v1_border_mask)
    distances = None
    nodal_areas = None
    surface_mm = None
    if physical_mode:
        if surface is None:
            raise ValueError("physical parcellation requires a cortical surface")
        if surface_units_per_mm <= 0.0:
            raise ValueError("surface_units_per_mm must be positive")
        surface_mm = np.asarray(surface, dtype=np.float64) / surface_units_per_mm
        distances = _mesh_geodesic_distances(
            surface_mm, cyclic_along=cyclic_along,
        )
        nodal_areas = _mesh_nodal_areas(
            surface_mm, cyclic_along=cyclic_along,
        )
    raw_field_sign = garrett_field_sign(
        retinotopy, sigma=0.0, cyclic_along=cyclic_along, surface=surface,
    )
    raw_field_sign[fixed_field_sign_mask] = float(fixed_field_sign_value)
    raw_field_sign[v1_border_mask] = float(v1_border_field_sign)
    if v1_interior_field_sign is not None:
        field_sign = _geodesic_gaussian_with_v1(
            raw_field_sign, surface_mm, field_sign_sigma_mm,
            cyclic_along=cyclic_along,
            v1_field_sign=v1_interior_field_sign,
            border_field_sign=v1_border_field_sign,
        )
    elif physical_mode:
        field_sign = _geodesic_gaussian(
            raw_field_sign, distances, field_sign_sigma_mm,
        )
    else:
        field_sign = matlab_gaussian_filter(
            raw_field_sign, sigma=field_sign_sigma,
            cyclic_along=cyclic_along,
        )
    field_sign[fixed_field_sign_mask] = float(fixed_field_sign_value)
    field_sign[v1_border_mask] = float(v1_border_field_sign)
    threshold = (
        float(field_sign_threshold)
        if field_sign_threshold is not None
        else threshold_sd * np.std(field_sign)
    )
    minimum_nodes = (
        int(min_patch_nodes)
        if min_patch_nodes is not None else int(min_patch_pixels)
    )

    def patch_is_large_enough(mask):
        if np.count_nonzero(mask) < minimum_nodes:
            return False
        if min_patch_area_mm2 is not None:
            if nodal_areas is None:
                raise ValueError("min_patch_area_mm2 requires physical mode")
            if float(np.sum(nodal_areas[mask])) < float(min_patch_area_mm2):
                return False
        return True

    core_labels = np.full(field_sign.shape, -1, dtype=np.int64)
    next_label = 0
    for sign_value in (1.0, -1.0):
        sign_mask = sign_value * field_sign >= threshold
        if physical_mode and cleanup_radius_mm > 0.0:
            sign_mask = _graph_binary_morphology(
                sign_mask, distances, cleanup_radius_mm, "closing",
            )
            sign_mask = _graph_binary_morphology(
                sign_mask, distances, cleanup_radius_mm, "opening",
            )
        sign_mask &= active_mask
        components = _cyclic_components(
            sign_mask,
            cyclic_along=cyclic_along,
        )
        for component in np.unique(components[components >= 0]):
            component_mask = components == component
            if not patch_is_large_enough(component_mask):
                continue
            core_labels[component_mask] = next_label
            next_label += 1
    core_centroids = None
    core_centroid_ids = None
    if physical_mode and assignment_method == "core_centroid":
        labels, core_centroids, core_centroid_ids = _fill_from_core_centroids(
            core_labels, surface_mm, nodal_areas, active_mask)
        # Centers stay fixed at the original cores: do not reassign using
        # dilated patches, which would restore nearest-node behavior.
        dilated_core_labels = core_labels.copy()
    elif physical_mode:
        core_signs = {int(pid): int(np.sign(np.median(field_sign[core_labels == pid])) or 1)
                      for pid in np.unique(core_labels[core_labels >= 0])}
        def fill_physical(seeds):
            if assignment_sign_threshold is not None:
                return _fill_sign_constrained(seeds, distances, field_sign, core_signs,
                                              assignment_sign_threshold, active_mask)
            return _fill_unassigned_geodesic(seeds, distances, target_mask=active_mask)
        labels = fill_physical(core_labels)
        nearest_core_distance = np.min(
            distances[:, np.flatnonzero(core_labels.ravel() >= 0)], axis=1,
        ).reshape(core_labels.shape)
        dilated_core_labels = np.where(
            active_mask
            & (nearest_core_distance <= float(dilation_radius_mm) + 1e-12),
            labels, -1,
        )
        if dilation_radius_mm > 0.0:
            labels = fill_physical(dilated_core_labels)
    else:
        labels = _fill_unassigned(core_labels, cyclic_along=cyclic_along)
        dilated_core_labels = labels.copy()
    labels = _relabel(labels)
    initial_labels = labels.copy()
    azimuth = retinotopy[:, :, 0]
    altitude = retinotopy[:, :, 1]
    visual_padding = (
        float(visual_space_close_iterations)
        * float(visual_space_pixel_size_deg)
        + float(visual_space_pixel_size_deg)
    )
    visual_bounds = (
        np.floor(azimuth.min() / visual_space_pixel_size_deg)
        * visual_space_pixel_size_deg - visual_padding,
        np.ceil(azimuth.max() / visual_space_pixel_size_deg)
        * visual_space_pixel_size_deg + visual_padding,
        np.floor(altitude.min() / visual_space_pixel_size_deg)
        * visual_space_pixel_size_deg - visual_padding,
        np.ceil(altitude.max() / visual_space_pixel_size_deg)
        * visual_space_pixel_size_deg + visual_padding,
    )
    split_pass_labels = []
    split_events = []

    # Criterion 2: split redundant visual representations at eccentricity
    # watersheds. Reassess because one split can expose another redundant map.
    def split_redundant(labels, row_cut=None, stage="before_cut"):
        for _ in range(int(max_split_passes)):
            changed = False
            rebuilt = np.full(labels.shape, -1, dtype=np.int64)
            rebuilt_next = 0
            for patch in np.unique(labels[labels >= 0]):
                patch_mask = labels == patch
                ratio = _triangle_visual_coverage_ratio(
                    patch_mask, azimuth, altitude,
                    pixel_size=visual_space_pixel_size_deg,
                    close_iterations=visual_space_close_iterations,
                    bounds=visual_bounds,
                )
                eccentricity, seeds = _eccentricity_minima(
                    patch_mask, azimuth, altitude, cyclic_along=cyclic_along,
                    distances=distances,
                    smoothing_sigma=eccentricity_sigma_mm,
                    bin_step=eccentricity_bin_step_deg, row_cut=row_cut,
                )
                if ratio > redundancy_threshold and len(seeds) > 1:
                    split = _watershed_patch(
                        patch_mask, eccentricity, seeds,
                        cyclic_along=cyclic_along, row_cut=row_cut,
                    )
                    valid_parts = [
                        part for part in np.unique(split[split >= 0])
                        if patch_is_large_enough(split == part)
                    ]
                    if len(valid_parts) > 1:
                        split_events.append({
                            "pass": len(split_pass_labels) + 1,
                            "stage": stage,
                            "source_patch": int(patch),
                            "coverage_ratio": float(ratio),
                            "seed_count": len(seeds),
                            "part_sizes": tuple(
                                int(np.count_nonzero(split == part))
                                for part in valid_parts
                            ),
                        })
                        for part in valid_parts:
                            rebuilt[split == part] = rebuilt_next
                            rebuilt_next += 1
                        # Attach any tiny watershed pieces to their nearest kept part.
                        leftovers = patch_mask & (rebuilt < 0)
                        if np.any(leftovers):
                            if physical_mode:
                                source_indices = np.flatnonzero(
                                    patch_mask.ravel() & (rebuilt.ravel() >= 0)
                                )
                                target_indices = np.flatnonzero(leftovers.ravel())
                                nearest = source_indices[np.argmin(
                                    distances[np.ix_(target_indices, source_indices)],
                                    axis=1,
                                )]
                                rebuilt.ravel()[target_indices] = rebuilt.ravel()[nearest]
                            else:
                                local = np.where(rebuilt >= 0, rebuilt, -1)
                                filled = _fill_unassigned(
                                    local, cyclic_along=cyclic_along,
                                )
                                rebuilt[leftovers] = filled[leftovers]
                        changed = True
                        continue
                rebuilt[patch_mask] = rebuilt_next
                rebuilt_next += 1
            labels = _relabel(rebuilt)
            split_pass_labels.append(labels.copy())
            if not changed:
                break

        return labels

    labels = split_redundant(labels)

    # Optional anatomical prior: disconnect patches across a radial row seam.
    # Applied after watershed splitting; merge candidates must also respect it.
    forced_split_events = []
    if forced_row_cut is not None:
        if not isinstance(forced_row_cut, (int, np.integer)) or not 0 <= forced_row_cut < labels.shape[0]:
            raise ValueError("forced_row_cut must be an integer row within the sheet")
        rebuilt = np.full(labels.shape, -1, dtype=np.int64)
        next_label = 0
        for patch in np.unique(labels[labels >= 0]):
            components = _components_with_row_cut(labels == patch, forced_row_cut)
            parts = np.unique(components[components >= 0])
            if len(parts) > 1:
                forced_split_events.append({"source_patch": int(patch), "part_sizes": [int(np.sum(components == part)) for part in parts]})
            for part in parts:
                rebuilt[components == part] = next_label
                next_label += 1
        labels = rebuilt
        if post_cut_split:
            labels = split_redundant(labels, row_cut=forced_row_cut, stage="after_cut")

    # Criterion 3: fuse adjacent same-sign patches when their combined visual
    # representation is nonredundant.
    merge_events = []
    while True:
        signs = _patch_signs(labels, field_sign)
        merged = False
        for first, second in _adjacent_pairs(labels, cyclic_along=cyclic_along):
            if signs[first] != signs[second]:
                continue
            first_mask = labels == first
            second_mask = labels == second
            if forced_row_cut is not None:
                components = _components_with_row_cut(first_mask | second_mask, forced_row_cut)
                if len(np.unique(components[components >= 0])) > 1:
                    continue
            if merge_overlap_threshold is None:
                union = first_mask | second_mask
                ratio = _triangle_visual_coverage_ratio(
                    union, azimuth, altitude,
                    pixel_size=visual_space_pixel_size_deg,
                    close_iterations=visual_space_close_iterations,
                    bounds=visual_bounds,
                )
                should_merge = ratio <= redundancy_threshold
                merge_metrics = {"coverage_ratio": float(ratio)}
            else:
                first_overlap, second_overlap = _visual_overlap_ratios(
                    first_mask, second_mask, azimuth, altitude,
                    pixel_size=visual_space_pixel_size_deg,
                    close_iterations=visual_space_close_iterations,
                    bounds=visual_bounds,
                )
                should_merge = (
                    first_overlap <= merge_overlap_threshold
                    and second_overlap <= merge_overlap_threshold
                )
                merge_metrics = {
                    "first_overlap_ratio": float(first_overlap),
                    "second_overlap_ratio": float(second_overlap),
                }
            if should_merge:
                merge_events.append({
                    "first_patch": int(first),
                    "second_patch": int(second),
                    **merge_metrics,
                })
                labels[labels == second] = first
                labels = _relabel(labels)
                merged = True
                break
        if not merged:
            break

    signs = _patch_signs(labels, field_sign)
    centers = np.asarray([
        (
            np.mean(np.nonzero(labels == patch)[0]),
            np.mean(np.nonzero(labels == patch)[1]),
            signs[patch],
        )
        for patch in np.unique(labels[labels >= 0])
    ])
    ratios = np.asarray([
        _triangle_visual_coverage_ratio(
            labels == patch, azimuth, altitude,
            pixel_size=visual_space_pixel_size_deg,
            close_iterations=visual_space_close_iterations,
            bounds=visual_bounds,
        )
        for patch in np.unique(labels[labels >= 0])
    ])
    unresolved_redundant = []
    for patch, ratio in zip(np.unique(labels[labels >= 0]), ratios):
        if ratio <= redundancy_threshold:
            continue
        patch_mask = labels == patch
        eccentricity, seeds = _eccentricity_minima(
            patch_mask, azimuth, altitude, cyclic_along=cyclic_along,
            distances=distances, smoothing_sigma=eccentricity_sigma_mm,
            bin_step=eccentricity_bin_step_deg,
        )
        candidate_sizes = ()
        retained_sizes = ()
        if len(seeds) > 1:
            candidate = _watershed_patch(
                patch_mask, eccentricity, seeds,
                cyclic_along=cyclic_along,
            )
            candidate_sizes = tuple(
                int(np.count_nonzero(candidate == part))
                for part in np.unique(candidate[candidate >= 0])
            )
            retained_sizes = tuple(
                int(np.count_nonzero(candidate == part))
                for part in np.unique(candidate[candidate >= 0])
                if patch_is_large_enough(candidate == part)
            )
        unresolved_redundant.append({
            "patch": int(patch),
            "coverage_ratio": float(ratio),
            "seed_count": len(seeds),
            "candidate_part_sizes": candidate_sizes,
            "retained_part_sizes": retained_sizes,
        })
    patch_areas_mm2 = (
        np.asarray([
            np.sum(nodal_areas[labels == patch])
            for patch in np.unique(labels[labels >= 0])
        ])
        if nodal_areas is not None else None
    )
    return {
        "labels": labels,
        "centers": centers,
        "raw_field_sign": raw_field_sign,
        "field_sign": field_sign,
        "threshold": threshold,
        "coverage_ratios": ratios,
        "core_labels": core_labels,
        "core_centroids_mm": core_centroids,
        "core_centroid_ids": core_centroid_ids,
        "dilated_core_labels": dilated_core_labels,
        "initial_labels": initial_labels,
        "split_pass_labels": tuple(split_pass_labels),
        "split_events": tuple(split_events),
        "merge_events": tuple(merge_events),
        "forced_split_events": tuple(forced_split_events),
        "unresolved_redundant": tuple(unresolved_redundant),
        "nodal_areas_mm2": nodal_areas,
        "patch_areas_mm2": patch_areas_mm2,
        "parameters": {
            "physical_mode": physical_mode,
            "assignment_method": assignment_method,
            "assignment_sign_threshold": assignment_sign_threshold,
            "forced_row_cut": forced_row_cut,
            "post_cut_split": bool(post_cut_split),
            "field_sign_sigma_mm": field_sign_sigma_mm,
            "field_sign_threshold": field_sign_threshold,
            "surface_units_per_mm": surface_units_per_mm,
            "min_patch_area_mm2": min_patch_area_mm2,
            "min_patch_nodes": minimum_nodes,
            "cleanup_radius_mm": cleanup_radius_mm,
            "dilation_radius_mm": dilation_radius_mm,
            "eccentricity_sigma_mm": eccentricity_sigma_mm,
            "eccentricity_bin_step_deg": eccentricity_bin_step_deg,
            "visual_space_pixel_size_deg": visual_space_pixel_size_deg,
            "visual_space_close_iterations": visual_space_close_iterations,
            "split_overlap_threshold": redundancy_threshold,
            "merge_overlap_threshold": merge_overlap_threshold,
            "max_split_passes": max_split_passes,
            "fixed_field_sign_nodes": int(np.count_nonzero(fixed_field_sign_mask)),
            "fixed_field_sign_value": float(fixed_field_sign_value),
            "v1_interior_field_sign": v1_interior_field_sign,
            "v1_border_field_sign": float(v1_border_field_sign),
            "v1_border_nodes": int(np.count_nonzero(v1_border_mask)),
        },
        "fixed_field_sign_mask": fixed_field_sign_mask,
        "v1_border_mask": v1_border_mask,
    }


def upsample_sheet(values, step=0.1):
    """Cubic-spline rows from 80 to 791 samples, matching the ribbon plots."""
    values = np.asarray(values, dtype=np.float64)
    source = np.arange(values.shape[0], dtype=np.float64)
    target = np.arange(0.0, values.shape[0] - 1.0 + step / 2.0, step)
    return target, CubicSpline(source, values, axis=0)(target)


def load_legacy_plot_data(run_dir, map_h=80, checkpoint=None, sigma=1.2):
    """Load one run and assemble every array needed by the ported figure."""
    run_dir = Path(run_dir)
    checkpoint_path, values = load_checkpoint(run_dir, checkpoint)
    retinotopy = reshape_model_map(values, map_h=map_h)
    boundary = np.loadtxt(run_dir / "boundary.csv", delimiter=",")
    full_retinotopy = prepend_v1_boundary(retinotopy, boundary)
    manifest_path = run_dir / "run-manifest.json"
    cyclic_along = False
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        cyclic_along = manifest.get("parameters", {}).get("cyclic_along", False)
    surface = generate_mouse_surface_grid(
        map_h=retinotopy.shape[0], map_w=retinotopy.shape[1],
        cyclic_along=cyclic_along,
    ).copy()
    # The legacy fill calls negate SVG y so dorsal is displayed upward.
    surface[:, :, 1] *= -1.0
    return {
        "checkpoint_path": checkpoint_path,
        "retinotopy": retinotopy,
        "boundary": boundary,
        "full_retinotopy": full_retinotopy,
        "surface": surface,
        "cyclic_along": cyclic_along,
        **matlab_polar_maps(
            full_retinotopy, sigma=sigma, cyclic_along=cyclic_along
        ),
    }
