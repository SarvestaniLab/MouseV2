"""Build an editable Figure 2 draft from the selected publication fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle
from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
    TwoSlopeNorm,
    to_rgba,
)
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull, cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "elastictopy"))

from legacy_plotting import (  # noqa: E402
    garrett_area_parcellation,
    matlab_gaussian_filter,
    matlab_polar_maps,
    prepend_v1_boundary,
    reshape_model_map,
    upsample_sheet,
)
from mouse_surface import DEFAULT_BORDER_SVG, generate_mouse_surface_grid  # noqa: E402
from fig2.zhuang_v1 import build_warped_zhuang_v1  # noqa: E402


BLUE = "#3a53a4"
RED = "#ed2224"
GRAY = "#4d4d4d"
FIELD_SIGN_COLORS = (
    "#08519c", "#3182bd", "#bdd7e7", "#ffffff", "#fcae91", "#d7301f", "#8c2d04",
)
FIELD_SIGN_CMAP = LinearSegmentedColormap.from_list(
    "field_sign_smooth",
    FIELD_SIGN_COLORS,
    N=256,
)
INJECTION_COLORS = ("#ed2224", "#f6c915", "#18a957")


def cortical_coordinate_units_per_mm() -> float:
    """Return the registered SVG-coordinate scale from the Waters provenance."""
    manifest_path = (
        PROJECT_ROOT / "data" / "waters2019" / "derived" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    transform = manifest["similarity_transform_to_zhuang_svg"]
    return float(np.exp(transform[0]) * manifest["pixels_per_mm"])


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = list(run_dir.glob("y-*.data"))
    if not checkpoints:
        raise FileNotFoundError(f"No model checkpoint found in {run_dir}")
    return max(checkpoints, key=lambda path: int(path.stem.rsplit("-", 1)[-1]))


def clean_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=6, width=0.6, length=2)


def panel_label(figure, label, x, y):
    figure.text(x, y, label, fontsize=30, fontweight="normal", va="top")


def plot_visual_field(axis, retinotopy, boundary, title, prototypes=None):
    if prototypes is not None:
        stride = max(1, len(prototypes) // 700)
        axis.scatter(
            prototypes[::stride, 0], prototypes[::stride, 1], s=2.0,
            color="#4771d1", alpha=0.45, linewidths=0,
        )
    # MATLAB brewermap's visible PuOr ordering is purple-to-orange here.
    trajectory_colors = plt.cm.PuOr(np.linspace(1.0, 0.0, retinotopy.shape[0]))
    for row in range(retinotopy.shape[0]):
        axis.plot(
            retinotopy[row, :, 0], retinotopy[row, :, 1],
            color=trajectory_colors[row], linewidth=0.67,
        )
    boundary_closed = np.vstack((boundary, boundary[0]))
    axis.plot(
        boundary_closed[:, 0], boundary_closed[:, 1],
        color="black", linewidth=1.5,
    )
    outer_edge_closed = np.vstack((retinotopy[:, -1, :], retinotopy[0, -1, :]))
    axis.plot(
        outer_edge_closed[:, 0], outer_edge_closed[:, 1],
        color="0.60", linewidth=0.95,
    )
    axis.set(xlabel="Azimuth (°)", ylabel="Elevation (°)")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-4, 143)
    axis.set_ylim(-43, 63)
    clean_axis(axis)
    axis.set_xticks((0, 50, 100))
    axis.set_yticks((-40, 0, 40))
    axis.tick_params(labelsize=7.5, width=0.75, length=2.5)
    axis.xaxis.label.set_size(8.5)
    axis.yaxis.label.set_size(8.5)
    axis.set_title(title, fontsize=9.5, pad=3)


def plot_cortical(
    axis, surface, values, title, cmap, norm, show_mesh=False,
    shading="gouraud",
):
    surface_closed = np.concatenate((surface, surface[:1]), axis=0)
    values_closed = np.concatenate((values, values[:1]), axis=0)
    image = axis.pcolormesh(
        surface_closed[:, :, 0], surface_closed[:, :, 1], values_closed,
        cmap=cmap, norm=norm, shading=shading, rasterized=True,
    )
    if show_mesh:
        for row in range(surface.shape[0]):
            axis.plot(
                surface[row, :, 0], surface[row, :, 1],
                color="white", alpha=0.34, lw=0.24,
            )
        for column in range(surface.shape[1]):
            axis.plot(
                surface_closed[:, column, 0], surface_closed[:, column, 1],
                color="white", alpha=0.34, lw=0.24,
            )
    axis.plot(
        surface_closed[:, 0, 0], surface_closed[:, 0, 1],
        color="black", lw=1.5,
    )
    axis.plot(
        surface_closed[:, -1, 0], surface_closed[:, -1, 1],
        color="0.60", lw=0.95,
    )
    axis.set_title(title, fontsize=10, pad=3)
    axis.set_aspect("equal")
    axis.axis("off")
    return image


def plot_cortical_areas(axis, surface, labels, title, colors, node_regions=False):
    """Render categorical areas; node_regions preserves every corner assignment."""
    polygons = []
    polygon_labels = []
    height, width = labels.shape
    for row in range(height):
        next_row = (row + 1) % height
        for column in range(width - 1):
            corners = np.asarray((
                surface[row, column], surface[next_row, column],
                surface[next_row, column + 1], surface[row, column + 1],
            ))
            corner_labels = np.asarray((
                labels[row, column], labels[next_row, column],
                labels[next_row, column + 1], labels[row, column + 1],
            )).copy()
            if node_regions:
                # The fixed V1 border has no area identity. Extend the first
                # active radial node to it, without voting across angular rows.
                if column == 0:
                    if corner_labels[0] < 0: corner_labels[0] = corner_labels[3]
                    if corner_labels[1] < 0: corner_labels[1] = corner_labels[2]
                center = corners.mean(axis=0)
                for index, label in enumerate(corner_labels):
                    if label < 0:
                        continue
                    polygons.append(np.asarray((
                        corners[index],
                        (corners[index] + corners[(index + 1) % 4]) / 2,
                        center,
                        (corners[index] + corners[(index - 1) % 4]) / 2,
                    )))
                    polygon_labels.append(int(label))
            else:
                valid = corner_labels[corner_labels >= 0]
                if not len(valid):
                    continue
                polygons.append(corners)
                polygon_labels.append(int(np.argmax(np.bincount(valid))))
    collection = PolyCollection(
        polygons,
        facecolors=np.asarray(colors)[np.asarray(polygon_labels, dtype=int)],
        edgecolors="none",
        antialiased=False,
        rasterized=True,
    )
    axis.add_collection(collection)
    surface_closed = np.concatenate((surface, surface[:1]), axis=0)
    axis.plot(
        surface_closed[:, 0, 0], surface_closed[:, 0, 1],
        color="black", lw=1.5,
    )
    axis.plot(
        surface_closed[:, -1, 0], surface_closed[:, -1, 1],
        color="0.60", lw=0.95,
    )
    axis.autoscale_view()
    axis.set_title(title, fontsize=10, pad=3)
    axis.set_aspect("equal")
    axis.axis("off")
    return collection


def contrasting_text_color(color):
    """Choose legible black or white text for an RGBA area color."""
    red, green, blue = to_rgba(color)[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.58 else "white"


def label_cortical_areas(axis, surface, labels, colors):
    """Place a matching numbered badge safely inside every territory."""
    height = labels.shape[0]
    for area_index, color in enumerate(colors):
        mask = labels == area_index
        # Tile the cyclic dimension so seam-spanning territories get a valid
        # interior-distance maximum rather than two artificial edge maxima.
        tiled_mask = np.concatenate((mask, mask, mask), axis=0)
        interior_distance = distance_transform_edt(tiled_mask)[height:2 * height]
        interior_distance[~mask] = -1.0
        row, column = np.unravel_index(
            np.argmax(interior_distance), interior_distance.shape
        )
        x_position, y_position = surface[row, column]
        axis.text(
            x_position, y_position, str(area_index + 1),
            fontsize=7.0, fontweight="bold", ha="center", va="center",
            color=contrasting_text_color(color), zorder=8,
            bbox={
                "boxstyle": "circle,pad=0.18",
                "facecolor": color,
                "edgecolor": "white",
                "linewidth": 0.8,
            },
        )


def add_model_row(figure, y, title, retinotopy, boundary, surface, field_sign_override=None):
    full = prepend_v1_boundary(retinotopy, boundary)
    maps = matlab_polar_maps(full, sigma=1.6, cyclic_along=True)
    if field_sign_override is not None:
        maps["field_sign_blurred"] = field_sign_override
    figure.text(
        0.240, y + 0.195, title,
        fontsize=13, ha="center", fontweight="normal",
    )
    axes = [
        figure.add_axes([0.035, y, 0.105, 0.17]),
        figure.add_axes([0.152, y, 0.092, 0.17]),
        figure.add_axes([0.256, y, 0.092, 0.17]),
        figure.add_axes([0.353, y, 0.092, 0.17]),
    ]
    plot_visual_field(axes[0], retinotopy, boundary, "")
    eccentricity = plot_cortical(
        axes[1], surface, maps["eccentricity"], "Eccentricity",
        "turbo", Normalize(0, 140), show_mesh=False,
    )
    angle = plot_cortical(
        axes[2], surface, maps["visual_angle"], "Visual angle",
        "turbo", Normalize(-90, 90), show_mesh=False,
    )
    sign = plot_cortical(
        axes[3], surface, maps["field_sign_blurred"], "Field sign",
        FIELD_SIGN_CMAP, TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), show_mesh=False,
    )
    for axis, image, ticks in (
        (axes[1], eccentricity, (0, 70, 140)),
        (axes[2], angle, (-90, 0, 90)),
        (axes[3], sign, (-1, 0, 1)),
    ):
        colorbar = figure.colorbar(image, ax=axis, fraction=0.05, pad=0.015, ticks=ticks)
        colorbar.ax.tick_params(labelsize=7, length=2)
    return maps


def _cortical_sites_from_rf(samples, target_rf):
    """Map tentative RF coordinates to local V1 cortical coordinates."""
    sites = []
    for target in np.asarray(target_rf, dtype=float):
        distance_squared = np.sum((samples[:, 2:4] - target) ** 2, axis=1)
        nearest = np.argpartition(distance_squared, 8)[:8]
        weights = 1.0 / (0.25 + distance_squared[nearest])
        sites.append(
            np.average(samples[nearest, :2], axis=0, weights=weights)
        )
    return np.asarray(sites)


def plot_panel_f(figure, boundary, fitted_full, surface, labels):
    """Plot cortical V1 injections and their matched retinotopic projections."""
    warped = build_warped_zhuang_v1(
        PROJECT_ROOT, surface=surface, boundary_rf=boundary
    )
    dense = np.asarray(warped["dense"])
    samples = np.asarray(warped["samples"])
    target_rf = np.asarray(((49.0, 30.0), (39.0, 12.0), (27.0, -5.0)))
    cortical_sites = _cortical_sites_from_rf(samples, target_rf)
    radius_mm = 0.25
    radius_cortex = radius_mm * cortical_coordinate_units_per_mm()

    figure.text(
        0.688,
        0.432,
        "V1 injections predict retinotopically matched HVA projections",
        fontsize=10.5,
        fontweight="normal",
        va="top",
    )
    axis_rf = figure.add_axes([0.665, 0.035, 0.145, 0.320])
    axis_cortex = figure.add_axes([0.822, 0.025, 0.157, 0.340])

    _, retinotopy_up = upsample_sheet(fitted_full, step=0.1)
    correspondence_colors = plt.cm.PuOr(
        np.linspace(0.0, 1.0, len(retinotopy_up))
    )
    boundary_closed = np.vstack((boundary, boundary[0]))
    axis_rf.fill(
        boundary_closed[:, 0], boundary_closed[:, 1], color="0.97", zorder=0
    )
    for row in range(0, len(retinotopy_up), 6):
        axis_rf.plot(
            retinotopy_up[row, :, 0],
            retinotopy_up[row, :, 1],
            color=correspondence_colors[-1 - row],
            alpha=0.78,
            lw=0.40,
            zorder=1,
        )
    for column in range(0, retinotopy_up.shape[1], 8):
        ring = retinotopy_up[:, column]
        axis_rf.plot(
            np.r_[ring[:, 0], ring[0, 0]],
            np.r_[ring[:, 1], ring[0, 1]],
            color="0.86",
            lw=0.30,
            zorder=1,
        )
    axis_rf.plot(
        boundary_closed[:, 0], boundary_closed[:, 1], color="black", lw=1.5
    )

    for site_number, (site, color) in enumerate(
        zip(cortical_sites, INJECTION_COLORS), start=1
    ):
        cortical_distance = np.linalg.norm(dense[:, :2] - site, axis=1)
        footprint = dense[cortical_distance <= radius_cortex, 2:4]
        if len(footprint) >= 3:
            hull = footprint[ConvexHull(footprint).vertices]
            axis_rf.fill(
                hull[:, 0],
                hull[:, 1],
                facecolor=to_rgba(color, 0.20),
                edgecolor=color,
                linewidth=1.0,
                zorder=4,
            )
        nearest = samples[
            np.argmin(np.linalg.norm(samples[:, :2] - site, axis=1))
        ]
        axis_rf.scatter(
            nearest[2], nearest[3], s=78, color=color,
            edgecolor="0.15", linewidth=0.8, zorder=6,
        )
        axis_rf.text(
            nearest[2], nearest[3], str(site_number), fontsize=5.8,
            ha="center", va="center", color="black", zorder=7,
        )
    axis_rf.set(
        title="Retinotopic coverage from V1 injections",
        xlabel="Azimuth (°)",
        ylabel="Elevation (°)",
        xlim=(-4, 143),
        ylim=(-43, 63),
    )
    axis_rf.set_aspect("equal", adjustable="box")
    clean_axis(axis_rf)
    axis_rf.set_xticks((0, 45, 90, 135))
    axis_rf.set_yticks((-40, 0, 40))
    axis_rf.set_title(axis_rf.get_title(), fontsize=8, pad=3)
    axis_rf.xaxis.label.set_size(7)
    axis_rf.yaxis.label.set_size(7)

    _, surface_up = upsample_sheet(surface, step=0.1)
    inner = surface[:, 0]
    outer = surface[:, -1]
    inner_closed = np.vstack((inner, inner[0]))
    outer_closed = np.vstack((outer, outer[0]))
    axis_cortex.fill(
        inner_closed[:, 0], inner_closed[:, 1], color="0.96", zorder=0
    )
    for row in range(0, len(surface_up), 6):
        axis_cortex.plot(
            surface_up[row, :, 0], surface_up[row, :, 1],
            color=correspondence_colors[-1 - row],
            alpha=0.78, lw=0.40, zorder=1,
        )
    for column in range(0, surface_up.shape[1], 8):
        ring = surface_up[:, column]
        axis_cortex.plot(
            np.r_[ring[:, 0], ring[0, 0]],
            np.r_[ring[:, 1], ring[0, 1]],
            color="0.88", lw=0.28, zorder=1,
        )
    axis_cortex.tricontour(
        dense[:, 0], dense[:, 1], dense[:, 2],
        levels=(0, 20, 40, 60, 80, 100, 120),
        colors="0.48", linewidths=0.42, zorder=2,
    )
    axis_cortex.tricontour(
        dense[:, 0], dense[:, 1], dense[:, 3],
        levels=(-40, -20, 0, 20, 40, 60),
        colors="0.48", linewidths=0.42, linestyles="dashed", zorder=2,
    )
    axis_cortex.plot(
        inner_closed[:, 0], inner_closed[:, 1], color="black", lw=1.8, zorder=3
    )
    axis_cortex.plot(
        outer_closed[:, 0], outer_closed[:, 1], color="0.55", lw=1.1, zorder=3
    )

    hva_rf = fitted_full[:, 1:, :].reshape(-1, 2)
    hva_xy = surface[:, 1:, :].reshape(-1, 2)
    nearest_samples = samples[cKDTree(samples[:, 2:4]).query(hva_rf)[1], :2]
    for site_number, (site, color) in enumerate(
        zip(cortical_sites, INJECTION_COLORS), start=1
    ):
        selected = np.linalg.norm(nearest_samples - site, axis=1) <= radius_cortex
        axis_cortex.scatter(
            hva_xy[selected, 0], hva_xy[selected, 1],
            s=26, facecolor=to_rgba(color, 0.82),
            edgecolor="white", linewidth=0.30, zorder=5,
        )
        axis_cortex.add_patch(
            Circle(
                site, radius_cortex, facecolor=to_rgba(color, 0.16),
                edgecolor=color, linewidth=0.9, zorder=5,
            )
        )
        axis_cortex.scatter(
            site[0], site[1], s=66, color=color,
            edgecolor="0.15", linewidth=0.8, zorder=6,
        )
        axis_cortex.text(
            site[0], site[1], str(site_number), fontsize=5.8,
            ha="center", va="center", color="black", zorder=7,
        )

    centroids = []
    for label_value in np.unique(labels):
        points = surface[labels == label_value]
        centroids.append(points.mean(axis=0) if len(points) else None)
    for label_value, centroid in enumerate(centroids, start=1):
        if centroid is not None:
            axis_cortex.text(
                centroid[0], centroid[1], f"HVA {label_value}",
                fontsize=4.8, color="0.42", ha="center", va="center", zorder=4,
            )
    axis_cortex.text(
        inner[:, 0].mean(), inner[:, 1].mean(), "V1",
        fontsize=6.2, ha="center", va="center", zorder=4,
    )
    scale_length = 0.5 * cortical_coordinate_units_per_mm()
    scale_x = outer[:, 0].min() + 0.15
    scale_y = outer[:, 1].min() - 0.18
    axis_cortex.plot(
        (scale_x, scale_x + scale_length), (scale_y, scale_y),
        color="black", lw=1.5, zorder=7,
    )
    axis_cortex.text(
        scale_x + scale_length / 2, scale_y - 0.12, "0.5 mm",
        fontsize=5.5, ha="center", va="top",
    )
    axis_cortex.set_title(
        "V1 injections and matched HVA projections", fontsize=8, pad=3
    )
    axis_cortex.set_aspect("equal")
    axis_cortex.axis("off")


def load_run(run_dir: Path):
    boundary = np.loadtxt(run_dir / "boundary.csv", delimiter=",")
    prototypes = np.loadtxt(run_dir / "vf.csv", delimiter=",")
    fitted = reshape_model_map(np.loadtxt(latest_checkpoint(run_dir), delimiter=","), map_h=80)
    full = prepend_v1_boundary(fitted, boundary)
    maps = matlab_polar_maps(full, sigma=1.2, cyclic_along=True)
    return {
        "run_dir": run_dir,
        "boundary": boundary,
        "prototypes": prototypes,
        "fitted": fitted,
        "full": full,
        "maps": maps,
    }


def build_figure(
    high_b1_run_dir: Path, primary_run_dir: Path,
    surface_svg: Path,
    modern_partition=False,
    forced_row_cut=0,
):
    high_b1 = load_run(high_b1_run_dir)
    primary = load_run(primary_run_dir)
    high_parameters = json.loads((high_b1_run_dir / "run-manifest.json").read_text())
    primary_parameters = json.loads((primary_run_dir / "run-manifest.json").read_text())
    boundary = primary["boundary"]
    prototypes = primary["prototypes"]
    fitted = primary["fitted"]
    fitted_full = primary["full"]
    fitted_maps = primary["maps"]
    if not np.array_equal(prototypes, high_b1["prototypes"]):
        raise ValueError("Model runs do not share the same spiral prototypes")
    if not np.array_equal(boundary, high_b1["boundary"]):
        raise ValueError("Model runs do not share the same V1 boundary")
    surface = generate_mouse_surface_grid(
        svg_path=surface_svg, map_h=80, map_w=8, cyclic_along=True,
    ).copy()
    surface[:, :, 1] *= -1.0
    modern_audits = {}
    if modern_partition:
        from fig2.plot_garrett_watershed_visual_footprints import parcellate
        from scripts.waters_organization_score import normalize_visual
        for key, run in (("high", high_b1), ("primary", primary)):
            normalized = normalize_visual(run["full"], np.isfinite(run["full"][:, 1:]).all(axis=2))
            modern_audits[key] = parcellate({"full": normalized}, surface,
                                           forced_row_cut=forced_row_cut,
                                           assignment_sign_threshold=0. if key == "primary" else None,
                                           merge_overlap_threshold=0.,
                                           redundancy_threshold=1.01, post_cut_split=True)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.linewidth": 0.6,
        "svg.fonttype": "none",
    })
    figure = plt.figure(figsize=(16.0, 9.9), facecolor="white")

    # A — current inputs.
    panel_label(figure, "A", 0.018, 0.982)
    figure.text(
        0.260, 0.965,
        "Visual field and cortical coverage by\nelastic net model",
        fontsize=16, fontweight="normal", ha="center", va="top",
    )
    axis_a1 = figure.add_axes([0.055, 0.650, 0.18, 0.250])
    axis_a1.scatter(
        prototypes[:, 0], prototypes[:, 1],
        s=1.7, color="#3a53a4", linewidths=0, alpha=0.82,
    )
    axis_a1.plot(
        np.r_[boundary[:, 0], boundary[0, 0]],
        np.r_[boundary[:, 1], boundary[0, 1]], color="black", lw=1.0,
    )
    axis_a1.set(
        title="",
        xlabel="Azimuth (°)", ylabel="Elevation (°)",
    )
    axis_a1.set_aspect("equal", adjustable="box")
    clean_axis(axis_a1)
    axis_a1.set_xticks((0, 50, 100))
    axis_a1.set_yticks((-40, 0, 40))
    axis_a1.tick_params(labelsize=9, width=0.8, length=3)
    axis_a1.xaxis.label.set_size(11)
    axis_a1.yaxis.label.set_size(11)
    axis_a2 = figure.add_axes([0.255, 0.640, 0.215, 0.270])
    for row in range(surface.shape[0]):
        axis_a2.plot(
            surface[row, :, 0], surface[row, :, 1],
            color=plt.cm.PuOr(1.0 - row / (surface.shape[0] - 1)), lw=1.45,
        )
    for column in range(surface.shape[1]):
        closed_column = np.vstack((surface[:, column, :], surface[0, column, :]))
        axis_a2.plot(
            closed_column[:, 0], closed_column[:, 1],
            color="0.20", lw=0.35, alpha=0.42,
        )
    axis_a2.scatter(
        surface[:, :, 0].ravel(), surface[:, :, 1].ravel(),
        s=0.55, color="black", alpha=0.48, linewidths=0, zorder=4,
    )
    inner_closed = np.vstack((surface[:, 0, :], surface[0, 0, :]))
    outer_closed = np.vstack((surface[:, -1, :], surface[0, -1, :]))
    axis_a2.plot(inner_closed[:, 0], inner_closed[:, 1], color="black", lw=3.0)
    axis_a2.plot(outer_closed[:, 0], outer_closed[:, 1], color="0.60", lw=2.0)
    x_min, x_max = surface[:, :, 0].min(), surface[:, :, 0].max()
    y_min, y_max = surface[:, :, 1].min(), surface[:, :, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    scale_bar_length = cortical_coordinate_units_per_mm()
    scale_x = x_min + 0.08 * x_range
    scale_y = y_min - 0.04 * y_range
    axis_a2.plot(
        [scale_x, scale_x + scale_bar_length], [scale_y, scale_y],
        color="black", lw=2.2,
    )
    axis_a2.text(
        scale_x + 0.5 * scale_bar_length,
        scale_y - 0.035 * y_range,
        "1 mm",
        fontsize=10, ha="center", va="top",
    )
    axis_a2.set_xlim(x_min - 0.04 * x_range, x_max + 0.04 * x_range)
    axis_a2.set_ylim(y_min - 0.12 * y_range, y_max + 0.04 * y_range)
    axis_a2.set_aspect("equal")
    axis_a2.axis("off")

    # B/C — matched models that differ primarily in their smoothness constraint.
    panel_label(figure, "B", 0.000, 0.570)
    add_model_row(
        figure, 0.350,
        f"V2 model with high smoothness constraint: β1={high_parameters.get('b1', .001):g}",
        high_b1["fitted"], boundary, surface,
        field_sign_override=modern_audits["high"]["field_sign"] if modern_partition else None,
    )
    panel_label(figure, "C", 0.000, 0.270)
    add_model_row(
        figure, 0.050,
        f"V2 model: β1={primary_parameters.get('b1', .00014):g}",
        fitted, boundary, surface,
        field_sign_override=modern_audits["primary"]["field_sign"] if modern_partition else None,
    )

    # D — fitted retinotopy ribbon.
    panel_label(figure, "D", 0.475, 0.982)
    figure.text(
        0.560, 0.965, "Visual field mapping\nalong cortex",
        fontsize=12, fontweight="normal", ha="center", va="top",
    )
    # Use figure-level labels for the projected ground-plane axes.  Native 3D
    # labels reserve a disproportionately large margin at this camera angle.
    figure.text(
        0.500, 0.038, "Azimuth (°)",
        fontsize=9, rotation=-28, ha="center", va="center",
    )
    figure.text(
        0.605, 0.026, "Elevation (°)",
        fontsize=9, rotation=18, ha="center", va="center",
    )
    axis_d = figure.add_axes([0.455, 0.043, 0.195, 0.980], projection="3d")
    ribbon_index, ribbon = upsample_sheet(fitted_full, step=0.1)
    ribbon_z = ribbon_index / 2.0
    ribbon_colors = plt.cm.PuOr(np.linspace(0.0, 1.0, len(ribbon_index)))
    ground_boundary = np.vstack((boundary, boundary[0]))
    axis_d.plot(
        ground_boundary[:, 0], ground_boundary[:, 1],
        np.full(len(ground_boundary), ribbon_z.min()),
        color="black", lw=1.5, clip_on=False,
    )
    for index in range(len(ribbon_index)):
        axis_d.plot(
            ribbon[index, :, 0], ribbon[index, :, 1],
            np.full(ribbon.shape[1], ribbon_z[index]),
            color=ribbon_colors[-1 - index], lw=0.50, clip_on=False,
        )
    axis_d.plot(
        ribbon[:, 0, 0], ribbon[:, 0, 1], ribbon_z,
        color="black", lw=3.0, clip_on=False,
    )
    axis_d.plot(
        ribbon[:, -1, 0], ribbon[:, -1, 1], ribbon_z,
        color="0.60", lw=2.0, clip_on=False,
    )
    axis_d.set(xlabel="", ylabel="", zlabel="")
    # Orthographic projection avoids perspective distortion of the cortical
    # position axis while retaining a useful view of the ground plane.
    axis_d.set_proj_type("ortho")
    axis_d.view_init(elev=30.0, azim=-35.0)
    axis_d.tick_params(labelsize=8, width=0.6, length=2, pad=1)
    axis_d.set_xlim(0, 140)
    axis_d.set_ylim(-40, 60)
    axis_d.set_zlim(ribbon_z.min(), ribbon_z.max())
    # The V1 ground-plane outline supplies the spatial frame, so numeric
    # ground-plane ticks only add clutter and force excess bottom margin.
    axis_d.set_xticks(())
    axis_d.set_yticks(())
    axis_d.set_zticks(())
    # A short custom z spine marks the rear corner without extending into the
    # title area.  The native spine is hidden below after axis styling.
    x_padding = np.ptp(axis_d.get_xlim()) / 48.0
    y_padding = np.ptp(axis_d.get_ylim()) / 48.0
    z_padding = np.ptp(axis_d.get_zlim()) / 48.0
    rear_x = axis_d.get_xlim()[0] - x_padding
    rear_y = axis_d.get_ylim()[0] - y_padding
    rear_z_min = axis_d.get_zlim()[0] - z_padding
    rear_z_span = np.ptp(axis_d.get_zlim()) + 2.0 * z_padding
    axis_d.plot(
        (rear_x, rear_x), (rear_y, rear_y),
        (rear_z_min, rear_z_min + 0.80 * rear_z_span),
        color="0.55", lw=0.6, clip_on=False,
    )
    # Keep the complete 0--40 trajectory inside the axes clipping region.
    axis_d.set_box_aspect((0.587, 0.473, 3.70), zoom=3.14)
    axis_d.set_facecolor((1.0, 1.0, 1.0, 0.0))
    axis_d.grid(False)
    for axis_dimension in (axis_d.xaxis, axis_d.yaxis, axis_d.zaxis):
        axis_dimension.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis_dimension.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis_dimension.pane.set_linewidth(0.0)
        axis_dimension.line.set_color("0.55")
        axis_dimension.line.set_linewidth(0.6)
        axis_dimension._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)
    axis_d.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # E — field-sign territories and their visual-field coverage.
    panel_label(figure, "E", 0.650, 0.982)
    figure.text(
        0.688, 0.965, ("Garrett areas: splits at R > 1.01; no forced cut" if forced_row_cut is None else "Garrett areas: apex cut + splits at R > 1.01") if modern_partition else "Garrett rules split field-sign maps into areas",
        fontsize=11, fontweight="normal", va="top",
    )
    partition_full = fitted_full
    if modern_partition:
        from scripts.waters_organization_score import normalize_visual
        partition_full = normalize_visual(fitted_full, np.isfinite(fitted_full[:, 1:]).all(axis=2))
    parcellation = modern_audits["primary"] if modern_partition else garrett_area_parcellation(
        partition_full,
        field_sign_sigma_mm=0.10,
        field_sign_threshold=0.25,
        surface_units_per_mm=cortical_coordinate_units_per_mm(),
        min_patch_area_mm2=0.01,
        min_patch_nodes=3,
        cleanup_radius_mm=0.03,
        dilation_radius_mm=0.15,
        eccentricity_sigma_mm=0.15,
        eccentricity_bin_step_deg=5.0,
        visual_space_pixel_size_deg=0.5,
        visual_space_close_iterations=15,
        redundancy_threshold=1.01,
        merge_overlap_threshold=0.10,
        max_split_passes=8,
        cyclic_along=True,
        surface=surface,
        v1_interior_field_sign=-1.0,
        v1_border_field_sign=0.0,
        forced_row_cut=forced_row_cut if modern_partition else None,
        assignment_sign_threshold=.05 if modern_partition else None,
    )
    panel_e_field_sign = parcellation["field_sign"]
    labels = parcellation["labels"]
    centers = parcellation["centers"]
    area_count = len(centers)
    axis_e0 = figure.add_axes([0.655, 0.675, 0.145, 0.190])
    plot_cortical(
        axis_e0, surface, panel_e_field_sign, "Field sign",
        FIELD_SIGN_CMAP, TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
    )
    axis_e1 = figure.add_axes([0.810, 0.675, 0.145, 0.190])
    area_colors = plt.cm.tab20(np.linspace(0, 1, area_count))
    if modern_partition:
        for sign, palette in ((-1, plt.cm.Blues), (1, plt.cm.OrRd)):
            ids = np.flatnonzero(np.sign(centers[:, 2]) == sign)
            for index, value in zip(ids, np.linspace(.45, .90, len(ids))):
                area_colors[index] = palette(value)
    plot_cortical_areas(
        axis_e1, surface, labels, "Garrett-rule candidate areas", area_colors,
        node_regions=modern_partition,
    )
    label_cortical_areas(axis_e1, surface, labels, area_colors)
    if modern_partition and forced_row_cut is not None:
        seam = (surface[forced_row_cut - 1] + surface[forced_row_cut]) / 2
        axis_e1.plot(seam[:, 0], seam[:, 1], "k--", lw=1, zorder=10)
    mini_rows = max(2, int(np.ceil(area_count / 4)))
    mini_step = .20 / mini_rows
    mini_positions = [
        (0.655 + (index % 4) * 0.082, 0.485 + (mini_rows - 1 - index // 4) * mini_step)
        for index in range(area_count)
    ]
    for label, (x, y) in enumerate(mini_positions):
        axis = figure.add_axes([x, y, 0.073, min(.070, mini_step * .65)])
        mask = labels == label
        points = fitted_full[mask]
        color = area_colors[label]
        axis.scatter(points[:, 0], points[:, 1], s=4.0, color=color, linewidths=0)
        axis.plot(
            np.r_[boundary[:, 0], boundary[0, 0]],
            np.r_[boundary[:, 1], boundary[0, 1]], color="black", lw=0.67,
        )
        axis.set(xlim=(-4, 143), ylim=(-43, 63))
        axis.set_title(
            f"Area {label + 1}", fontsize=8.5, fontweight="bold",
            color=contrasting_text_color(color), pad=3,
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": color,
                "edgecolor": "black",
                "linewidth": 0.45,
            },
        )
        axis.set_aspect("equal", adjustable="box")
        axis.axis("off")

    # F - cortical V1 injections mapped through the warped Zhuang retinotopy.
    panel_label(figure, "F", 0.650, 0.445)
    plot_panel_f(figure, boundary, fitted_full, surface, labels)

    return figure


def save_figure(figure, output_stem: Path, formats=("svg", "pdf", "png"), bbox_inches=None):
    """Write a figure in the requested publication and review formats."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for output_format in formats:
        save_kwargs = {"facecolor": "white"}
        if output_format == "png":
            save_kwargs["dpi"] = 220
        if bbox_inches is not None:
            save_kwargs["bbox_inches"] = bbox_inches
            save_kwargs["pad_inches"] = 0
        figure.savefig(output_stem.with_suffix(f".{output_format}"), **save_kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-b1-run-dir", type=Path, required=True)
    parser.add_argument("--primary-run-dir", type=Path, required=True)
    parser.add_argument(
        "--surface-svg", type=Path,
        default=DEFAULT_BORDER_SVG,
    )
    parser.add_argument(
        "--output-stem", type=Path,
        default=PROJECT_ROOT / "artifacts" / "figure2_remake" / "figure2_updated",
    )
    args = parser.parse_args()
    figure = build_figure(
        args.high_b1_run_dir.resolve(), args.primary_run_dir.resolve(),
        args.surface_svg.resolve(),
    )
    save_figure(figure, args.output_stem.resolve())
    plt.close(figure)


if __name__ == "__main__":
    main()
