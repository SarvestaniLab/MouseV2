"""Build the mouse V2 surface grid used by the elastic-net regularizer.

The default source is the Waters individual-V1, detection-conditional boundary.
The previous Zhuang tracing remains available explicitly for legacy parity.
Distance calculation is kept separate and accepts coordinates of any
dimensionality, so a future 3D atlas surface can use the same interface.
"""

from functools import lru_cache
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ZHUANG_BORDER_SVG = PROJECT_ROOT / "data" / "ZhuangTracing9_borders.svg"
WATERS_CONDITIONAL_BORDER_SVG = (
    PROJECT_ROOT
    / "data"
    / "waters2019"
    / "derived"
    / "WatersV1_NormalConditionalMedian_borders.svg"
)
WATERS_REGISTERED_MOUSE_MEDIAN_BORDER_SVG = (
    WATERS_CONDITIONAL_BORDER_SVG.parent
    / "WatersV1_RegisteredMouseMedian_borders.svg"
)
WATERS_INDIVIDUAL_V1_BORDER_SVG = (
    WATERS_CONDITIONAL_BORDER_SVG.parent
    / "WatersV1_IndividualV1Median_borders.svg"
)
DEFAULT_BORDER_SVG = WATERS_INDIVIDUAL_V1_BORDER_SVG

_TOKEN_RE = re.compile(
    r"[MmLlHhVvCcSsZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
)


def _sample_line(start, end, samples):
    t = np.linspace(0.0, 1.0, samples)[:, None]
    return (1.0 - t) * start + t * end


def _sample_cubic(start, control1, control2, end, samples):
    t = np.linspace(0.0, 1.0, samples)[:, None]
    return (
        (1.0 - t) ** 3 * start
        + 3.0 * (1.0 - t) ** 2 * t * control1
        + 3.0 * (1.0 - t) * t**2 * control2
        + t**3 * end
    )


def sample_svg_path(path_data, step=0.001):
    """Sample the M/L/H/V/C/S subset used by the legacy border tracing."""
    if not 0.0 < step <= 1.0:
        raise ValueError("step must be in (0, 1]")
    samples = int(round(1.0 / step)) + 1
    tokens = _TOKEN_RE.findall(path_data.replace(",", " "))
    points = []
    current = np.zeros(2, dtype=np.float64)
    subpath_start = current.copy()
    previous_control = None
    command = None
    i = 0

    def number(index):
        return float(tokens[index])

    while i < len(tokens):
        if tokens[i].isalpha():
            command = tokens[i]
            i += 1
        if command is None:
            raise ValueError("SVG path data must begin with a command")

        relative = command.islower()
        operation = command.upper()
        if operation == "M":
            target = np.array([number(i), number(i + 1)])
            i += 2
            if relative:
                target += current
            current = target
            subpath_start = current.copy()
            points.append(current[None, :])
            command = "l" if relative else "L"
            previous_control = None
        elif operation == "L":
            target = np.array([number(i), number(i + 1)])
            i += 2
            if relative:
                target += current
            points.append(_sample_line(current, target, samples))
            current = target
            previous_control = None
        elif operation == "H":
            target = current.copy()
            target[0] = current[0] + number(i) if relative else number(i)
            i += 1
            points.append(_sample_line(current, target, samples))
            current = target
            previous_control = None
        elif operation == "V":
            target = current.copy()
            target[1] = current[1] + number(i) if relative else number(i)
            i += 1
            points.append(_sample_line(current, target, samples))
            current = target
            previous_control = None
        elif operation == "C":
            values = np.array([number(i + offset) for offset in range(6)]).reshape(3, 2)
            i += 6
            if relative:
                values += current
            control1, control2, target = values
            points.append(
                _sample_cubic(current, control1, control2, target, samples)
            )
            current = target
            previous_control = control2
        elif operation == "S":
            values = np.array([number(i + offset) for offset in range(4)]).reshape(2, 2)
            i += 4
            if relative:
                values += current
            control2, target = values
            control1 = (
                2.0 * current - previous_control
                if previous_control is not None
                else current.copy()
            )
            points.append(
                _sample_cubic(current, control1, control2, target, samples)
            )
            current = target
            previous_control = control2
        elif operation == "Z":
            points.append(_sample_line(current, subpath_start, samples))
            current = subpath_start.copy()
            previous_control = None
            command = None
        else:
            raise ValueError(f"Unsupported SVG path command: {command}")

    return np.concatenate(points, axis=0)


def load_svg_paths(svg_path=DEFAULT_BORDER_SVG, step=0.001):
    """Return sampled paths in their SVG document order."""
    root = ET.parse(svg_path).getroot()
    path_data = [
        element.attrib["d"]
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "path" and "d" in element.attrib
    ]
    if len(path_data) != 2:
        raise ValueError(f"Expected exactly two border paths, found {len(path_data)}")
    return [sample_svg_path(data, step=step) for data in path_data]


def resample_nearest_by_arclength(points, count, endpoint=True):
    """Sample a traced path at approximately uniform arclength positions.

    ``endpoint=False`` is used for cyclic training grids so the closing point
    is represented by row zero rather than by a duplicated final row.
    """
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    targets = np.linspace(0.0, cumulative[-1], count, endpoint=endpoint)
    indices = np.abs(cumulative[:, None] - targets[None, :]).argmin(axis=0)
    return points[indices]


def resample_closed_values(values, count):
    """Return ``count`` unique periodic samples from an endpoint-closed trace."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 3:
        raise ValueError("values must contain at least three 2D samples")
    if count < 2:
        raise ValueError("count must be at least two")
    if not np.allclose(values[0], values[-1]):
        raise ValueError("periodic resampling requires matching endpoints")
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, count, endpoint=False)
    return np.column_stack(
        [np.interp(target, source, values[:, axis]) for axis in range(values.shape[1])]
    )


def build_surface_grid(
    inner_border, outer_border, n_along=80, n_across=9, cyclic_along=False
):
    """Interpolate a regular grid between corresponding border samples."""
    inner = resample_nearest_by_arclength(
        inner_border, n_along, endpoint=not cyclic_along
    )
    outer = resample_nearest_by_arclength(
        outer_border, n_along, endpoint=not cyclic_along
    )
    fraction = np.linspace(0.0, 1.0, n_across)[None, :, None]
    return inner[:, None, :] + fraction * (outer - inner)[:, None, :]


def outward_normal_intersections(inner, outer):
    """Intersect outward normals from sampled inner nodes with an outer contour."""
    inner = np.asarray(inner, dtype=np.float64)
    outer = np.asarray(outer, dtype=np.float64)
    tangent = np.roll(inner, -1, axis=0) - np.roll(inner, 1, axis=0)
    normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0.0):
        raise ValueError("inner border contains a degenerate tangent")
    normals /= lengths[:, None]
    center = inner.mean(axis=0)
    inward = np.sum(normals * (center - inner), axis=1) > 0.0
    normals[inward] *= -1.0

    if not np.allclose(outer[0], outer[-1]):
        outer = np.vstack((outer, outer[0]))
    segment_start = outer[:-1]
    segment_vector = outer[1:] - outer[:-1]

    def cross(left, right):
        return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]

    intersections = np.empty_like(inner)
    for index, (point, normal) in enumerate(zip(inner, normals)):
        denominator = cross(normal, segment_vector)
        offset = segment_start - point
        usable = np.abs(denominator) > 1e-12
        ray_distance = np.full(len(segment_start), np.nan)
        segment_fraction = np.full(len(segment_start), np.nan)
        ray_distance[usable] = (
            cross(offset[usable], segment_vector[usable]) / denominator[usable]
        )
        segment_fraction[usable] = (
            cross(offset[usable], normal) / denominator[usable]
        )
        hits = (
            usable
            & (ray_distance >= -1e-10)
            & (segment_fraction >= -1e-10)
            & (segment_fraction <= 1.0 + 1e-10)
        )
        if not hits.any():
            raise ValueError(
                "outward normal from inner-border row {} does not intersect "
                "the outer contour".format(index)
            )
        distance = max(0.0, float(np.nanmin(ray_distance[hits])))
        intersections[index] = point + distance * normal
    return intersections, normals


def build_normal_surface_grid(
    inner_border, outer_border, n_along=80, n_across=9, cyclic_along=False
):
    """Build across-border columns on true outward normals from V1."""
    inner = resample_nearest_by_arclength(
        inner_border, n_along, endpoint=not cyclic_along
    )
    outer, _ = outward_normal_intersections(inner, outer_border)
    fraction = np.linspace(0.0, 1.0, n_across)[None, :, None]
    return inner[:, None, :] + fraction * (outer - inner)[:, None, :]


def surface_grid_distances(surface, cyclic_along=False):
    """Calculate grid-edge distances using all 2D or 3D coordinate axes."""
    surface = np.asarray(surface, dtype=np.float64)
    if surface.ndim != 3 or surface.shape[1] < 2:
        raise ValueError("surface must have shape (along, across + 1, dimensions)")
    height, across_points, _ = surface.shape
    width = across_points - 1

    along = np.full((height, width), np.nan)
    along[:-1] = np.linalg.norm(surface[1:, :width] - surface[:-1, :width], axis=2)
    if cyclic_along:
        along[-1] = np.linalg.norm(surface[0, :width] - surface[-1, :width], axis=1)
    across = np.linalg.norm(surface[:, 1:] - surface[:, :-1], axis=2)

    top_right = np.full((height, width), np.nan)
    top_right[1:] = np.linalg.norm(
        surface[:-1, 1:] - surface[1:, :-1], axis=2
    )
    lower_right = np.full((height, width), np.nan)
    lower_right[:-1] = np.linalg.norm(
        surface[1:, 1:] - surface[:-1, :-1], axis=2
    )
    if cyclic_along:
        top_right[0] = np.linalg.norm(
            surface[-1, 1:] - surface[0, :-1], axis=1
        )
        lower_right[-1] = np.linalg.norm(
            surface[0, 1:] - surface[-1, :-1], axis=1
        )
    return {
        "griddist_along": along,
        "griddist_across": across,
        "griddist_topright": top_right,
        "griddist_lowerright": lower_right,
    }


@lru_cache(maxsize=None)
def generate_mouse_surface_grid(
    svg_path=DEFAULT_BORDER_SVG, map_h=80, map_w=8, step=0.001,
    cyclic_along=True, normal_across=None,
):
    """Reconstruct the cortical sheet from the selected two-path SVG.

    The default returns distinct along-border rows for cyclic elastic-net
    training. Set ``cyclic_along=False`` only for historical endpoint-duplicated
    comparisons.
    """
    svg_path = Path(svg_path)
    outer_path, inner_path = load_svg_paths(svg_path, step=step)
    if normal_across is None:
        normal_across = svg_path.resolve() in {
            WATERS_CONDITIONAL_BORDER_SVG.resolve(),
            WATERS_REGISTERED_MOUSE_MEDIAN_BORDER_SVG.resolve(),
            WATERS_INDIVIDUAL_V1_BORDER_SVG.resolve(),
        }
    # Loadup_borders.m reverses both paths before arclength sampling.
    builder = build_normal_surface_grid if normal_across else build_surface_grid
    return builder(
        inner_path[::-1], outer_path[::-1], n_along=map_h, n_across=map_w + 1,
        cyclic_along=cyclic_along,
    )


@lru_cache(maxsize=None)
def generate_mouse_surface_distances(
    svg_path=DEFAULT_BORDER_SVG, map_h=80, map_w=8, step=0.001,
    cyclic_along=True, normal_across=None,
):
    """Calculate cortical grid distances, including the cyclic seam by default."""
    return surface_grid_distances(
        generate_mouse_surface_grid(
            svg_path, map_h, map_w, step, cyclic_along, normal_across
        ),
        cyclic_along=cyclic_along,
    )
