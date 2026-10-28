"""Shared paths, arguments, and layout for Figure 2 exports."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from matplotlib.transforms import Bbox


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "elastictopy"))

from scripts.verify_figure2_freeze import verify as verify_figure2_freeze  # noqa: E402
from scripts.make_figure2_updated import save_figure
from scripts.make_figure2_manuscript import build_manuscript_figure  # noqa: E402


DEFAULT_HIGH_B1_RUN = (
    PROJECT_ROOT / "data" / "publication" / "figure2" / "high_smoothness"
)
DEFAULT_PRIMARY_RUN = (
    PROJECT_ROOT / "data" / "publication" / "figure2" / "primary"
)
# Pin the renderer's current geometry independently of future model defaults.
# The manuscript runs used this same individual-V1 training geometry.
DEFAULT_SURFACE_SVG = (
    PROJECT_ROOT / "data" / "waters2019" / "derived"
    / "WatersV1_IndividualV1Median_borders.svg"
)

# Normalized crops in the 16 x 9.9 inch full composition.
PANEL_BOUNDS = {
    "A": (0.000, 0.585, 0.480, 0.415),
    "B": (0.000, 0.330, 0.470, 0.255),
    "C": (0.000, 0.030, 0.470, 0.255),
    # Include the complete 3D axes (x=0.455..0.650) plus a small safety margin.
    # The standalone renderer hides neighboring panels while exporting D.
    "D": (0.438, 0.000, 0.225, 1.000),
    "E": (0.650, 0.475, 0.350, 0.525),
    "F": (0.650, 0.000, 0.350, 0.485),
}


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--high-b1-run-dir", type=Path, default=DEFAULT_HIGH_B1_RUN)
    parser.add_argument("--primary-run-dir", type=Path, default=DEFAULT_PRIMARY_RUN)
    parser.add_argument("--surface-svg", type=Path, default=DEFAULT_SURFACE_SVG)


def build_from_args(args):
    if (args.high_b1_run_dir.resolve() == DEFAULT_HIGH_B1_RUN.resolve()
            or args.primary_run_dir.resolve() == DEFAULT_PRIMARY_RUN.resolve()):
        verify_figure2_freeze(PROJECT_ROOT)
    return build_manuscript_figure(
        args.high_b1_run_dir.resolve(),
        args.primary_run_dir.resolve(),
        args.surface_svg.resolve(),
    )


def panel_bbox_inches(figure, panel: str) -> Bbox:
    x, y, width, height = PANEL_BOUNDS[panel]
    figure_width, figure_height = figure.get_size_inches()
    return Bbox.from_bounds(
        x * figure_width,
        y * figure_height,
        width * figure_width,
        height * figure_height,
    )


__all__ = [
    "PANEL_BOUNDS",
    "PROJECT_ROOT",
    "add_input_arguments",
    "build_from_args",
    "panel_bbox_inches",
    "save_figure",
]
