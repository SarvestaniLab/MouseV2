"""Render the complete Figure 2 composition as SVG, PDF, and PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from fig2.common import PROJECT_ROOT, add_input_arguments, build_from_args, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=PROJECT_ROOT / "fig2" / "output" / "full" / "fig2",
    )
    args = parser.parse_args()

    figure = build_from_args(args)
    save_figure(figure, args.output_stem.resolve())
    plt.close(figure)


if __name__ == "__main__":
    main()
