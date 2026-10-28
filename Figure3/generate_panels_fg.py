"""Command-line entry point for Figure 3 panels F and G."""

import argparse
from pathlib import Path

from panels_fg import load_cells, plot_panels_fg, write_analysis_summary


HERE = Path(__file__).resolve().parent
REGISTRATION_INPUTS = {
    "legacy": HERE / "data" / "han2022_figure3fg_legacy_registration.csv",
    "objective_equal_mouse": (
        HERE / "data" / "han2022_figure3fg_objective_equal_mouse.csv"
    ),
    "objective_equal_cell": (
        HERE / "data" / "han2022_figure3fg_objective_equal_cell.csv"
    ),
    "objective_spatial_bin": (
        HERE / "data" / "han2022_figure3fg_objective_spatial_bin.csv"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the Figure 3F/G analysis in Python."
    )
    parser.add_argument(
        "--registration", choices=tuple(REGISTRATION_INPUTS),
        default="legacy",
        help=("Manual legacy registration (default while objective modes are "
              "under visual review), or one of three objective weightings."),
    )
    parser.add_argument(
        "--input", type=Path,
        help="Override the table selected by --registration.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "output" / "figure3_fg.png",
    )
    parser.add_argument(
        "--summary", type=Path,
        default=HERE / "output" / "figure3_fg_summary.json",
    )
    parser.add_argument(
        "--fit-method", choices=("tobit", "legacy"), default="tobit",
        help=("Censored-normal maximum likelihood (default), or the submitted "
              "MATLAB equations for historical reproduction."),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input or REGISTRATION_INPUTS[args.registration]
    cells = load_cells(input_path)
    analyses = plot_panels_fg(cells, args.output, fit_method=args.fit_method)
    write_analysis_summary(analyses, args.summary)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
