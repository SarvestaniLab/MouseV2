"""Publication layout preserving the selected full-legacy F/G analysis.

Run from the repository root: python figure3/refine_panels_fg.py
"""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np

from panels_fg import (
    AREA_ORDER, METRICS, _draw_violin, _physical_tick_labels,
    analyze_metric, load_cells, write_analysis_summary,
)

HERE = Path(__file__).resolve().parent


def main():
    source = HERE / "data/han2022_figure3fg_legacy_registration.csv"
    output = HERE / "output"
    output.mkdir(exist_ok=True)
    stem = output / "figure3_fg_full_legacy_refined"
    cells = load_cells(source)
    analyses = {key: analyze_metric(cells, key, fit_method="legacy")
                for key in METRICS}
    with plt.rc_context({
        "font.family": "Arial", "font.size": 8,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "axes.linewidth": 0.7, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "svg.fonttype": "none", "svg.hashsalt": "figure3-full-legacy",
        "savefig.facecolor": "white",
    }):
        fig = plt.figure(figsize=(11.2, 6.4))
        grid = fig.add_gridspec(2, 4, width_ratios=(1.15, 2, 0.07, 1.15),
                               left=0.075, right=0.985, top=0.94,
                               bottom=0.15, wspace=0.45, hspace=0.48)
        for row, (metric, analysis) in enumerate(analyses.items()):
            settings = METRICS[metric]
            base = float(settings["base"])
            unit = str(settings["unit"])
            left, middle, cax, right = [fig.add_subplot(grid[row, col])
                                       for col in range(4)]
            _draw_violin(left, analysis.uncensored_cells, metric,
                         f"Preferred {str(settings['title']).lower()}\n({unit})")
            ticks, labels = _physical_tick_labels(base, analysis.upper_bound)
            left.set_yticks(ticks, labels)
            left.set_ylim(analysis.lower_bound, analysis.upper_bound)
            left.set_title("Tuning by area", pad=10)
            left.text(-0.32, 1.13, settings["panel"], transform=left.transAxes,
                      fontsize=16, weight="bold", va="top")

            all_cells = analysis.all_cells
            xmin, xmax = np.percentile(all_cells.eccentricity_deg, (0.5, 99.5))
            ymin, ymax = np.percentile(all_cells.elevation_deg, (0.5, 99.5))
            x = np.linspace(xmin, xmax, 1201)
            y = np.linspace(ymin, ymax, 801)
            xx, yy = np.meshgrid((x[:-1] + x[1:]) / 2,
                                 (y[:-1] + y[1:]) / 2)
            values = base * 2 ** analysis.fit.predict(xx, yy)
            field = middle.imshow(values, origin="lower", aspect="auto",
                                  extent=(xmin, xmax, ymin, ymax),
                                  cmap="viridis", interpolation="none")
            middle.scatter(all_cells.eccentricity_deg, all_cells.elevation_deg,
                           s=0.45, color="white", alpha=0.60, linewidths=0)
            for area in AREA_ORDER:
                selected = analysis.uncensored_cells.query("area == @area")
                xm, ym = selected[["eccentricity_deg", "elevation_deg"]].mean()
                middle.scatter(xm, ym, s=27, facecolor="white",
                               edgecolor="black", linewidth=0.8, zorder=4)
                middle.annotate(area, (xm, ym), xytext=(5, 4),
                                textcoords="offset points", fontsize=10.5,
                                path_effects=[pe.withStroke(linewidth=2.0,
                                                           foreground="white", alpha=0.75)],
                                zorder=5)
            middle.set(xlim=(xmin, xmax), ylim=(ymin, ymax),
                       xlabel="Eccentricity from focea (°)", ylabel="Elevation (°)",
                       title="Retinotopic plane fit")
            middle.title.set_y(1.02)
            cb = fig.colorbar(field, cax=cax)
            cax.set_title("cycles/\ndegree" if row == 0 else unit,
                          fontsize=7, pad=7)
            cb.solids.set_rasterized(True)

            _draw_violin(right, analysis.uncensored_cells, "residual_octaves",
                         "Residual (octaves)")
            limit = (analysis.upper_bound - analysis.lower_bound) / 2
            right.set_ylim(-limit, limit)
            right.set_yticks(np.arange(-limit, limit + 1, 1))
            right.axhline(0, color="0.5", linestyle=(0, (3, 3)),
                          linewidth=0.7, zorder=0)
            right.set_title("After retinotopic subtraction", pad=10)
        handles = [
            Line2D([], [], marker="o", linestyle="none", color="0.72",
                   markeredgecolor="black", markeredgewidth=0.35,
                   markersize=4, label="Mouse mean"),
            Line2D([], [], marker="o", color="black", markerfacecolor="white",
                   markersize=5, linewidth=1, label="Across-mouse mean and 95% CI"),
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.04),
                   ncol=2, frameon=False, fontsize=8, handlelength=1.5)
        fig.text(0.53, 0.023,
                 "Violins: individual cells   •   n: mice   •   Focea: azimuth 0°, elevation +20°",
                 ha="center", fontsize=8, color="0.25")
        for suffix in (".png", ".svg"):
            fig.savefig(stem.with_suffix(suffix), dpi=300)
        plt.close(fig)
    write_analysis_summary(analyses, Path(str(stem) + "_summary.json"))
    manifest = {
        "registration": "legacy", "fit_method": "legacy",
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "analysis_target": "selected manuscript Figure 3F/G full-legacy workflow",
        "svg": ("Single 1200 x 800 embedded image per plane background; "
                "raster colorbar fills; vector overlays, axes, and violins"),
        "outputs": {suffix: hashlib.sha256(stem.with_suffix(suffix).read_bytes()).hexdigest()
                    for suffix in (".png", ".svg")},
    }
    Path(str(stem) + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()
