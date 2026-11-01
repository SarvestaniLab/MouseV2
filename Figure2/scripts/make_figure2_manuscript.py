"""Reproduce the Figure 2 manuscript layout from the frozen seed-13 maps.

Presentation reconstructed against page 6 of 2026_10_01_biorxiv_DPR.pdf.
The original saved ribbon interpolation is retained, without later redraws.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "elastictopy"))
import json
import numpy as np
import matplotlib.pyplot as plt
from scripts.make_figure2_updated import build_figure, latest_checkpoint

def verify_pair(high_run, primary_run):
    manifests = [json.loads((run / "run-manifest.json").read_text())
                 for run in (high_run, primary_run)]
    high, primary = manifests
    for key in ("seed", "b2", "b3", "parameters", "source_sha256"):
        if high[key] != primary[key]:
            raise ValueError(f"Comparator and primary differ in {key}.")
    for filename in ("init.csv", "vf.csv", "boundary.csv"):
        np.testing.assert_array_equal(np.loadtxt(high_run / filename, delimiter=","),
                                      np.loadtxt(primary_run / filename, delimiter=","))
    for run, manifest in zip((high_run, primary_run), manifests):
        if latest_checkpoint(run).name != manifest["final_checkpoint"]:
            raise ValueError(f"Latest checkpoint differs from manifest in {run}.")
    return manifests

def place(axis, bounds):
    # Colorbars otherwise retain a locator tied to the old parent rectangle.
    axis.set_axes_locator(None)
    axis.set_position(bounds)

def compact_layout(figure):
    """Restyle the shared renderer without changing maps, partitions or data."""
    axes = figure.axes
    # Shared renderer: A(2), B(4+3 bars), C(4+3 bars), D(1), E(2+N), F(2).
    if len(axes) < 23 or axes[16].name != "3d":
        raise ValueError("Shared Figure 2 axes changed; update compact_layout.")
    figure.set_size_inches(15.64, 10.38)
    letters = {"A": (.010, .988), "B": (.010, .536), "C": (.010, .280),
               "D": (.461, .988), "E": (.647, .988), "F": (.647, .372)}
    for text in list(figure.texts):
        label = text.get_text()
        if label in letters:
            text.set_position(letters[label])
            text.set_fontsize(44)
        elif label.startswith("Visual field and cortical"):
            text.set_position((.240, .950))
            text.set_fontsize(18)
        elif label.startswith("V2 model with high smoothness"):
            text.set_text("High smoothness constraint: " + label.split(": ", 1)[1])
            text.set_position((.275, .495))
            text.set_fontsize(18)
        elif label.startswith("V2 model:"):
            text.set_text("Low smoothness constraint: " + label.split(": ", 1)[1])
            text.set_position((.275, .250))
            text.set_fontsize(18)
        elif label.startswith("Garrett areas:"):
            text.set_text("Garrett areas: splits at R > 1.01")
            text.set_position((.695, .950))
            text.set_fontsize(16)
        elif label.startswith(("Visual field mapping", "V1 injections predict")):
            text.remove()
        elif label.startswith("Azimuth"):
            text.set_position((.524, .037))
            text.set_text("Azimuth")
            text.set_fontsize(13)
        elif label.startswith("Elevation"):
            text.set_position((.599, .026))
            text.set_text("Elevation")
            text.set_fontsize(13)

    place(axes[0], [.068, .637, .199, .217])
    axes[0].tick_params(labelsize=15)
    axes[0].xaxis.label.set_size(18)
    axes[0].yaxis.label.set_size(18)
    place(axes[1], [.307, .596, .153, .300])
    for text in axes[1].texts:
        text.set_fontsize(16)

    for start, y in ((2, .318), (9, .066)):
        row = axes[start:start + 7]
        for axis, x, width in zip(row[:4], (.045, .164, .270, .377),
                                  (.106, .084, .084, .084)):
            place(axis, [x, y, width, .145])
            axis.set_title("")
        row[0].tick_params(labelbottom=False, labelleft=False)
        row[0].xaxis.label.set_size(11)
        row[0].yaxis.label.set_size(11)
        for bar, x in zip(row[4:], (.251, .358, .464)):
            place(bar, [x, y + .025, .0045, .100])
            bar.tick_params(labelsize=9, length=0, pad=2)
            for spine in bar.spines.values():
                spine.set_visible(False)
    for x, label in zip((.209, .316, .420), ("Eccentricity", "Visual angle", "Field sign")):
        figure.text(x, .044, label, ha="center", fontsize=18)

    axis_d = axes[16]
    place(axis_d, [.475, .045, .162, .920])
    axis_d.set_box_aspect((.587, .473, 3.70), zoom=3.78)
    # Restore the enclosing 3D frame visible in the reference. Remove only
    # the previous custom rear spine, leaving the scientific ribbon intact.
    axis_d.lines[-1].remove()
    x_limits, y_limits, z_limits = axis_d.get_xlim(), axis_d.get_ylim(), axis_d.get_zlim()
    rear_x = x_limits[0] - np.ptp(x_limits) / 48
    rear_y = y_limits[0] - np.ptp(y_limits) / 48
    z_padding = np.ptp(z_limits) / 48
    axis_d.plot((rear_x, rear_x), (rear_y, rear_y),
                (z_limits[0] - z_padding, z_limits[1] + z_padding),
                color="0.55", lw=.8, clip_on=False)
    for dimension in (axis_d.xaxis, axis_d.yaxis, axis_d.zaxis):
        dimension.pane.set_edgecolor("0.55")
        dimension.pane.set_linewidth(.8)
        dimension.pane.set_alpha(1)
        dimension.pane.set_facecolor((1, 1, 1, 0))
        dimension.line.set_color("0.55")

    place(axes[17], [.671, .653, .127, .210])
    place(axes[18], [.845, .653, .127, .210])
    for axis in axes[17:19]:
        axis.set_title(axis.get_title(), fontsize=16, pad=6)
    area_axis = axes[18]
    # Move IDs radially outside the outer border, preserving each area's color.
    inner = np.column_stack(area_axis.lines[0].get_data())
    outer = np.column_stack(area_axis.lines[1].get_data())
    center = inner.mean(axis=0)
    footprints = axes[19:-2]
    area_colors = [axis.collections[0].get_facecolors()[0] for axis in footprints]
    for text, color in zip(area_axis.texts, area_colors):
        point = np.asarray(text.get_position())
        direction = point - center
        directions = outer - center
        cosine = directions @ direction / np.maximum(np.linalg.norm(directions, axis=1), 1e-12)
        edge = outer[np.argmax(cosine)]
        text.set_position(center + 1.12 * (edge - center))
        text.set_bbox(None)
        text.set_color(color)
        text.set_fontsize(14)
        text.set_fontweight("normal")
        text.set_clip_on(False)
    # Eight areas use the reference's two rows; additional areas expand upward
    # within the same footprint block rather than disappearing off the canvas.
    rows = max(2, int(np.ceil(len(footprints) / 4)))
    step = .235 / rows
    for i, (axis, color) in enumerate(zip(footprints, area_colors)):
        place(axis, [.662 + (i % 4) * .082, .410 + (rows - 1 - i // 4) * step,
                     .073, min(.080, step * .68)])
        axis.title.set_bbox(None)
        axis.set_title(axis.get_title(), fontsize=19, color=color,
                       fontweight="normal", pad=5)

    rf, cortex = axes[-2:]
    place(rf, [.681, .059, .148, .175])
    place(cortex, [.840, .022, .151, .270])
    rf.set_title("Retinotopic coverage\nfrom V1 injections", fontsize=18, pad=16)
    cortex.set_title("V1 injections &\naxonal projections", fontsize=18, pad=10)
    rf.tick_params(labelsize=10)
    rf.xaxis.label.set_size(11)
    rf.yaxis.label.set_size(10)
    for text in list(cortex.texts):
        if text.get_text().startswith("HVA") or text.get_text() == "V1":
            text.remove()
        elif text.get_text() == "0.5 mm":
            text.set_fontsize(10)
    return figure


def build_manuscript_figure(high_run, primary_run, surface_svg):
    high_run, primary_run = Path(high_run), Path(primary_run)
    verify_pair(high_run, primary_run)
    figure = build_figure(high_run, primary_run, Path(surface_svg),
                          modern_partition=True, forced_row_cut=None)
    compact_layout(figure)
    # Manuscript headings and axis labels, before later explanatory redesigns.
    for text in list(figure.texts):
        if text.get_text().startswith("Garrett areas:"):
            text.remove()
    figure.axes[18].set_title("Candidate areas", fontsize=16, pad=6)
    for index in (2, 9):
        figure.axes[index].set_xlabel("Azimuth", fontsize=16)
        figure.axes[index].set_ylabel("Elevation", fontsize=16)
    # Positions measured from the 1528 x 1010 image embedded on PDF page 6.
    figure.set_size_inches(15.28, 10.10)
    def pixels(axis, left, top, width, height):
        place(axis, [left / 1528, 1 - (top + height) / 1010,
                     width / 1528, height / 1010])
    pixels(figure.axes[0], 118, 158, 296, 212)
    pixels(figure.axes[1], 466, 130, 244, 268)
    for start, top in ((2, 546), (9, 795)):
        pixels(figure.axes[start], 84, top + 11, 158, 115)
        for axis, left in zip(figure.axes[start + 1:start + 4], (260, 417, 577)):
            pixels(axis, left, top, 127, 137)
        for axis, left in zip(figure.axes[start + 4:start + 7], (392, 551, 709)):
            pixels(axis, left, top + 17, 4.5, 100)
    pixels(figure.axes[17], 1010, 92, 200, 218)
    pixels(figure.axes[18], 1270, 92, 200, 218)
    for i, axis in enumerate(figure.axes[19:-2]):
        pixels(axis, 1005 + (i % 4) * 122, 367 + (i // 4) * 117, 104, 75)
    pixels(figure.axes[-2], 1030, 743, 220, 159)
    pixels(figure.axes[-1], 1275, 718, 208, 225)
    positions = {"A": (30, 28), "B": (30, 455), "C": (30, 708),
                 "D": (700, 28), "E": (960, 28), "F": (960, 604)}
    for text in figure.texts:
        label = text.get_text()
        if label in positions:
            x, y = positions[label]
            text.set_position((x / 1528, 1 - y / 1010))
        elif label.startswith("Visual field and cortical"):
            text.set_position((380 / 1528, 1 - 60 / 1010))
            text.set_fontsize(20)
        elif label.startswith("High smoothness"):
            text.set_position((378 / 1528, 1 - 495 / 1010))
            text.set_va("top")
            text.set_fontsize(20)
        elif label.startswith("Low smoothness"):
            text.set_position((378 / 1528, 1 - 736 / 1010))
            text.set_va("top")
            text.set_fontsize(20)
        elif label in ("Eccentricity", "Visual angle", "Field sign"):
            x = {"Eccentricity": 324, "Visual angle": 482, "Field sign": 641}[label]
            text.set_position((x / 1528, 1 - 956 / 1010))
    figure.axes[16].set_box_aspect((.587, .473, 3.70), zoom=3.68)
    for axis in figure.axes[17:19]:
        axis.title.set_fontsize(20)
    figure.axes[-2].xaxis.label.set_size(16)
    figure.axes[-2].yaxis.label.set_size(16)
    figure.axes[-2].set_title("Retinotopic coverage\nfrom V1 injections", fontsize=18, pad=24)
    figure.axes[-1].set_title("V1 injections &\naxonal projections", fontsize=18, pad=18)
    for text in figure.texts:
        if text.get_text() in positions:
            text.set_fontsize(48)
        elif text.get_text() == "Azimuth":
            text.set_position((780 / 1528, 1 - 930 / 1010))
            text.set_rotation(-40)
            text.set_fontsize(16)
        elif text.get_text() == "Elevation":
            text.set_position((885 / 1528, 1 - 951 / 1010))
            text.set_fontsize(16)
    # Arial is used in the draft; Liberation Sans is its portable metric-compatible fallback.
    from matplotlib.text import Text
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = "Arial" if "Arial" in available else "Liberation Sans"
    for text in figure.findobj(Text):
        text.set_fontfamily(font)
    return figure


if __name__ == "__main__":
    from fig2.compose_figure import main
    main()
