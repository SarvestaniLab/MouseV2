"""Render panels A/B from frozen fitted maps and the saved partition field."""

from pathlib import Path
import os, json
import numpy as np
import matplotlib

OUT = Path(__file__).resolve().parents[1] / "figure2_compositions/clean_export"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl_cache")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap

ROOT = OUT.parents[1]
BASE = ROOT / "data"
d = np.load(BASE / "figure2_data.npz")
meta = json.loads((BASE / "figure2_shared_model.json").read_text())
partition_field = np.load(BASE / "panel_B_partition_field.npz")
np.testing.assert_array_equal(partition_field["labels"], d["labels"])
INK = "#29333a"
MUTED = "#76838b"
TEAL = "#407b8b"
GRID = "#c9d3d7"
LAND = ["#00b5ad", "#f06489", "#efb51b"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig = plt.figure(figsize=(12, 430.5 / 72))


def text(x, y, t, size=8, **kw):
    fig.text(x, y, t, fontsize=size, **kw)


def heading(letter, x, title):
    text(x, 1 - 26.44 / 430.5, letter, 22, weight="bold")
    if letter == "A":
        text(
            x + 31 / 864, 1 - 26.44 / 430.5, "V1 provides the retinotopic reference", 16
        )
    else:
        text(x + 31 / 864, 1 - 26.44 / 430.5, title, 16)


def close(p):
    return np.vstack((p, p[0]))


def faces(shape, cyclic=False):
    h, w = shape
    f = []
    for i in range(h if cyclic else h - 1):
        for j in range(w - 1):
            a = i * w + j
            b = ((i + 1) % h) * w + j
            f.extend([[a, b, b + 1], [a, b + 1, a + 1]])
    return np.array(f)


heading("A", 0.025, "V1 has a simple 1-to-1 mapping from visual field to cortex.")
heading("B", 0.55, "Applying the same model beyond V1\u2026")
# Match HVA light display smoothing, adapted to V1's four open borders.
from scipy.ndimage import gaussian_filter

raw_v1 = d["panel_b_rf_v1"]
v1_visual_display = gaussian_filter(
    raw_v1, sigma=(3, 1.5, 0), mode=("reflect", "reflect", "nearest")
)
# Exponentially tapered boundary correction, including corner double-count removal.
err = raw_v1 - v1_visual_display


def edge_weights(n):
    t = np.arange(n, dtype=float)
    a = np.exp(-t / 6) * (1 - np.exp(-(n - 1 - t) / 6)) / (1 - np.exp(-(n - 1) / 6))
    return a, a[::-1]


a, b = edge_weights(raw_v1.shape[0])
c, e = edge_weights(raw_v1.shape[1])
v1_visual_display += (
    a[:, None, None] * err[0][None, :, :] + b[:, None, None] * err[-1][None, :, :]
)
v1_visual_display += (
    c[None, :, None] * err[:, 0][:, None, :] + e[None, :, None] * err[:, -1][:, None, :]
)
for u, v, corner in [
    (a, c, err[0, 0]),
    (a, e, err[0, -1]),
    (b, c, err[-1, 0]),
    (b, e, err[-1, -1]),
]:
    v1_visual_display -= u[:, None, None] * v[None, :, None] * corner
v1_visual_display[[0, -1]] = raw_v1[[0, -1]]
v1_visual_display[:, [0, -1]] = raw_v1[:, [0, -1]]
assert np.isfinite(v1_visual_display).all()
np.savez_compressed(
    OUT / "panel_A_display.npz", original=raw_v1, smoothed=v1_visual_display
)
delta = np.linalg.norm(v1_visual_display - raw_v1, axis=2)
(OUT / "panel_A_smoothing.json").write_text(
    json.dumps(
        dict(
            method="Light Gaussian display smoothing matching HVA sigma: (3,1.5) dense samples, reflect on both nonperiodic V1 axes, six-sample exponential boundary correction with corner compensation; all four perimeter edges exact.",
            scope="Panel A visual-field grid and highlighted transects only; trained RF arrays and quantitative analysis unchanged.",
            rms_displacement_deg=float(np.sqrt(np.mean(delta**2))),
            max_displacement_deg=float(delta.max()),
        ),
        indent=2,
    )
)

# The fitted V1 example now fills the upper half of A.
av = fig.add_axes([0.07, 0.54, 0.22, 0.305])
ac = fig.add_axes([0.325, 0.54, 0.17, 0.305])
for ax, p, title in [
    (av, v1_visual_display, "Visual field"),
    (ac, d["panel_b_cortex_v1"], "V1 cortex"),
]:
    for i in range(0, p.shape[0], 3):
        ax.plot(*p[i].T, c="#aabdc3", lw=0.32)
        ax.plot(*p[:, i].T, c="#aabdc3", lw=0.32)
    ax.set_aspect("equal")
text(0.18, 1 - 61 / 430.5, "Visual field", 11, ha="center")
text(0.41, 1 - 61 / 430.5, "V1 cortex", 11, ha="center")
sites = np.array(meta["opening_landmarks_cortex"])
srf = np.array(meta["opening_landmarks_rf"])
native = d["cortex_v1"]
for n, (p, r, c) in enumerate(zip(sites, srf, LAND), 1):
    row = np.argmin(np.linalg.norm(native - p, axis=2)) // native.shape[1]
    for ax, grid, site in [(ac, d["panel_b_cortex_v1"], p), (av, v1_visual_display, r)]:
        ax.plot(*grid[row * 3].T, c=c, lw=1.7, zorder=3)
av.plot(*close(d["rf_surround"][:, 0]).T, c=TEAL, lw=0.9)
ac.plot(*close(d["cortex_surround"][:, 0]).T, c=TEAL, lw=0.9)
av.set(xlim=(-5, 145), ylim=(-45, 65), xlabel="Azimuth (°)", ylabel="Elevation (°)")
av.set_xticks([0, 45, 90, 135])
av.set_yticks([-40, 0, 40])
av.tick_params(labelsize=7, length=3)
ac.axis("off")
fig.add_artist(
    FancyArrowPatch(
        (0.286, 0.721),
        (0.33, 0.721),
        transform=fig.transFigure,
        arrowstyle="<->",
        mutation_scale=10,
        lw=0.8,
        color=MUTED,
    )
)
# Scale A's opening maps about their visible top edges.
fig.canvas.draw()
for axis, grid in [(av, v1_visual_display), (ac, d["panel_b_cortex_v1"])]:
    points = grid.reshape(-1, 2)
    old_top = axis.transData.transform(points)[:, 1].max()
    box = axis.get_position()
    scale = 0.85
    axis.set_position(
        [
            box.x0 + box.width * (1 - scale) / 2,
            box.y0 + box.height * (1 - scale),
            box.width * scale,
            box.height * scale,
        ]
    )
    fig.canvas.draw()
    delta = (old_top - axis.transData.transform(points)[:, 1].max()) / fig.bbox.height
    box = axis.get_position()
    axis.set_position([box.x0, box.y0 + delta, box.width, box.height])
from constraint_mapping_examples_final import draw_constraints

constraint_start = len(fig.axes)
draw_constraints(fig, text)
for diagram in fig.axes[constraint_start:]:
    box = diagram.get_position()
    diagram.set_position(
        [
            box.x0 + box.width * 0.125,
            box.y0 + box.height * 0.125,
            box.width * 0.75,
            box.height * 0.75,
        ]
    )
    for line in diagram.lines:
        line.set_linewidth(line.get_linewidth() * 0.85)
    for collection in diagram.collections:
        if hasattr(collection, "get_sizes"):
            collection.set_sizes(collection.get_sizes() * 0.64)
# Lift A's artwork into the space released by the single-line title.
for a in fig.axes:
    box = a.get_position()
    a.set_position([box.x0, box.y0 + 18 / 430.5, box.width, box.height])
for t in fig.texts:
    x, y = t.get_position()
    if x < 0.54 and y < 0.9:
        t.set_position((x, y + 18 / 430.5))
from matplotlib.transforms import Affine2D

for artist in fig.artists:
    artist.set_transform(Affine2D().translate(0, 18 / 430.5) + fig.transFigure)
# B: use saved fields and shared cortical/color scales.
b_axes_start = len(fig.axes)
b_row_labels = []
cols = [0.567, 0.712, 0.857]
cmaps = [
    "Spectral",
    "Spectral",
    LinearSegmentedColormap.from_list("sign", ["#0047d9", "#ffffff", "#e20b24"]),
]
norms = [Normalize(0, 140), Normalize(-40, 60), Normalize(-1, 1)]
ext = d["cortex_surround"].reshape(-1, 2)
lo = ext.min(0)
hi = ext.max(0)
for row, (domain, y) in enumerate([("v1", 0.55), ("surround", 0.235)]):
    p = d["panel_b_cortex_" + domain]
    r = d["panel_b_rf_" + domain]
    t = faces(p.shape[:2], row == 1)
    tr = mtri.Triangulation(*p.reshape(-1, 2).T, t)
    values = [r[:, :, 0], r[:, :, 1], d["field_sign_" + domain + "_display"]]
    text(
        0.542,
        y + 0.17,
        "V1" if row == 0 else "Higher visual cortex",
        11,
        rotation=90,
        ha="center",
        va="center",
    )
    b_row_labels.append(fig.texts[-1])
    for j in range(3):
        ax = fig.add_axes([cols[j] - 0.007, y, 0.14, 0.34])
        if row == 1 and j == 2:
            native = partition_field["cortex"]
            native_tri = faces(native.shape[:2], True)
            ax.tripcolor(
                mtri.Triangulation(*native.reshape(-1, 2).T, native_tri),
                partition_field["field_sign"].ravel(),
                shading="gouraud",
                cmap=cmaps[j],
                norm=norms[j],
                rasterized=True,
            )
        else:
            ax.tripcolor(
                tr,
                values[j].ravel(),
                shading="gouraud",
                cmap=cmaps[j],
                norm=norms[j],
                rasterized=True,
            )
        for col in [0] if row == 0 else [0, -1]:
            ax.plot(*close(d["cortex_surround"][:, col]).T, c=INK, lw=0.7)
        ax.set(xlim=(lo[0] - 0.15, hi[0] + 0.15), ylim=(lo[1] - 0.15, hi[1] + 0.15))
        ax.set_aspect("equal")
        ax.axis("off")
        if row == 0:
            text(
                cols[j] + 0.063,
                1 - 43 / 430.5,
                ["Azimuth (°)", "Elevation (°)", "Field sign"][j],
                11,
                ha="center",
            )
        if row == 1:
            barax = fig.add_axes([cols[j] + 0.005, 0.168, 0.12, 0.009])
            bar = fig.colorbar(
                plt.cm.ScalarMappable(norm=norms[j], cmap=cmaps[j]),
                cax=barax,
                orientation="horizontal",
            )
            bar.set_ticks([[0, 70, 140], [-40, 0, 60], [-1, 0, 1]][j])
            bar.ax.tick_params(labelsize=7, length=2, pad=3)
            bar.outline.set_visible(False)
# Align actual cortical map apices, rather than axes boxes with different padding.
fig.canvas.draw()
reference = d["panel_b_cortex_v1"].reshape(-1, 2)
a_top = ac.transData.transform(reference)[:, 1].max()
b_top = fig.axes[b_axes_start].transData.transform(reference)[:, 1].max()
dy = (a_top - b_top) / fig.bbox.height
for a in fig.axes[b_axes_start:]:
    box = a.get_position()
    a.set_position([box.x0, box.y0 + dy, box.width, box.height])
for t in b_row_labels:
    x, y = t.get_position()
    t.set_position((x, y + dy))
# Spend the whitespace above the colorbars on the between-row title.
for index in [3, 5, 7]:
    a = fig.axes[b_axes_start + index]
    box = a.get_position()
    a.set_position([box.x0, box.y0 - 22 / 430.5, box.width, box.height])
x, y = b_row_labels[1].get_position()
b_row_labels[1].set_position((x, y - 22 / 430.5))
# Continue B's title between its two rows, centered in the visible map gap.
fig.canvas.draw()
upper_bottom = (
    fig.axes[b_axes_start]
    .transData.transform(d["panel_b_cortex_v1"].reshape(-1, 2))[:, 1]
    .min()
)
lower_top = (
    fig.axes[b_axes_start + 3]
    .transData.transform(d["panel_b_cortex_surround"].reshape(-1, 2))[:, 1]
    .max()
)
text(
    0.55 + 51 / 864,
    (upper_bottom * 0.30 + lower_top * 0.70) / fig.bbox.height,
    "\u2026creates a complex retinotopic map",
    16,
    ha="left",
    va="center",
)
# Reclaim A's spare right margin so both B title lines use regular 16 pt type.
for a in fig.axes[b_axes_start:]:
    box = a.get_position()
    a.set_position([box.x0 - 20 / 864, box.y0, box.width, box.height])
for t in fig.texts:
    x, y = t.get_position()
    if x >= 0.54:
        t.set_position((x - 20 / 864, y))
# Increase title-to-subheading clearance; keep map tops aligned.
subtitle_names = {
    "Visual field",
    "V1 cortex",
    "Azimuth (\u00b0)",
    "Elevation (\u00b0)",
    "Field sign",
}
for t in fig.texts:
    if t.get_text() in subtitle_names:
        x, y = t.get_position()
        t.set_position((x, y - 4 / 430.5))
# Move the opening maps by the same amount, retaining subtitle-to-map clearance.
for a in [av, ac, *fig.axes[b_axes_start : b_axes_start + 3]]:
    box = a.get_position()
    a.set_position([box.x0, box.y0 - 4 / 430.5, box.width, box.height])
x, y = b_row_labels[0].get_position()
b_row_labels[0].set_position((x, y - 4 / 430.5))
# Reduce the band height while retaining the maps' physical sizes and positions.
from matplotlib.transforms import Affine2D

ratio = 430.5 / 346.5
for a in fig.axes:
    box = a.get_position()
    a.set_position([box.x0, 1 - (1 - box.y0) * ratio, box.width, box.height * ratio])
for t in fig.texts:
    x, y = t.get_position()
    t.set_position((x, 1 - (1 - y) * ratio))
for artist in fig.artists:
    artist.set_transform(
        Affine2D().translate(0, 18 / 430.5).scale(1, ratio).translate(0, 1 - ratio)
        + fig.transFigure
    )
fig.set_size_inches(12, 346.5 / 72)
for ext in ["png", "pdf", "svg"]:
    fig.savefig(OUT / f"panels_AB_redesigned.{ext}", dpi=260, facecolor="white")
plt.close(fig)
print("A/B redesign rendered")
