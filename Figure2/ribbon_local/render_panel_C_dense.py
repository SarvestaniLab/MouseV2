"""Publication-style panel C from the latest saved fit; no model changes."""

from pathlib import Path
import os, json, hashlib
import numpy as np
import matplotlib

OUT = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl_cache")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import proj3d

BASE = OUT.parent / "data"
d = np.load(BASE / "figure2_data.npz")
meta = json.loads((BASE / "figure2_shared_model.json").read_text())
from scipy.ndimage import gaussian_filter

xy = d["panel_c_cortex"]
raw = d["panel_c_rf"]
h, w = xy.shape[:2]
rf = gaussian_filter(raw, sigma=(3, 1.5, 0), mode=("wrap", "reflect", "nearest"))
rf += (raw[:, 0] - rf[:, 0])[:, None, :] * np.exp(-np.arange(w)[None, :, None] / 6)
rf[:, 0] = raw[:, 0]
lift = np.load(OUT / "ribbon_smoothing_variants.npz")["light"]
cmap = LinearSegmentedColormap.from_list(
    "around",
    [
        "#7961ac",
        "#318db4",
        "#37a697",
        "#9fb94f",
        "#e5b64c",
        "#dc8857",
        "#b7659a",
        "#7961ac",
    ],
)
colors = cmap(np.arange(h) / h)
dark = "#39434b"
gray = "#aeb6ba"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": dark,
        "axes.labelcolor": dark,
        "xtick.color": dark,
        "ytick.color": dark,
    }
)


def close(q):
    return np.concatenate((q, q[:1]))


def faces(n, m, cyclic=False):
    f = []
    for i in range(n if cyclic else n - 1):
        for j in range(m - 1):
            a = i * m + j
            b = ((i + 1) % n) * m + j
            f.extend([[a, b, b + 1], [a, b + 1, a + 1]])
    return np.array(f)


f = faces(h, w, True)
vf = faces(*d["cortex_v1"].shape[:2])
vxy = d["cortex_v1"].reshape(-1, 2)
vrf = d["rf_v1"].reshape(-1, 2)
tri = mtri.Triangulation(*vxy.T, vf)
contours = []
tmp, ta = plt.subplots()
for k, levels in [(0, [30, 60, 90, 120]), (1, [-20, 0, 20, 40])]:
    cs = ta.tricontour(tri, vrf[:, k], levels=levels)
    for ss in cs.allsegs:
        for seg in ss:
            if len(seg) > 1:
                contours.append(seg)
plt.close(tmp)
landC = np.array(meta["opening_landmarks_cortex"])
landR = np.array(meta["opening_landmarks_rf"])
landColors = ["#ae426b", "#a78417", "#007d7b"]
fig = plt.figure(figsize=(12, 302.8 / 72))
left = fig.add_axes([0.018, 0.19, 0.245, 0.65])
middle = fig.add_axes([0.27, 0.16, 0.455, 0.68], projection="3d")
right = fig.add_axes([0.76, 0.25, 0.22, 0.56])


def flatmesh(ax, p):
    for i in range(0, h, 3):
        ax.plot(*p[i].T, c=colors[i], lw=0.30, alpha=0.65, zorder=2)
    for j in range(0, w, 3):
        q = close(p[:, j])
        ax.add_collection(
            LineCollection(
                np.stack((q[:-1], q[1:]), axis=1),
                colors=colors,
                linewidths=0.35,
                alpha=0.65,
                zorder=2,
            )
        )
    ax.plot(*close(p[:, 0]).T, c=dark, lw=1.05, zorder=3)
    ax.plot(*close(p[:, -1]).T, c=dark, lw=0.9, ls=(0, (4, 2.5)), zorder=3)


left.add_collection(
    PolyCollection(vxy[vf], facecolors="#f1f3f3", edgecolors="none", zorder=0)
)
for gridline in list(d["cortex_v1"]) + list(d["cortex_v1"].transpose(1, 0, 2)):
    left.plot(*gridline.T, c="#b5c4c9", lw=0.24, alpha=0.60, zorder=1)
flatmesh(left, xy)
left.text(
    *(vxy.mean(0) + [0.45, -0.3]),
    "V1",
    ha="center",
    va="center",
    fontsize=9,
    color="#778187",
)
left.set_aspect("equal")
left.axis("off")
left.autoscale_view()
units = json.loads((OUT / "cumulative_height_audit.json").read_text())["units_per_mm"]
bottom = xy[:, :, 1].min() - 0.35
xstart = xy[:, :, 0].min()
left.plot([xstart, xstart + 0.5 * units], [bottom, bottom], c=dark, lw=1.5)
left.text(
    xstart + 0.25 * units, bottom - 0.17, "0.5 mm", ha="center", va="top", fontsize=7
)
right.add_collection(
    PolyCollection(vrf[vf], facecolors="#f1f3f3", edgecolors="none", zorder=0)
)
# Reverse material side in the visual-field projection: compare signed
# triangle orientation with the original cortical sheet, not with V1 sign.
ct = xy.reshape(-1, 2)[f]
rt = rf.reshape(-1, 2)[f]
ca = ct[:, 1] - ct[:, 0]
cb = ct[:, 2] - ct[:, 0]
ra = rt[:, 1] - rt[:, 0]
rb = rt[:, 2] - rt[:, 0]
reverse = (
    (ca[:, 0] * cb[:, 1] - ca[:, 1] * cb[:, 0])
    * (ra[:, 0] * rb[:, 1] - ra[:, 1] * rb[:, 0])
) < 0
# Single opaque light-gray fill avoids darkness accumulating at triangle seams.
right.add_collection(
    PolyCollection(rt[reverse], facecolors="#d6dce0", edgecolors="none", zorder=1)
)
for gridline in list(d["rf_v1"]) + list(d["rf_v1"].transpose(1, 0, 2)):
    right.plot(*gridline.T, c="#c5cfd3", lw=0.22, alpha=0.40, zorder=0.5)
flatmesh(right, rf)
right.set(xlim=(-5, 145), ylim=(-45, 65), xlabel="Azimuth (°)", ylabel="Elevation (°)")
right.set_xticks([0, 45, 90, 135])
right.set_yticks([-40, 0, 40])
right.set_aspect("equal")
right.tick_params(labelsize=7, length=3, pad=3)
ax = middle
ax.computed_zorder = False
vbase = np.column_stack((vxy, np.zeros(len(vxy))))
ax.add_collection3d(
    Poly3DCollection(vbase[vf], facecolors="#f1f3f3", edgecolors="none", zorder=0)
)
for seg in list(d["cortex_v1"]) + list(d["cortex_v1"].transpose(1, 0, 2)):
    ax.plot(seg[:, 0], seg[:, 1], 0, c="#b5c4c9", lw=0.23, alpha=0.55, zorder=1)
b = close(d["cortex_surround"][:, 0])
ax.plot(b[:, 0], b[:, 1], 0, c=gray, lw=0.85, zorder=2)
# Original HVA cortical footprint on the reference plane, independent of RF registration.
for i in range(0, h, 3):
    q = xy[i]
    ax.plot(
        q[:, 0], q[:, 1], np.zeros(len(q)), c="#d9dfe2", lw=0.22, alpha=0.38, zorder=1
    )
for j in range(0, w, 3):
    q = close(xy[:, j])
    ax.plot(
        q[:, 0], q[:, 1], np.zeros(len(q)), c="#d9dfe2", lw=0.22, alpha=0.38, zorder=1
    )
q = close(xy[:, -1])
ax.plot(q[:, 0], q[:, 1], np.zeros(len(q)), c="#b8c1c7", lw=0.75, alpha=0.85, zorder=2)

elev, azim = 32, -110
cam = np.array(
    [
        np.cos(np.deg2rad(elev)) * np.cos(np.deg2rad(azim)),
        np.cos(np.deg2rad(elev)) * np.sin(np.deg2rad(azim)),
        np.sin(np.deg2rad(elev)),
    ]
)
vv = lift.reshape(-1, 3)[f]
normal = np.cross(vv[:, 1] - vv[:, 0], vv[:, 2] - vv[:, 0])
light = np.array([-0.3, -0.6, 0.74])
light /= np.linalg.norm(light)
shade = 0.83 + 0.17 * np.abs(normal @ light) / np.maximum(
    np.linalg.norm(normal, axis=1), 1e-12
)
fc = colors[f[:, 0] // w].copy()
fc[:, :3] = (0.12 + 0.88 * fc[:, :3]) * shade[:, None]
fc[:, 3] = 0.62
# Orient the two material sides using the original cortical sheet winding.
source_tri = xy.reshape(-1, 2)[f]
a = source_tri[:, 1] - source_tri[:, 0]
b = source_tri[:, 2] - source_tri[:, 0]
winding = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
back = (normal * np.sign(winding)[:, None]) @ cam < 0

ax.add_collection3d(Poly3DCollection(vv, facecolors=fc, edgecolors="none", zorder=3))
for i in range(0, h, 3):
    ax.plot(*lift[i].T, c=colors[i], lw=0.29, alpha=0.65, zorder=4)
for j in range(3, w - 1, 3):
    q = close(lift[:, j])
    ax.add_collection3d(
        Line3DCollection(
            np.stack((q[:-1], q[1:]), axis=1),
            colors=colors,
            linewidths=0.28,
            alpha=0.55,
            zorder=4,
        )
    )
for j, style in [(0, "-"), (-1, (0, (4, 2.5)))]:
    ax.plot(*close(lift[:, j]).T, c=dark, lw=1.05, ls=style, zorder=5)
vc = vxy.mean(0)
ax.text(
    vc[0],
    vxy[:, 1].min() + 0.2,
    0,
    "V1 reference",
    ha="center",
    fontsize=9,
    color="#778187",
    zorder=8,
)
floor_xy = np.column_stack((xy.reshape(-1, 2), np.zeros(h * w)))
points = np.vstack((lift.reshape(-1, 3), vbase, floor_xy))
lo = points.min(0)
hi = points.max(0)
ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(0, hi[2]))
ax.set_box_aspect(hi - lo, zoom=1)
ax.view_init(elev=elev, azim=azim)
ax.set_proj_type("ortho")
ax.set_axis_off()


def bounds():
    px, py, _ = proj3d.proj_transform(*points.T, ax.get_proj())
    return ax.transData.transform(np.column_stack((px, py)))


b = bounds()
box = ax.get_window_extent()
zoom = 1.55 * min(box.width / np.ptp(b[:, 0]), box.height / np.ptp(b[:, 1]))
ax.set_box_aspect(hi - lo, zoom=zoom)
b = bounds()
shift = fig.transFigure.inverted().transform(
    box.get_points().mean(0)
) - fig.transFigure.inverted().transform((b.min(0) + b.max(0)) / 2)
rect = ax.get_position()
ax.set_position([rect.x0 + shift[0], rect.y0 + shift[1], rect.width, rect.height])
# Balance subtitle clearances without rescaling the views.
rect = middle.get_position()
middle.set_position([rect.x0, rect.y0 - 7 / 302.8, rect.width, rect.height])
rect = right.get_position()
right.set_position([rect.x0, rect.y0 + 11 / 302.8, rect.width, rect.height])
for artist in [*ax.collections, *ax.lines, *ax.texts]:
    artist.set_clip_on(False)
fig.text(0.025, 0.915, "C", fontsize=22, weight="bold")
fig.text(
    0.061,
    0.915,
    "The model predicts that a single V2 surface will fold to cover the visual field",
    fontsize=16,
)
for x, title in [
    (0.14, "Cortical sheet"),
    (0.50, "Mapping onto V1"),
    (0.87, "Visual field"),
]:
    fig.text(x, 0.85, title, ha="center", fontsize=11)


# Compact palette and edge keys, without numerical color values.
cb = fig.add_axes([0.035, 0.106, 0.13, 0.013])
cb.imshow(np.arange(256)[None, :], cmap=cmap, aspect="auto")
cb.axis("off")
fig.text(0.10, 0.082, "Position around cortex", ha="center", fontsize=7)
fig.legend(
    handles=[
        Line2D([0], [0], color=dark, lw=1, label="V1/V2 border"),
        Line2D([0], [0], color=dark, lw=1, ls=(0, (4, 2.5)), label="Outer boundary"),
    ],
    loc="lower left",
    bbox_to_anchor=(0.22, 0.065),
    frameon=False,
    ncol=2,
    fontsize=7,
    handlelength=2.8,
    columnspacing=2,
)
fig.text(
    0.975,
    0.083,
    "Gray fill: exposed reverse side",
    ha="right",
    fontsize=7.5,
    color="#596970",
)

for ext in ["png", "svg", "pdf"]:
    fig.savefig(OUT / f"panel_C_dense.{ext}", dpi=220, facecolor="white")
plt.close(fig)
(OUT / "panel_C_dense_notes.txt").write_text(
    "Source: selected seed 2026111604. Original fitted data unchanged. Faint HVA grid and outer outline restored at z=0 using original cortical XY, with no registration transform.\nLeft: saved cortical display mesh. Right: light periodic Gaussian display smoothing of RF XY (3 samples around, 1.5 across), anchored to exact inner RF boundary; source fitted RF arrays unchanged. Common continuous around-domain palette. Gray fill in right visual-field mesh marks triangles whose displayed RF orientation reverses relative to original cortical winding. This is based on smoothed display coordinates, not the quantitative field-sign map. Middle ribbon shading restored to v1. Registered XY and RF XY are separately smoothed display approximations, not exact mutual inverses.\nMiddle: light registered-XY display smoothing from ribbon_smoothing_variants.npz, original panel_c_height (no extra exaggeration). This is an illustrative registration surface; source illustrative fills retained.\nV1 is the trained V1 reference throughout. No numbered landmarks are drawn in this panel C layout.\nSurface smoothing is not evidence for reversals; use the fitted visual-field mesh and field-sign results.\nEditable vector SVG and PDF exported alongside PNG.\n",
    encoding="utf-8",
)
print("Refined panel C exported.")
