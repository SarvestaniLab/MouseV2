"""Render panels D/E from saved cortical parcels and injection memberships."""

from pathlib import Path
import json, numpy as np, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figure2_compositions/clean_export"
BASE = ROOT / "data"
d = np.load(BASE / "figure2_data.npz")
meta = json.loads((BASE / "figure2_shared_model.json").read_text())
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.5,
        "text.color": "#29333a",
        "axes.labelcolor": "#29333a",
        "axes.linewidth": 0.65,
    }
)
W, H = 864, 971.5
f = plt.figure(figsize=(12, H / 72))
f.patch.set_alpha(0)
c1 = d["cortex_v1"]
c2 = d["cortex_surround"]
v1 = d["rf_v1"]
v2 = d["rf_surround"]
close = lambda p: np.vstack([p, p[0]])
colors = meta["injection_colors"]
sites = np.array(meta["injection_centers"])
srf = np.array(meta["injection_rf"])
units = json.loads((ROOT / "ribbon_local/cumulative_height_audit.json").read_text())[
    "units_per_mm"
]
radius = meta["injection_radius_mm"] * units
# E focuses on cortical injection sites and their surrounding targets.
a = f.add_axes([0.695, 0.035, 0.295, 0.225])
target_axis = a
for p in list(c2[::8]) + [close(p) for p in c2.transpose(1, 0, 2)[::8]]:
    a.plot(*p.T, c="#c2cdd2", lw=0.3)
for j in [0, -1]:
    a.plot(*close(c2[:, j]).T, c="#839199", lw=0.7)
for k, (site, col) in enumerate(zip(sites, colors)):
    mask = d["injection_membership"][:, k].reshape(v2.shape[:2]).copy()
    mask[:, 0] = False
    # Native cortical adjacency only, including the periodic seam around V1.
    grid_ids = np.arange(mask.size).reshape(mask.shape)
    candidate_edges = np.concatenate(
        [
            np.stack([grid_ids.ravel(), np.roll(grid_ids, -1, axis=0).ravel()], axis=1),
            np.stack([grid_ids[:, :-1].ravel(), grid_ids[:, 1:].ravel()], axis=1),
        ]
    )
    selected_edges = candidate_edges[np.all(mask.ravel()[candidate_edges], axis=1)]
    a.add_collection(
        LineCollection(c2.reshape(-1, 2)[selected_edges], colors=[col], linewidths=0.55)
    )
    # Preserve isolated selected nodes without inventing non-neighbor connections.
    isolated = np.setdiff1d(np.flatnonzero(mask), np.unique(selected_edges))
    if len(isolated):
        a.scatter(*c2.reshape(-1, 2)[isolated].T, s=3, c=col, edgecolors="none")
    a.add_patch(Circle(site, radius, facecolor=col, alpha=0.38))
    a.add_patch(Circle(site, radius, fill=False, ec=col, lw=0.9))
    a.scatter(*site, s=14, c=col, edgecolors="#29333a", lw=0.4)
    a.text(*(site + [radius + 0.05, 0]), "ABC"[k], fontsize=8)
lo = c2.reshape(-1, 2).min(0)
hi = c2.reshape(-1, 2).max(0)
a.set(xlim=(lo[0] - 0.1, hi[0] + 0.1), ylim=(lo[1] - 0.4, hi[1] + 0.1))
a.set_aspect("equal")
a.axis("off")
x, y = lo[0], lo[1] - 0.2
a.plot([x, x + 0.5 * units], [y, y], c="#29333a", lw=1)
a.text(x + 0.25 * units, y - 0.14, "0.5 mm", ha="center", va="top", fontsize=7.5)

# D: reconstruct current saved subdivisions and RF plots from data.
from matplotlib.collections import PolyCollection, LineCollection
from cortical_areas import plot_cortical_areas

INK = "#202326"
labels = d["labels"]
n = int(meta["parcels"])
parcel_signs = np.asarray(meta["parcel_signs"])
if n < 1 or parcel_signs.shape != (n,) or not np.isin(parcel_signs, [-1, 1]).all():
    raise ValueError("Expected one positive or negative field sign per parcel")
if not np.array_equal(np.unique(labels[labels >= 0]), np.arange(n)):
    raise ValueError("Parcel labels must match the saved field-sign order")
RED = (0.80, 0.12, 0.12)
BLUE = (0.10, 0.35, 0.80)
# Colors follow each parcel's computed sign, independent of count or ID order.
colors = np.array([(*RED, 1.0) if sign > 0 else (*BLUE, 1.0) for sign in parcel_signs])

fig = f
closed = close
plt.rcParams.update({"font.size": 8})
d_axes = []


def ax(bounds, title=None):
    x, y, w, h = bounds
    a = f.add_axes([x, y * 828 / H, w, h * 828 / H])
    d_axes.append(a)
    return a


def cortical(a):
    for j in [0, -1]:
        a.plot(*close(c2[:, j]).T, c=INK if j == 0 else "#8d969c", lw=0.8, zorder=5)
    a.set_aspect("equal")
    a.axis("off")
    a.set(xlim=(lo[0] - 0.2, hi[0] + 0.2), ylim=(lo[1] - 0.35, hi[1] + 0.2))


def vf(a, ticks=False):
    a.plot(*close(v2[:, 0]).T, c=INK, lw=0.8)
    a.set(xlim=(-5, 145), ylim=(-44, 66))
    a.set_aspect("equal")
    a.axis("off")


# Composite alpha against the white page before rasterizing adjacent fill cells,
# avoiding repeated alpha blending at shared cell edges. Meshes retain native alpha.
cortex_colors = colors.copy()
cortex_colors[:, :3] = colors[:, :3] * colors[:, 3, None] + (1 - colors[:, 3, None])
cortex_colors[:, 3] = 1.0
a = ax([0.025, 0.037, 0.16, 0.162])
plot_cortical_areas(a, c2, labels, "", cortex_colors, node_regions=True)
# Match the node-region fill geometry exactly: boundaries run from a quad's
# center to an edge midpoint whenever adjacent corner parcel labels differ.
parcel_boundaries = []
for row in range(labels.shape[0]):
    nxt = (row + 1) % labels.shape[0]
    for col in range(labels.shape[1] - 1):
        corners = np.array(
            [c2[row, col], c2[nxt, col], c2[nxt, col + 1], c2[row, col + 1]]
        )
        ids = np.array(
            [
                labels[row, col],
                labels[nxt, col],
                labels[nxt, col + 1],
                labels[row, col + 1],
            ]
        )
        if col == 0:
            if ids[0] < 0:
                ids[0] = ids[3]
            if ids[1] < 0:
                ids[1] = ids[2]
        for j in range(4):
            k = (j + 1) % 4
            if ids[j] >= 0 and ids[k] >= 0 and ids[j] != ids[k]:
                parcel_boundaries.append(
                    [corners.mean(axis=0), (corners[j] + corners[k]) / 2]
                )
a.add_collection(
    LineCollection(parcel_boundaries, colors="white", linewidths=0.65, zorder=4)
)
# E shares the exact saved subdivision geometry shown in D.
target_axis.add_collection(
    LineCollection(parcel_boundaries, colors="#68747c", linewidths=0.60, zorder=1.9)
)
cortical(a)
a.text(*c2[:, 0].mean(axis=0), "V1", ha="center", fontsize=8)
anchors = []
outer = c2[:, -1]
origin = outer.mean(axis=0)
for k in range(n):
    p = c2[labels == k].mean(axis=0)
    direction = p - c2[:, 0].mean(axis=0)
    anchors.append(
        outer[
            np.argmax(
                (outer - c2[:, 0].mean(axis=0))
                @ direction
                / np.maximum(
                    np.linalg.norm(outer - c2[:, 0].mean(axis=0), axis=1), 1e-8
                )
            )
        ]
    )
anchors = np.asarray(anchors)
angles = np.arctan2(anchors[:, 1] - origin[1], anchors[:, 0] - origin[0])
order = np.argsort(angles)
ordered = angles[order].copy()
for _ in range(60):
    for j in range(n):
        nxt = (j + 1) % n
        gap = (ordered[nxt] - ordered[j]) % (2 * np.pi)
        if gap < 0.20:
            delta = (0.20 - gap) / 2
            ordered[j] -= delta
            ordered[nxt] += delta
angles[order] = ordered
parcel_radii = np.linalg.norm(anchors - origin, axis=1) * 1.20
# Put leader endpoints inside the exact saved parcel, choosing its deepest node.
from scipy.spatial import cKDTree

parcel_anchors = []
for k in range(n):
    candidates = c2[labels == k]
    distances = cKDTree(np.vstack([c2[labels != k], c2[:, 0], c2[:, -1]])).query(
        candidates
    )[0]
    parcel_anchors.append(candidates[np.argmax(distances)])
parcel_anchors = np.asarray(parcel_anchors)
for k, angle in enumerate(angles):
    pos = origin + parcel_radii[k] * np.array([np.cos(angle), np.sin(angle)])
    a.annotate(
        str(k + 1),
        parcel_anchors[k],
        xytext=pos,
        ha="center",
        va="center",
        fontsize=10.5,
        color=INK,
        weight="bold",
        annotation_clip=False,
        zorder=12,
        bbox=dict(boxstyle="round,pad=.13", fc="white", ec="none"),
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkA=2, shrinkB=0),
    )
    a.plot(
        *parcel_anchors[k],
        marker="o",
        ms=2.4,
        mfc=INK,
        mec="white",
        mew=0.45,
        zorder=11,
    )
# Native cortical neighbors: periodic along the surrounding sheet, open across it.
# Keep an edge only when both endpoints belong to the same parcel.
h, w = labels.shape
node_ids = np.arange(h * w).reshape(h, w)
edges = np.concatenate(
    [
        np.stack([node_ids.ravel(), np.roll(node_ids, -1, axis=0).ravel()], axis=1),
        np.stack([node_ids[:, :-1].ravel(), node_ids[:, 1:].ravel()], axis=1),
    ]
)
flat_labels = labels.ravel()
flat_rf = v2.reshape(-1, 2)
mesh_audit = []
nr = int(np.ceil(n / 4))
height = 0.19 / nr
for k in range(n):
    a = ax(
        [
            0.198 + (k % 4) * 0.111,
            0.035 + (nr - 1 - k // 4) * height,
            0.104,
            height * 0.88,
        ]
    )
    parcel_edges = edges[np.all(flat_labels[edges] == k, axis=1)]
    segments = flat_rf[parcel_edges]
    a.add_collection(LineCollection(segments, colors=[colors[k]], linewidths=0.38))
    mesh_audit.append(
        dict(parcel=k + 1, nodes=int(np.sum(labels == k)), edges=len(parcel_edges))
    )
    vf(a, False)
    a.set_title(
        str(k + 1), color=tuple(colors[k, :3]), fontsize=10.5, pad=2, weight="bold"
    )

for da in d_axes:
    box = da.get_position()
    da.set_position(
        [box.x0, box.y0 * 1.27 + 0.025 * 828 / H - 14 / H, box.width, box.height * 1.27]
    )

for x, letter, title in [
    (21.6, "D", "Typical rules of area designation split the complex map"),
    (574.56, "E", "Simulated tracer mapping"),
]:
    f.text(x / W, 1 - 708.887 / H, letter, fontsize=22, weight="bold")
    f.text((x + 31) / W, 1 - 708.887 / H, title, fontsize=16)
    if letter == "D":
        f.text(
            (x + 31) / W, 1 - 726.887 / H, "along the fold-line reversals", fontsize=16
        )
# Sign key belongs to D; parcel numbers distinguish identities within each group.
for x, col, label in [
    (0.22, RED, "Positive field sign"),
    (0.44, BLUE, "Negative field sign"),
]:
    f.add_artist(
        plt.Line2D(
            [x, x + 0.018], [0.047, 0.047], transform=f.transFigure, color=col, lw=2.5
        )
    )
    f.text(x + 0.024, 0.047, label, fontsize=8, va="center", color=INK)
(ROOT / "panel_D_palette.json").write_text(
    json.dumps(
        dict(
            space="Uniform RGB by field-sign group; white cortical parcel boundaries",
            positive_rgb=RED,
            negative_rgb=BLUE,
            parcel_signs=parcel_signs.tolist(),
            alpha_range=[1.0, 1.0],
            rgba=colors.tolist(),
        ),
        indent=2,
    )
)
for ext in ["pdf", "svg"]:
    f.savefig(OUT / f"panels_DE_clean.{ext}", transparent=True, dpi=300)

(ROOT / "panel_D_mesh_audit.json").write_text(
    json.dumps(
        dict(
            rule="Only original four-neighbor cortical-grid edges with both endpoints in the same parcel; periodic along, open across; no RF-space triangulation",
            parcels=mesh_audit,
        ),
        indent=2,
    )
)
