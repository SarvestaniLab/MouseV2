"""Schematic loss preferences, not model runs or ablations."""

import numpy as np
from matplotlib.patches import FancyArrowPatch


def draw_constraints(fig, text):
    ink = "#29333a"
    muted = "#596970"
    teal = "#407b8b"
    gray = "#c4cfd3"
    colors = ["#00b5ad", "#f06489", "#efb51b"]
    titles = ["Coverage", "Smoothness", "Boundary agreement", "Local isotropy"]
    captions = [
        "RFs span the visual field",
        "Cortical neighbors have nearby RFs",
        "Border RFs match fixed V1 RFs",
        "Equal stretch in both directions",
    ]
    labels = [
        ("Restricted coverage", "Broader coverage"),
        ("Large RF jumps", "Smaller RF jumps"),
        ("Border mismatch", "Matched border"),
        ("Unequal stretch", "Equal stretch"),
    ]
    u, v = np.meshgrid(np.linspace(-0.65, 0.65, 5), np.linspace(-0.48, 0.48, 4))
    regular = np.stack([u, v], axis=-1)

    def mesh(ax, p):
        for line in list(p) + list(p.transpose(1, 0, 2)):
            ax.plot(*line.T, color=gray, lw=0.65, zorder=2, solid_capstyle="round")
        ax.scatter(
            *p.reshape(-1, 2).T,
            s=7,
            facecolors="white",
            edgecolors=gray,
            linewidths=0.65,
            zorder=4
        )
        for i, c in zip([0, 1, 3], colors):
            ax.plot(*p[i].T, color=c, lw=1.05, zorder=3, solid_capstyle="round")
            ax.scatter(
                *p[i].T,
                s=8,
                facecolors="white",
                edgecolors=c,
                linewidths=0.75,
                zorder=5
            )

    for k in range(4):
        x = 0.045 + 0.21 * (k % 2)
        y = 0.29 + 18 / 430.5 - 0.15 * (k // 2)
        text(x, y + 0.14 - (3 / 430.5 if k >= 2 else 0), titles[k], 11, weight="medium")
        for side in range(2):
            ax = fig.add_axes([x + side * 0.095, y + 0.033, 0.095, 0.11])
            ax.set(xlim=(-1, 1), ylim=(-0.72, 0.72))
            ax.set_aspect("equal")
            ax.axis("off")
            if k == 0:
                # Same cortical connectivity, redistributed RF coordinates.
                ax.plot(
                    [-0.91, 0.91, 0.91, -0.91, -0.91],
                    [-0.65, -0.65, 0.65, 0.65, -0.65],
                    c=gray,
                    lw=0.6,
                )
                p = regular.copy()
                if side == 0:
                    p = p * [0.38, 0.6] + [-0.38, 0]
                else:
                    p = p * [1.22, 1.13]
                mesh(ax, p)
            elif k == 1:
                p = regular.copy()
                if side == 0:
                    p[1, 2] += [0.09, 0.65]
                    p[2, 3] += [-0.15, -0.65]
                mesh(ax, p)
            elif k == 2:
                # Visual-field RFs: fixed V1 circles and movable border crosses.
                target = np.array([[-0.5, -0.42], [-0.1, 0], [0.35, 0.42]])
                shifted = (
                    target + np.array([[0.55, 0.12], [0.5, -0.18], [0.4, -0.24]])
                    if side == 0
                    else target
                )
                ax.plot(*target.T, c=ink, lw=0.65)
                for j, c in enumerate(colors):
                    if side == 0:
                        ax.plot(
                            [target[j, 0], shifted[j, 0]],
                            [target[j, 1], shifted[j, 1]],
                            c=gray,
                            lw=0.75,
                            ls=":",
                        )
                    end = shifted[j] + [0.3, -0.05]
                    ax.plot(
                        [shifted[j, 0], end[0]], [shifted[j, 1], end[1]], c=c, lw=0.9
                    )
                    ax.scatter(
                        *target[j],
                        s=30,
                        facecolors="white",
                        edgecolors=c,
                        lw=0.9,
                        zorder=4
                    )
                    ax.scatter(*shifted[j], s=17, c=c, marker="x", lw=0.9, zorder=5)
            else:
                # Equal-area local RF patches: equal cortical steps are implicit in grid.
                q = np.linspace(-0.42, 0.42, 4)
                a, b = np.meshgrid(q, q)
                p = np.stack([a, b], axis=-1)
                if side == 0:
                    p = p * [1.9, 1 / 1.9]
                mesh(ax, p)
            text(
                x + side * 0.095 + 0.0475,
                y + 0.019 + (4 / 430.5 if k < 2 else 0),
                labels[k][side],
                7.2,
                ha="center",
                color=muted,
            )
        fig.add_artist(
            FancyArrowPatch(
                (x + 0.087, y + 0.086),
                (x + 0.103, y + 0.086),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=7,
                lw=0.7,
                color=muted,
            )
        )
