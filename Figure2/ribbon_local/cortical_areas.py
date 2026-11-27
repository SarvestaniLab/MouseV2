"""Draw the saved cortical parcels using their native grid cells."""

import numpy as np
from matplotlib.collections import PolyCollection


def plot_cortical_areas(axis, surface, labels, title, colors, node_regions=False):
    """Render categorical areas; node_regions preserves every corner assignment."""
    polygons = []
    polygon_labels = []
    height, width = labels.shape
    for row in range(height):
        next_row = (row + 1) % height
        for column in range(width - 1):
            corners = np.asarray(
                (
                    surface[row, column],
                    surface[next_row, column],
                    surface[next_row, column + 1],
                    surface[row, column + 1],
                )
            )
            corner_labels = np.asarray(
                (
                    labels[row, column],
                    labels[next_row, column],
                    labels[next_row, column + 1],
                    labels[row, column + 1],
                )
            ).copy()
            if node_regions:
                # The fixed V1 border has no area identity. Extend the first
                # active radial node to it, without voting across angular rows.
                if column == 0:
                    if corner_labels[0] < 0:
                        corner_labels[0] = corner_labels[3]
                    if corner_labels[1] < 0:
                        corner_labels[1] = corner_labels[2]
                center = corners.mean(axis=0)
                for index, label in enumerate(corner_labels):
                    if label < 0:
                        continue
                    polygons.append(
                        np.asarray(
                            (
                                corners[index],
                                (corners[index] + corners[(index + 1) % 4]) / 2,
                                center,
                                (corners[index] + corners[(index - 1) % 4]) / 2,
                            )
                        )
                    )
                    polygon_labels.append(int(label))
            else:
                valid = corner_labels[corner_labels >= 0]
                if not len(valid):
                    continue
                polygons.append(corners)
                polygon_labels.append(int(np.argmax(np.bincount(valid))))
    collection = PolyCollection(
        polygons,
        facecolors=np.asarray(colors)[np.asarray(polygon_labels, dtype=int)],
        edgecolors="none",
        antialiased=False,
        rasterized=True,
    )
    axis.add_collection(collection)
    surface_closed = np.concatenate((surface, surface[:1]), axis=0)
    axis.plot(
        surface_closed[:, 0, 0],
        surface_closed[:, 0, 1],
        color="black",
        lw=1.5,
    )
    axis.plot(
        surface_closed[:, -1, 0],
        surface_closed[:, -1, 1],
        color="0.60",
        lw=0.95,
    )
    axis.autoscale_view()
    axis.set_title(title, fontsize=10, pad=3)
    axis.set_aspect("equal")
    axis.axis("off")
    return collection
