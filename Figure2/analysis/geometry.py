"""Interpolation, field signs, V1 inverse mapping, and illustrative registration."""

import numpy as np
import matplotlib.tri as mtri
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

FIELD_SIGN_SIGMA_MM = 0.075


def triangles(shape, cyclic=False):
    h, w = shape
    t = []
    for i in range(h if cyclic else h - 1):
        for j in range(w - 1):
            a = i * w + j
            b = ((i + 1) % h) * w + j
            t.extend([(a, b, b + 1), (a, b + 1, a + 1)])
    return np.array(t)


def interpolate_sheet(values, cyclic=False, factor=3, linear=False):
    """Interpolate coordinate values before deriving fields or assigning colors.

    Periodic cubic around the ribbon, PCHIP on open axes. No node averaging.
    """
    h, w = values.shape[:2]
    along = np.arange(h * factor if cyclic else (h - 1) * factor + 1) / factor
    across = np.arange((w - 1) * factor + 1) / factor
    if linear:
        first = (
            interp1d(np.arange(h + 1), np.concatenate([values, values[:1]]), axis=0)(
                along
            )
            if cyclic
            else interp1d(np.arange(h), values, axis=0)(along)
        )
        result = interp1d(np.arange(w), first, axis=1)(across)
        np.testing.assert_allclose(
            result[::factor, ::factor], values, rtol=0, atol=1e-12
        )
        return result
    if cyclic:
        first = CubicSpline(
            np.arange(h + 1),
            np.concatenate([values, values[:1]]),
            axis=0,
            bc_type="periodic",
        )(along)
    else:
        first = PchipInterpolator(np.arange(h), values, axis=0)(along)
    result = PchipInterpolator(np.arange(w), first, axis=1)(across)
    np.testing.assert_allclose(result[::factor, ::factor], values, rtol=0, atol=1e-12)
    return result


def fields(rf, cortex, cyclic=False):
    def grad(a):
        return (
            (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) / 2
            if cyclic
            else np.gradient(a, axis=0)
        )

    # Transform grid finite differences into physical cortical derivatives.
    dc = np.stack([grad(cortex), np.gradient(cortex, axis=1)], axis=-2)
    dr = np.stack([grad(rf), np.gradient(rf, axis=1)], axis=-2)
    jac = np.linalg.solve(dc, dr)
    az, el = jac[..., 0], jac[..., 1]
    sign = np.linalg.det(jac) / np.maximum(
        np.linalg.norm(az, axis=-1) * np.linalg.norm(el, axis=-1), 1e-12
    )
    return [rf[..., 0], rf[..., 1], sign]


def smooth_field_sign(values, cortex, tri, units_per_mm, sigma_mm=FIELD_SIGN_SIGMA_MM):
    """Area-weighted Gaussian average within one cortical domain (3 sigma)."""
    xy = cortex.reshape(-1, 2) / units_per_mm
    vertex = xy[tri]
    area = (
        np.abs(np.cross(vertex[:, 1] - vertex[:, 0], vertex[:, 2] - vertex[:, 0])) / 2
    )
    mass = np.zeros(len(xy))
    np.add.at(mass, tri.ravel(), np.repeat(area / 3, 3))
    tree = cKDTree(xy)
    raw = values.ravel()
    result = np.empty_like(raw)
    for i, neighbors in enumerate(tree.query_ball_point(xy, 3 * sigma_mm)):
        ids = np.asarray(neighbors)
        weight = mass[ids] * np.exp(
            -np.sum((xy[ids] - xy[i]) ** 2, axis=1) / (2 * sigma_mm**2)
        )
        result[i] = np.dot(weight, raw[ids]) / weight.sum()
    return result.reshape(values.shape)


def inverse_map(query, rf, cortex, tri, sites, radius):
    """Piecewise-linear inverse on actual V1 triangles, retaining all preimages.

    For drawing only, choose the most interior barycentric solution when there
    are multiple preimages. Predictions use ANY preimage in each injection.
    """
    v = rf.reshape(-1, 2)[tri]
    c = cortex.reshape(-1, 2)[tri]
    mat = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]], axis=-1)
    good = abs(np.linalg.det(mat)) > 1e-10
    v, c, mat = v[good], c[good], mat[good]
    inv = np.linalg.inv(mat)
    answer = np.full((len(query), 2), np.nan)
    counts = np.zeros(len(query), int)
    membership = np.zeros((len(query), len(sites)), bool)
    ambiguous = np.zeros(len(query), bool)
    for start in range(0, len(query), 128):
        q = query[start : start + 128]
        uv = np.einsum("tij,qtj->qti", inv, q[:, None] - v[None, :, 0])
        bary = np.concatenate([1 - uv.sum(axis=-1, keepdims=True), uv], axis=-1)
        valid = np.min(bary, axis=-1) >= -1e-8
        xy = np.einsum("qtk,tkd->qtd", bary, c)
        counts[start : start + len(q)] = valid.sum(axis=1)
        best = np.argmax(np.where(valid, bary.min(axis=-1), -np.inf), axis=1)
        has = valid.any(axis=1)
        answer[start + np.flatnonzero(has)] = xy[np.flatnonzero(has), best[has]]
        chosen = xy[np.arange(len(q)), best]
        ambiguous[start : start + len(q)] = np.any(
            valid & (np.sum((xy - chosen[:, None]) ** 2, axis=-1) > 1e-12), axis=1
        )
        for k, site in enumerate(sites):
            membership[start : start + len(q), k] = np.any(
                valid & (np.sum((xy - site) ** 2, axis=-1) <= radius**2), axis=1
            )
    return answer, counts, membership, ambiguous


def register_ribbon(r0, x0, cortex_v1, rf_v1):
    """Invert V1, mark ambiguous inverses, then fill display gaps and lift.

    Gap filling and height are illustrative; they do not alter the fitted RF map.
    """
    v1data = {"visual_grid": rf_v1}
    H, W = r0.shape[:2]
    UPSAMPLE = 3
    d = {
        "hva_cortex_xy": x0,
        "hva_rf_az_el": r0,
        "v1_cortex_xy": cortex_v1.reshape(-1, 2),
        "v1_rf_az_el": v1data["visual_grid"].reshape(-1, 2),
        "v1_faces": triangles(cortex_v1.shape[:2]),
    }
    aa = np.arange(H * UPSAMPLE) / UPSAMPLE
    bb = np.arange((W - 1) * UPSAMPLE + 1) / UPSAMPLE

    # Periodic cubic around the annulus; shape-preserving interpolation across it.
    def smooth(v):
        s = CubicSpline(
            np.arange(H + 1), np.concatenate((v, v[:1])), axis=0, bc_type="periodic"
        )
        return PchipInterpolator(np.arange(W), s(aa), axis=1)(bb), s

    # Interpolation only: every measured/fitted node remains an exact knot.
    x_display = x0.copy()
    r_display = r0.copy()
    xy, sx = smooth(x_display)
    rf, sr = smooth(r_display)
    xy = interpolate_sheet(x_display, cyclic=True, factor=UPSAMPLE, linear=True)
    np.testing.assert_allclose(xy[::UPSAMPLE, ::UPSAMPLE], x0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rf[::UPSAMPLE, ::UPSAMPLE], r0, rtol=0, atol=1e-12)
    assert np.allclose(sr(0, 1), sr(H, 1)) and np.allclose(sr(0, 2), sr(H, 2))
    vxy = d["v1_cortex_xy"]
    vrf = d["v1_rf_az_el"]
    vf = d["v1_faces"]
    span = np.ptp(vxy, axis=0).max()
    reg, hits, _, ambiguous_mask = inverse_map(
        rf.reshape(-1, 2), vrf, vxy, vf, np.empty((0, 2)), 0.0
    )
    # Exclude multiple distinct inverse solutions before illustrative gap filling.
    reg[ambiguous_mask] = np.nan
    # Confirm each retained inverse reconstructs its RF via the original cortical mesh.
    ctri = mtri.Triangulation(vxy[:, 0], vxy[:, 1], vf)
    valid_inverse = np.isfinite(reg).all(axis=1)
    reconstructed = np.column_stack(
        [
            mtri.LinearTriInterpolator(ctri, vrf[:, k])(*reg[valid_inverse].T).filled(
                np.nan
            )
            for k in range(2)
        ]
    )
    maxerr = float(
        np.nanmax(
            np.linalg.norm(reconstructed - rf.reshape(-1, 2)[valid_inverse], axis=1)
        )
    )
    print(
        "Inverse check:",
        maxerr,
        "nonfinite forward checks:",
        int(np.sum(~np.isfinite(reconstructed).all(axis=1))),
        flush=True,
    )
    assert maxerr < 1e-6
    # Boundary-tolerance points may lie just outside the cortical triangulation.
    # They are restored from prescribed border correspondence below.
    forward_uncovered = np.flatnonzero(valid_inverse)[
        ~np.isfinite(reconstructed).all(axis=1)
    ]
    reg[forward_uncovered] = np.nan
    reg = reg.reshape(len(aa), len(bb), 2)
    # Known prescribed border correspondence, including periodic interpolation.
    reg[:, 0] = xy[:, 0]
    assert np.allclose(reg[::UPSAMPLE, 0], d["hva_cortex_xy"][:, 0])
    # Fill display registration along each original cortical transect, anchored
    # by its exact fixed border and valid fitted V1 inverse locations.
    registration_interpolated = ~np.isfinite(reg).all(axis=2)
    registration_before_fill = reg.copy()
    # The outermost row has a few unresolved interpolated samples. Complete that
    # cyclic edge between its valid neighbors before interpolating across-strip.
    good_edge = np.isfinite(reg[:, -1]).all(axis=1)
    edge_x = aa[good_edge]
    edge_y = reg[good_edge, -1]
    edge_interp = PchipInterpolator(
        np.concatenate((edge_x - H, edge_x, edge_x + H)),
        np.concatenate((edge_y, edge_y, edge_y)),
        axis=0,
    )
    reg[~good_edge, -1] = edge_interp(aa[~good_edge])
    for i in range(len(aa)):
        good = np.isfinite(reg[i]).all(axis=1)
        missing = ~good
        if missing.any():
            assert good[0] and good[-1], "No extrapolation allowed at either edge"
            reg[i, missing] = PchipInterpolator(
                bb[good], reg[i, good], axis=0, extrapolate=False
            )(bb[missing])
    assert np.isfinite(reg).all()
    assert np.array_equal(
        reg[~registration_interpolated],
        registration_before_fill[~registration_interpolated],
    )

    # Shortest Euclidean cortical distance to the original closed V1/V2 border.
    border_xy = d["hva_cortex_xy"][:, 0]
    segment = np.roll(border_xy, -1, axis=0) - border_xy
    q = xy.reshape(-1, 2)[:, None, :] - border_xy[None, :, :]
    tau = np.clip(
        np.einsum("nsi,si->ns", q, segment) / np.sum(segment**2, axis=1), 0, 1
    )
    distance = np.sqrt(
        np.min(np.sum((q - tau[:, :, None] * segment) ** 2, axis=2), axis=1)
    ).reshape(xy.shape[:2])
    distance[:, 0] = 0  # Exact border under the periodic display interpolation.
    height_scale = 0.26 * span / distance.max()
    height_offset = 0.30 * span
    z = height_offset + height_scale * distance
    assert np.isfinite(z).all() and np.all(distance >= 0)
    return dict(
        panel_c_cortex=xy,
        panel_c_rf=rf,
        panel_c_registered=reg,
        panel_c_height=z,
        panel_c_registration_interpolated=registration_interpolated,
    )


def smooth_ribbon(registered, height):
    """Selected light display smoothing with the inner border held fixed."""
    xyz = np.dstack((registered, height))
    result = xyz.copy()
    result[:, :, :2] = gaussian_filter(
        xyz[:, :, :2], sigma=(3, 1.5, 0), mode=("wrap", "reflect", "nearest")
    )
    result[:, :, :2] += (xyz[:, 0, :2] - result[:, 0, :2])[:, None, :] * np.exp(
        -np.arange(result.shape[1])[None, :, None] / 6
    )
    result[:, 0] = xyz[:, 0]
    return result
