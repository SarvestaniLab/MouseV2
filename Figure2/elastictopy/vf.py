########################################
# generate landmarks on the visual field 
########################################

import numpy as np
from numpy import linalg as LA
import csv
from scipy.interpolate import PchipInterpolator


# Finite extrema in the repository's extracted Zhuang Fig. 9c altitude array.
ALLEN_ALTITUDE_RANGE_DEG = (-25.061082070208943, 30.028485360851864)
# Electrophysiological V1 center coverage from the peripheral iso-elevation
# contours in Wagor et al. (1980), Fig. 4A.
WAGOR_V1_ALTITUDE_RANGE_DEG = (-40.0, 60.0)
WAGOR_V1_AZIMUTH_RANGE_DEG = (0.0, 140.0)
MOUSE_TEARDROP_ALTITUDE_RANGE_DEG = WAGOR_V1_ALTITUDE_RANGE_DEG
MOUSE_TEARDROP_AZIMUTH_RANGE_DEG = WAGOR_V1_AZIMUTH_RANGE_DEG
LANDMARK_ALTITUDE_OFFSET_DEG = 20.0
MASKED_SPIRAL_PILOT_COUNT = 4096
MASKED_SPIRAL_SAFETY_FACTOR = 1.01

# Sparse modelling anchors for the smooth synthetic envelope. These encode the
# deliberately idealized black curves in Tracing.png rather than tracing their
# pixels. PCHIP joins them with C1-continuous, shape-preserving cubic segments.
MOUSE_TEARDROP_UPPER_ANCHORS_DEG = np.asarray([
    [0.0, 40.0],
    [25.0, 57.0],
    [60.0, 60.0],
    [90.0, 57.0],
    [120.0, 38.0],
    [140.0, -13.0],
])
MOUSE_TEARDROP_LOWER_ANCHORS_DEG = np.asarray([
    [0.0, -16.0],
    [25.0, -29.0],
    [65.0, -40.0],
    [68.0, -40.0],
    [82.0, -24.0],
    [110.0, -17.0],
    [140.0, -13.0],
])
_MOUSE_TEARDROP_UPPER_SPLINE = PchipInterpolator(
    MOUSE_TEARDROP_UPPER_ANCHORS_DEG[:, 0],
    MOUSE_TEARDROP_UPPER_ANCHORS_DEG[:, 1],
    extrapolate=False,
)
_MOUSE_TEARDROP_LOWER_SPLINE = PchipInterpolator(
    MOUSE_TEARDROP_LOWER_ANCHORS_DEG[:, 0],
    MOUSE_TEARDROP_LOWER_ANCHORS_DEG[:, 1],
    extrapolate=False,
)


def mouse_teardrop_altitude_bounds():
    """Return the Wagor-calibrated extrema of the synthetic V1 envelope."""
    return MOUSE_TEARDROP_ALTITUDE_RANGE_DEG


def mouse_teardrop_altitude_limits(azimuth_deg):
    """Return the sparse-anchor parametric V1 envelope at each azimuth.

    Separate shape-preserving cubic curves define the upper and lower borders;
    the vertical-meridian segment and common posterior tip close the envelope.
    The curve is intentionally not a fit to either paper's detailed boundary.
    """
    azimuth = np.asarray(azimuth_deg, dtype=np.float64)
    azimuth_min, azimuth_max = MOUSE_TEARDROP_AZIMUTH_RANGE_DEG
    inside = (azimuth >= azimuth_min) & (azimuth <= azimuth_max)
    lower = np.where(inside, _MOUSE_TEARDROP_LOWER_SPLINE(azimuth), np.nan)
    upper = np.where(inside, _MOUSE_TEARDROP_UPPER_SPLINE(azimuth), np.nan)
    return lower, upper


def required_spiral_radius(polygon, altitude_offset=LANDMARK_ALTITUDE_OFFSET_DEG):
    """Return the radius needed to reach a polygon from the spiral center."""
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) == 0:
        raise ValueError("polygon must have shape (n, 2)")
    sampling_center = np.asarray([0.0, float(altitude_offset)])
    return float(np.max(np.linalg.norm(polygon - sampling_center, axis=1)))


### generate landmarks on the visual field
def spiral(c, r0, r1, n, shuffleP=True):
    """Make a spiral with ``n`` points between radii ``r0`` and ``r1``.

    The historical radial transform uses ``c`` as its density parameter. Its
    resulting areal density is proportional to ``r**(2*c/(1-c))``, rather
    than ``r**c``. Negative values concentrate points near the origin.
    """
    def f(r):
        return np.sqrt(np.power(r, 1.0-c)/(1.0-c))

    n0 = np.power(r0 * r0 * (1.0-c), 1.0/(1.0-c))
    n1 = np.power(r1 * r1 * (1.0-c), 1.0/(1.0-c))

    lst = []
    for k in range(1, n+1):
        d = f( (n1-n0)/(n-1)*k + n0-(n1-n0)/(n-1) )
        theta = 3.1415926 * (3.0 - np.sqrt(5.0)) * (k-1.0)
        x = d * np.cos(theta)
        y = d * np.sin(theta)
        lst.append((x,y))
    res = np.asarray(lst, dtype=np.float64)
    if shuffleP:
        np.random.shuffle(res)
    return res

def generate_RVF_landmarks(c, r0, r1, n, noise=False):
    # generate landmarks on the RIGHT visual field
    # returns a numpy array

    # generate twice as many points because spiral() covers the entire visual field (left and right)
    pts = spiral(c, r0, r1, 2*n)

    # boolean array, find those that are in the right hemifield
    pts_idx = pts[:, 0] >= 0.0

    # make a mask array
    pts_idx = np.stack([pts_idx, pts_idx])
    pts_idx = np.transpose(pts_idx)

    # keep points whose x coordinate is larger or equal to 0
    zz = np.extract(pts_idx, pts)
    # but since the extracted is flattend, I have to reshape it
    zz = zz.reshape([int(zz.shape[0]/2), 2])

    #if noise:
    #    zz = zz + np.random.normal(0.0, 0.1, zz.shape)

    return zz


def generate_RVF_teardrop_landmarks(c, r0, r1, n, noise=False):
    # generate landmarks on the RIGHT visual field masked to the teardrop shape of mouse visual field
    # returns a numpy array

    # Generate enough candidates to return exactly n landmarks. The historical
    # fixed multiplier was coupled to one particular mask size, so changing
    # biological coverage also changed the number of model prototypes.
    candidate_count = int(np.ceil(5.0 * n))
    while True:
        pts = spiral(c, r0, r1, candidate_count)

        # Shift the sampling distribution upward to match the mouse visual axis.
        pts[:, 1] = pts[:, 1] + LANDMARK_ALTITUDE_OFFSET_DEG
        y = pts[:, 1]
        x = pts[:, 0]

        keep = np.ones(x.shape, dtype=bool)
        keep[x < 0] = False
        low, high = mouse_teardrop_altitude_limits(x[x > 0])
        keep[x > 0] = ((y[x > 0]) > low) & ((y[x > 0]) < high)

        accepted = pts[keep]
        if accepted.shape[0] >= n:
            return accepted[:n]
        candidate_count *= 2


def points_inside_polygon(points, polygon):
    """Return a point-in-polygon mask using ray casting."""
    points = np.asarray(points, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 4:
        raise ValueError("polygon must have shape (m, 2) with at least four points")

    x = points[:, 0]
    y = points[:, 1]
    x0 = polygon[:, 0]
    y0 = polygon[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    crossing = ((y0[:, None] > y) != (y1[:, None] > y)) & (
        x
        < (x1 - x0)[:, None] * (y - y0[:, None])
        / (y1 - y0 + np.finfo(float).eps)[:, None]
        + x0[:, None]
    )
    return np.logical_xor.reduce(crossing, axis=0)


def generate_polygon_landmarks(
    c, r0, r1, n, polygon, altitude_offset=LANDMARK_ALTITUDE_OFFSET_DEG
):
    """Generate exactly ``n`` regularly spaced prototypes inside a polygon.

    A deterministic pilot estimates the fraction of the full spiral retained
    by the mask. The final candidate lattice is sized from that acceptance
    rate, avoiding the historical 5x-to-10x jump followed by random thinning.
    Any small final excess is removed at evenly spaced indices so the
    golden-angle lattice retains its local spacing.
    """
    if n <= 0:
        raise ValueError("n must be greater than zero")

    def accepted_candidates(candidate_count):
        points = spiral(c, r0, r1, candidate_count, shuffleP=False)
        points[:, 1] += altitude_offset
        return points[points_inside_polygon(points, polygon)]

    pilot_count = max(MASKED_SPIRAL_PILOT_COUNT, int(n))
    pilot_accepted = accepted_candidates(pilot_count)
    acceptance_rate = pilot_accepted.shape[0] / pilot_count
    if acceptance_rate <= 0:
        raise ValueError("polygon accepts no spiral candidates")

    candidate_count = int(np.ceil(
        n / acceptance_rate * MASKED_SPIRAL_SAFETY_FACTOR
    ))
    for _ in range(8):
        accepted = accepted_candidates(candidate_count)
        if accepted.shape[0] >= n:
            break
        observed_rate = accepted.shape[0] / candidate_count
        if observed_rate <= 0:
            raise ValueError("polygon accepts no spiral candidates")
        candidate_count = int(np.ceil(
            n / observed_rate * MASKED_SPIRAL_SAFETY_FACTOR
        ))
    else:
        raise RuntimeError("could not generate enough masked spiral candidates")

    if accepted.shape[0] == n:
        return accepted
    selected = np.rint(
        np.linspace(0, accepted.shape[0] - 1, n)
    ).astype(int)
    return accepted[selected]


def generate_vf_landmarks(c, r0, r1, n, vf='full', filename="", noise=False):
    assert vf in ['upper', 'lower', 'full','tear']

    if vf in ['upper', 'lower']:
        NN = n * 2
    else:
        NN = n
    
    if vf=='tear':
        pts = generate_RVF_teardrop_landmarks(c, r0, r1, NN, noise)
    else:
        pts = generate_RVF_landmarks(c, r0, r1, NN, noise)

    # boolean array, find those that are in the right quadrant
    if vf=='upper':
        pts_idx = pts[:, 1] >= 0.0
    elif vf=='lower':
        pts_idx = pts[:, 1] <= 0.0
    else:
        # this should be all True
        pts_idx = pts[:, 0] >= 0.0
    
    # make a mask array
    pts_idx = np.stack([pts_idx, pts_idx])
    pts_idx = np.transpose(pts_idx)

    zz = np.extract(pts_idx, pts)
    # but since the extracted is flattend, I have to reshape it
    zz = zz.reshape([int(zz.shape[0]/2), 2])

    if filename:
        export_vf_landmarks(zz, filename=filename)

    return zz



def export_vf_landmarks(pts, filename='vf.csv'):
    with open(filename, mode='w') as vf_csv:
        writer = csv.writer(vf_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for pt in pts:
            writer.writerow(pt)
