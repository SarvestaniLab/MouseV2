# This code was written for Tensorflow 1. For Tensorflow 2, use the compatibility mode
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np
from .mouse_surface import DEFAULT_BORDER_SVG, generate_mouse_surface_grid

# Defaults for the selected Figure 2 run. The replay supplies its parameters
# explicitly from training/training_config.json.
DEFAULT_MOUSE_BETA1 = 1.45e-5
DEFAULT_MOUSE_BETA2 = 100.0
DEFAULT_MOUSE_BETA3 = 0.03

B3_LEGACY_GLOBAL_AXIS_BALANCE = "legacy_global_axis_balance"
B3_LOCAL_METRIC_ISOTROPY = "local_metric_isotropy"
B3_OBJECTIVES = (B3_LEGACY_GLOBAL_AXIS_BALANCE, B3_LOCAL_METRIC_ISOTROPY)

# map_h is the side that has a boundary
# this geometry is defined by neighborhood()
# for example, (h, w) = (5, 3)
# index:
#    x 0 5 10
#    x 1 6 11
#    x 2 7 12
#    x 3 8 13
#    x 4 9 14
# x is boundary


def neighborhood(h, w):
    """
    Numbers {0...h*w-1} arranged in a h x w grid, produce a list of neighboring numbers.
    The ith element of the list is a list of numbers that neighbor i
    """

    # this line defines how the geometry is coded
    # for (h, w) = (4, 2)
    # 0 4
    # 1 5
    # 2 6
    # 3 7
    x = np.transpose(np.array(range(h * w)).reshape([w, h]))

    if h < 2 or w < 1:
        raise ValueError("cyclic neighborhood requires h >= 2 and w >= 1")
    lst = []
    for i in range(w):
        for j in range(h):
            neighbors = []
            for ii in [-1, 0, 1]:
                for jj in [-1, 0, 1]:
                    if 0 <= i + ii < w:
                        neighbor = x[(j + jj) % h, i + ii]
                        if neighbor not in neighbors:
                            neighbors.append(neighbor)
            lst.append(
                neighbors
            )  # neighbors[indx] 0..h*w-1, inside of borders have eight (nine?),
    return lst


def neighborhood_x(h, w):
    """
    numbers {0...h*w-1} arranged in a hxw grid, produce a list of neighboring numbers.
    The ith element of the list is a list of numbers that neighbor i
    """
    # this line defines how the geometry is coded
    # for (h, w) = (4, 2)
    # 0 4
    # 1 5
    # 2 6
    # 3 7
    x = np.transpose(np.array(range(h * w)).reshape([w, h]))
    if h < 2 or w < 1:
        raise ValueError("cyclic neighborhood requires h >= 2 and w >= 1")
    lst = []
    for column in range(w):
        for row in range(h):
            neighbors = []
            for row_offset in [-3, -2, -1, 0, 1, 2, 3]:
                neighbor = x[(row + row_offset) % h, column]
                if neighbor not in neighbors:
                    neighbors.append(neighbor)
            lst.append(neighbors)
    return lst


def neighborhood_y(h, w):
    """
    numbers {0...h*w-1} arranged in a hxw grid, produce a list of neighboring numbers.
    The ith element of the list is a list of numbers that neighbor i
    """
    # this line defines how the geometry is coded
    # for (h, w) = (4, 2)
    # 0 4
    # 1 5
    # 2 6
    # 3 7
    x = np.transpose(np.array(range(h * w)).reshape([w, h]))
    lst = []
    for column in range(w):
        for row in range(h):
            neighbors = []
            for column_offset in [-3, -2, -1, 0, 1, 2, 3]:
                neighbor_column = column + column_offset
                if 0 <= neighbor_column < w:
                    neighbors.append(x[row, neighbor_column])
            lst.append(neighbors)
    return lst


def make_distance_weighted_smoothness_mask(cortical_x, cortical_y, neighbors):
    """Weight each neighbor pair by inverse squared cortical distance.

    Coordinates have shape ``(h, w)`` and use the same column-major node
    ordering as ``neighborhood``: row is position along the V1/V2 boundary,
    and column is transverse position across the higher-visual-area strip.
    Computing distances from coordinates handles cardinal, diagonal, and
    longer-range neighbors uniformly.
    """
    cortical_x = np.asarray(cortical_x, dtype=np.float64)
    cortical_y = np.asarray(cortical_y, dtype=np.float64)
    if cortical_x.ndim != 2 or cortical_x.shape != cortical_y.shape:
        raise ValueError("cortical_x and cortical_y must be same-shape 2D arrays")
    if not np.all(np.isfinite(cortical_x)) or not np.all(np.isfinite(cortical_y)):
        raise ValueError("cortical coordinates must be finite")

    x_flat = cortical_x.ravel(order="F")
    y_flat = cortical_y.ravel(order="F")
    node_count = x_flat.size
    if len(neighbors) != node_count:
        raise ValueError(
            f"neighbors must contain {node_count} node lists, received {len(neighbors)}"
        )

    mask = np.zeros((node_count, node_count), dtype=np.float64)
    for i, node_neighbors in enumerate(neighbors):
        for j in node_neighbors:
            if not 0 <= j < node_count:
                raise ValueError(f"neighbor index {j} is outside [0, {node_count})")
            if j == i:
                mask[i, j] = 1.0
                continue
            dx = x_flat[i] - x_flat[j]
            dy = y_flat[i] - y_flat[j]
            distance_squared = dx * dx + dy * dy
            if distance_squared == 0.0:
                raise ValueError(f"distinct neighboring nodes {i} and {j} overlap")
            mask[i, j] = 1.0 / distance_squared
    return mask


def make_mask_from_traces(h, w, neighbors, surface_svg=DEFAULT_BORDER_SVG):
    """Build the smoothness mask for the trainable, extrastriate nodes.

    Surface column zero is occupied by the fixed V1 boundary when results are
    assembled for analysis.  The ``w`` trainable columns therefore correspond
    to surface columns 1 through ``w``.
    """
    surface = generate_mouse_surface_grid(svg_path=surface_svg, map_h=h, map_w=w)
    model_nodes = surface[:, 1 : w + 1]
    return make_distance_weighted_smoothness_mask(
        model_nodes[:, :, 0], model_nodes[:, :, 1], neighbors
    )


def make_model_node_area_weights(h, w, surface_svg=DEFAULT_BORDER_SVG):
    """Return unit-mean cortical quadrature weights for trainable nodes."""
    surface = generate_mouse_surface_grid(svg_path=surface_svg, map_h=h, map_w=w)
    next_rows = np.roll(surface, -1, axis=0)
    p00, p10 = surface[:, :-1], next_rows[:, :-1]
    p01, p11 = surface[:, 1:], next_rows[:, 1:]
    cell_area = 0.5 * (
        np.abs(np.cross(p10 - p00, p11 - p00)) + np.abs(np.cross(p11 - p00, p01 - p00))
    )
    weights = np.zeros(surface.shape[:2], dtype=np.float64)
    quarter = 0.25 * cell_area
    previous_quarter = np.roll(quarter, 1, axis=0)
    weights[:, :-1] += quarter + previous_quarter
    weights[:, 1:] += quarter + previous_quarter
    model_weights = weights[:, 1 : w + 1].reshape(-1, order="F")
    if not np.all(np.isfinite(model_weights)) or np.any(model_weights <= 0.0):
        raise ValueError("physical cortical node areas must be finite and positive")
    return model_weights / np.mean(model_weights)


def make_grid8_local_isotropy_geometry(h, w, surface_svg=DEFAULT_BORDER_SVG):
    """Precompute a cyclic, topology-aware local Jacobian operator.

    Each trainable node is connected to its immediate axial and diagonal grid
    neighbors. Rows wrap around the closed cortical ribbon; columns remain
    open. The SVG coordinates define local directions only: the normalized
    isotropy energy below is unchanged by a uniform rescaling of the SVG.
    """
    if h < 3 or w < 2:
        raise ValueError("local metric isotropy requires h >= 3 and w >= 2")
    surface = generate_mouse_surface_grid(svg_path=surface_svg, map_h=h, map_w=w)
    positions = np.asarray(surface[:, 1 : w + 1], dtype=np.float64).reshape(
        (-1, 2), order="F"
    )
    pairs = {
        tuple(sorted((source, target)))
        for source, node_neighbors in enumerate(neighborhood(h, w))
        for target in node_neighbors
        if source != target
    }
    undirected = np.asarray(sorted(pairs), dtype=np.int64).reshape((-1, 2))
    edges = np.concatenate((undirected, undirected[:, ::-1]), axis=0)
    deltas = positions[edges[:, 1]] - positions[edges[:, 0]]
    distance_squared = np.einsum("ij,ij->i", deltas, deltas)
    if np.any(distance_squared <= 0.0):
        raise ValueError("local metric isotropy contains coincident cortical nodes")
    weights = 1.0 / distance_squared

    node_count = len(positions)
    cortical_moment = np.zeros((node_count, 2, 2), dtype=np.float64)
    contributions = weights[:, None, None] * deltas[:, :, None] * deltas[:, None, :]
    np.add.at(cortical_moment, edges[:, 0], contributions)
    valid_nodes = np.linalg.matrix_rank(cortical_moment, tol=1e-10) == 2
    if not np.all(valid_nodes):
        raise ValueError(
            "local metric isotropy requires a two-dimensional neighborhood "
            "at every cortical node"
        )
    inverse_moment = np.linalg.pinv(cortical_moment, rcond=1e-10)
    edge_coefficients = np.einsum(
        "eab,eb->ea",
        inverse_moment[edges[:, 0]],
        weights[:, None] * deltas,
    )

    operator_indices = []
    operator_values = []
    nodes = np.arange(node_count, dtype=np.int64)
    for component in range(2):
        row_offset = component * node_count
        off_diagonal = np.column_stack((row_offset + edges[:, 0], edges[:, 1]))
        diagonal = np.column_stack((row_offset + nodes, nodes))
        diagonal_values = -np.bincount(
            edges[:, 0],
            weights=edge_coefficients[:, component],
            minlength=node_count,
        )
        operator_indices.append(np.concatenate((off_diagonal, diagonal), axis=0))
        operator_values.append(
            np.concatenate((edge_coefficients[:, component], diagonal_values))
        )
    operator_indices = np.concatenate(operator_indices, axis=0)
    operator_values = np.concatenate(operator_values)
    order = np.lexsort((operator_indices[:, 1], operator_indices[:, 0]))
    return {
        "positions": positions,
        "undirected_edges": undirected,
        "valid_nodes": valid_nodes,
        "derivative_operator_indices": operator_indices[order],
        "derivative_operator_values": operator_values[order],
        "derivative_operator_shape": np.asarray(
            (2 * node_count, node_count), dtype=np.int64
        ),
    }


def local_metric_isotropy_energy(visual_positions, geometry, name="reg3_local"):
    """Sum rotation-independent local anisotropy over cortical nodes.

    A weighted least-squares Jacobian maps local cortical displacement to
    visual displacement. For its metric ``G = J J^T``, the normalized
    deviatoric energy is zero for equal stretch in every direction and treats
    axis-aligned and diagonal anisotropy equally. Overall visual magnification
    remains controlled by the other objective terms.
    """
    dtype = visual_positions.dtype.base_dtype
    node_count = len(geometry["valid_nodes"])
    with tf.name_scope(name):
        operator = tf.SparseTensor(
            indices=tf.constant(
                geometry["derivative_operator_indices"], dtype=tf.int64
            ),
            values=tf.constant(geometry["derivative_operator_values"], dtype=dtype),
            dense_shape=tf.constant(
                geometry["derivative_operator_shape"], dtype=tf.int64
            ),
        )
        derivatives = tf.sparse_tensor_dense_matmul(operator, visual_positions)
        jacobian = tf.transpose(tf.reshape(derivatives, (2, node_count, 2)), (1, 0, 2))
        metric = tf.matmul(jacobian, jacobian, transpose_b=True)
        g00 = metric[:, 0, 0]
        g11 = metric[:, 1, 1]
        g01 = metric[:, 0, 1]
        trace = g00 + g11
        epsilon = tf.cast(np.finfo(dtype.as_numpy_dtype).eps, dtype)
        node_energy = (tf.square(g00 - g11) + 4.0 * tf.square(g01)) / (
            tf.square(trace) + epsilon
        )
        return tf.reduce_sum(node_energy)


def make_boundary_mask(n, w):
    """
    n: numer of elements on the boundary
    return a n*n array
    """
    # horizontal connections need to be modulated, colmult as above. only need to consider cols directly above and below (+-1)

    if n < 2:
        raise ValueError("cyclic boundary mask requires at least two rows")
    x = np.zeros([n, n], dtype=np.float64)
    for i in range(n):
        col = np.floor(i / n)  # column of map, 0 to w
        # weights are different depending on row/column, need to use trapezoids or diagonals won't be reciprical
        # neighbours that are out of bounds have already been removed (hopefully)
        colmult = (
            1.0 + col * 1.0 / w
        )  # from 0.5 to 1, corresponding to 6mm inner to 12mm outer circumference (this also gets shrunk by the bevels to ~11mm)
        for j in [-1, 0, 1]:
            x[i, (i + j) % n] = 1 / colmult
    return x


def prototype_squared_distances(x, y):
    """Two-coordinate distances without a (nodes, prototypes, 2) temporary.

    Direct coordinate subtraction avoids cancellation in the alternative
    ||y||² + ||x||² - 2 y·x formula for nearly coincident points.
    """
    dx = y[:, 0:1] - x[None, :, 0]
    dy = y[:, 1:2] - x[None, :, 1]
    return tf.square(dx) + tf.square(dy)


def main_cost(x, y, kappa, stable_logsumexp=False, node_area_weights=None):
    yx_normsq = prototype_squared_distances(x, y)

    log_kernel = -1.0 * yx_normsq / (2.0 * kappa * kappa)
    if node_area_weights is not None:
        weights = np.asarray(node_area_weights, dtype=np.float64)
        node_count = y.shape.as_list()[0]
        if weights.shape != (node_count,):
            raise ValueError("node_area_weights must have one value per node")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("node_area_weights must be finite and positive")
        weights = weights / np.mean(weights)
        log_kernel = (
            log_kernel + tf.log(tf.constant(weights, dtype=tf.float64))[:, None]
        )
    if stable_logsumexp:
        log_yx_gauss_sum = tf.reduce_logsumexp(log_kernel, axis=0)
        yx_cost = -1.0 * kappa * tf.reduce_sum(log_yx_gauss_sum)
        # Retain the historical diagnostic return type. Extremely small sums
        # may underflow here, but the optimized cost and its gradient do not.
        return yx_cost, tf.exp(log_yx_gauss_sum)

    # note that all entries are between 0.0 and 1.0
    yx_gauss = tf.exp(log_kernel)

    yx_gauss_sum = tf.reduce_sum(yx_gauss, axis=0)
    yx_gauss_sum = yx_gauss_sum

    # to protect the log from becoming -inf
    yx_gauss_sum = tf.clip_by_value(
        yx_gauss_sum, clip_value_min=1.0e-306, clip_value_max=1.0e306
    )

    yx_cost = -1.0 * kappa * tf.reduce_sum(tf.log(yx_gauss_sum))
    return yx_cost, yx_gauss_sum


def weighted_edge_energy(left, right, mask, name="edge_energy"):
    """Exact weighted squared-distance objective on nonzero directed edges.

    The mask is fixed NumPy geometry. Self edges are retained: in particular,
    diagonal boundary pairs connect different (trainable/fixed) point sets.
    Reductions can differ from the dense formulation by floating-point roundoff.
    """
    mask = np.asarray(mask, dtype=np.float64)
    rows, cols = np.nonzero(mask)
    with tf.name_scope(name):
        delta = tf.gather(left, rows) - tf.gather(right, cols)
        squared = tf.reduce_sum(tf.square(delta), axis=1)
        return tf.reduce_sum(tf.constant(mask[rows, cols], dtype=tf.float64) * squared)


def generate_elastic_net(
    vf_prototypes,
    boundary_retinotopy,
    y,
    map_h,
    map_w,
    surface_svg=DEFAULT_BORDER_SVG,
    include_reg1=True,
    include_reg2=True,
    include_reg3=True,
    prototype_batch_size=None,
    prototype_batch_seed=None,
    prototype_cost_scale=1.0,
    b3_objective=B3_LEGACY_GLOBAL_AXIS_BALANCE,
    stable_coverage=False,
    physical_node_area_weights=False,
):
    # beta1: within-area smoothness
    # beta2: between-area congruence
    # kappa: annealing parameter
    beta1 = tf.placeholder(tf.float64, shape=(), name="b1")
    beta2 = tf.placeholder(tf.float64, shape=(), name="b2")
    beta3 = tf.placeholder(tf.float64, shape=(), name="b3")  # even x,y gradients
    kappa = tf.placeholder(tf.float64, shape=(), name="k")

    prototype_count = len(vf_prototypes)
    if prototype_batch_size is not None:
        prototype_batch_size = int(prototype_batch_size)
        if prototype_batch_size <= 0 or prototype_batch_size > prototype_count:
            raise ValueError(
                "prototype_batch_size must be in [1, {}]; got {}".format(
                    prototype_count, prototype_batch_size
                )
            )
    x_full = tf.constant(vf_prototypes, dtype=tf.float64, name="x_full")
    if prototype_batch_size is None or prototype_batch_size == prototype_count:
        x = tf.identity(x_full, name="x")
    else:
        # A fresh without-replacement subset is drawn for every optimizer step.
        # Keeping the batch at the historical 2,000 prototypes preserves the
        # data-term mass, and therefore the meaning of the frozen beta values,
        # while a larger pool supplies denser visual-field coverage over time.
        x = tf.random_shuffle(
            x_full, seed=prototype_batch_seed, name="prototype_shuffle"
        )[:prototype_batch_size]
    b = tf.constant(boundary_retinotopy, dtype=tf.float64, name="b")

    # this is the boundary of the cortical map
    yb = y[:map_h]

    yx_cost, yx_gauss = main_cost(
        x,
        y,
        kappa,
        stable_logsumexp=stable_coverage,
        node_area_weights=(
            make_model_node_area_weights(map_h, map_w, surface_svg=surface_svg)
            if physical_node_area_weights
            else None
        ),
    )
    prototype_cost_scale = float(prototype_cost_scale)
    if not np.isfinite(prototype_cost_scale) or prototype_cost_scale <= 0:
        raise ValueError("prototype_cost_scale must be finite and positive")
    if prototype_cost_scale != 1.0:
        yx_cost = tf.multiply(
            tf.constant(prototype_cost_scale, dtype=tf.float64),
            yx_cost,
            name="scaled_prototype_cost",
        )

    # Masks are static geometry. Evaluate only their nonzero directed edges;
    # retain both orientations and the original weights (no normalization).
    def regularizer(mask, left=y, right=y, name="regularizer"):
        return weighted_edge_energy(left, right, mask, name=name)

    if include_reg1:
        reg1 = regularizer(
            make_mask_from_traces(
                map_h, map_w, neighborhood(map_h, map_w), surface_svg=surface_svg
            ),
            name="reg1",
        )
    else:
        reg1 = tf.constant(0.0, dtype=tf.float64, name="reg1_disabled")

    if include_reg2:
        reg2 = regularizer(
            make_boundary_mask(map_h, map_w), left=yb, right=b, name="reg2"
        )
    else:
        reg2 = tf.constant(0.0, dtype=tf.float64, name="reg2_disabled")

    if b3_objective not in B3_OBJECTIVES:
        raise ValueError(
            "b3_objective must be one of {}; got {!r}".format(
                B3_OBJECTIVES, b3_objective
            )
        )

    if include_reg3 and b3_objective == B3_LEGACY_GLOBAL_AXIS_BALANCE:
        ex = regularizer(
            make_mask_from_traces(
                map_h, map_w, neighborhood_x(map_h, map_w), surface_svg=surface_svg
            ),
            name="reg3_x",
        )
        ey = regularizer(
            make_mask_from_traces(
                map_h, map_w, neighborhood_y(map_h, map_w), surface_svg=surface_svg
            ),
            name="reg3_y",
        )
        reg3 = tf.abs(ex - ey)
    elif include_reg3:
        reg3 = local_metric_isotropy_energy(
            y,
            make_grid8_local_isotropy_geometry(map_h, map_w, surface_svg=surface_svg),
        )
    else:
        reg3 = tf.constant(0.0, dtype=tf.float64, name="reg3_disabled")

    ##### total cost
    cost = yx_cost + beta1 * reg1 + beta2 * reg2 + beta3 * reg3

    total_cost = cost
    partial_costs = [yx_cost, reg1, reg2, reg3]
    tf_placeholders = [beta1, beta2, beta3, kappa]
    return total_cost, partial_costs, tf_placeholders, yx_gauss


def generate_optimizer(cost, eta0=0.05, m=0.8, eta_decay=0.99):
    """
    cost - the cost function
    eta0 - initial learning rate
    m - momentum
    """

    global_step = tf.Variable(0, trainable=False)
    current_eta = tf.train.exponential_decay(
        eta0, global_step, 10, eta_decay, staircase=False
    )

    opt = tf.train.RMSPropOptimizer(current_eta, momentum=m).minimize(
        cost, global_step=global_step
    )
    return opt
