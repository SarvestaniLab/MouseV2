# This code was written for Tensorflow 1. For Tensorflow 2, use the compatibility mode
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from helpers import checkpoint_stem, full_filename
from mouse_surface import DEFAULT_BORDER_SVG, generate_mouse_surface_grid

# Provisional defaults for the corrected Waters workflow.  Beta1 is the center
# of the single-seed smoothness window that reduced RF-grid distortion without
# collapsing most field-sign reversals.  Beta2 balances V1-boundary RMSE against
# coverage and near-zero cells.  Beta3 is the low, nonzero value that minimized
# along-versus-outward RF neighbor-RMS imbalance; the historical 3e-12 value is
# effectively zero at the current cost scale.  Confirm all three across seeds.
DEFAULT_MOUSE_BETA1 = 2.25e-5
DEFAULT_MOUSE_BETA2 = 0.2
DEFAULT_MOUSE_BETA3 = 3.0e-6

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
    x = np.transpose(np.array(range(h*w)).reshape([w, h]))

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
            lst.append(neighbors) #neighbors[indx] 0..h*w-1, inside of borders have eight (nine?),
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
    x = np.transpose(np.array(range(h*w)).reshape([w, h]))
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
    x = np.transpose(np.array(range(h*w)).reshape([w, h]))
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
    surface = generate_mouse_surface_grid(
        svg_path=surface_svg, map_h=h, map_w=w
    )
    model_nodes = surface[:, 1 : w + 1]
    return make_distance_weighted_smoothness_mask(
        model_nodes[:, :, 0], model_nodes[:, :, 1], neighbors
    )

def make_boundary_mask(n,w):
    """
        n: numer of elements on the boundary
        return a n*n array
    """
    #horizontal connections need to be modulated, colmult as above. only need to consider cols directly above and below (+-1)

    if n < 2:
        raise ValueError("cyclic boundary mask requires at least two rows")
    x = np.zeros([n, n], dtype=np.float64)
    for i in range(n):
        col=np.floor(i/n) # column of map, 0 to w
        # weights are different depending on row/column, need to use trapezoids or diagonals won't be reciprical
        #neighbours that are out of bounds have already been removed (hopefully)
        colmult=1.0+col*1.0/w  #from 0.5 to 1, corresponding to 6mm inner to 12mm outer circumference (this also gets shrunk by the bevels to ~11mm)
        for j in [-1, 0, 1]:
            x[i, (i + j) % n] = 1/colmult
    return x

def initialize_map(h, w, x, filename="", random_seed=None, initial_values=None):
    """
        Initialize cortical map, save to a file, and create a tensorflow variable "y"

        initialze a cortical map of the size map_w * map_h
        each point on the map is randomly assigned to prototypes on the visual field (x) with some added noise
    """

    # if random_seed:
        # np.random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    map_size = h * w

    if initial_values is None:
        idx = np.random.randint(x.shape[0], size = map_size)
        y = x[idx, :] # randomly sample from x with replacement
        n = np.random.normal(0.0, 1.0, y.shape) # the standard deviation is set to 5.0 deg
        res = y + n
    else:
        res = np.asarray(initial_values, dtype=np.float64)
        if res.shape != (map_size, 2):
            raise ValueError(
                "initial_values must have shape ({}, 2); got {}".format(
                    map_size, res.shape
                )
            )
        res = res.copy()

    if filename:
        np.savetxt(filename, res, fmt='%f', delimiter=",")

    y = tf.get_variable("y", dtype = tf.float64, initializer = res)
    
    return y

def prototype_squared_distances(x, y):
    """Two-coordinate distances without a (nodes, prototypes, 2) temporary.

    Direct coordinate subtraction avoids cancellation in the alternative
    ||y||² + ||x||² - 2 y·x formula for nearly coincident points.
    """
    dx = y[:, 0:1] - x[None, :, 0]
    dy = y[:, 1:2] - x[None, :, 1]
    return tf.square(dx) + tf.square(dy)


def main_cost(x, y, kappa):
    yx_normsq = prototype_squared_distances(x, y)

    # note that all entries are between 0.0 and 1.0
    yx_gauss = tf.exp(-1.0 * yx_normsq / (2.0 * kappa * kappa))

    yx_gauss_sum = tf.reduce_sum(yx_gauss, axis=0)
    yx_gauss_sum = yx_gauss_sum

    # to protect the log from becoming -inf 
    yx_gauss_sum = tf.clip_by_value(yx_gauss_sum, clip_value_min=1.0e-306, clip_value_max=1.0e+306)

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
    vf_prototypes, boundary_retinotopy, y, map_h, map_w,
    surface_svg=DEFAULT_BORDER_SVG,
    include_reg1=True,
    include_reg2=True,
    include_reg3=True,
):
    # beta1: within-area smoothness
    # beta2: between-area congruence
    # kappa: annealing parameter
    beta1 = tf.placeholder(tf.float64, shape=(), name="b1")
    beta2 = tf.placeholder(tf.float64, shape=(), name="b2")
    beta3 = tf.placeholder(tf.float64, shape=(), name="b3") #even x,y gradients
    kappa = tf.placeholder(tf.float64, shape=(), name="k")

    x = tf.constant(vf_prototypes, dtype=tf.float64, name='x')
    b = tf.constant(boundary_retinotopy, dtype=tf.float64, name='b')

    # this is the boundary of the cortical map
    yb = y[:map_h]

    yx_cost, yx_gauss = main_cost(x, y, kappa)
    
    # Masks are static geometry. Evaluate only their nonzero directed edges;
    # retain both orientations and the original weights (no normalization).
    def regularizer(mask, left=y, right=y, name="regularizer"):
        return weighted_edge_energy(left, right, mask, name=name)

    if include_reg1:
        reg1 = regularizer(make_mask_from_traces(
            map_h, map_w, neighborhood(map_h, map_w), surface_svg=surface_svg
        ), name="reg1")
    else:
        reg1 = tf.constant(0.0, dtype=tf.float64, name="reg1_disabled")

    if include_reg2:
        reg2 = regularizer(make_boundary_mask(map_h, map_w),
                           left=yb, right=b, name="reg2")
    else:
        reg2 = tf.constant(0.0, dtype=tf.float64, name="reg2_disabled")

    if include_reg3:
        ex = regularizer(make_mask_from_traces(
            map_h, map_w, neighborhood_x(map_h, map_w), surface_svg=surface_svg
        ), name="reg3_x")
        ey = regularizer(make_mask_from_traces(
            map_h, map_w, neighborhood_y(map_h, map_w), surface_svg=surface_svg
        ), name="reg3_y")
        reg3 = tf.abs(ex - ey)
    else:
        reg3 = tf.constant(0.0, dtype=tf.float64, name="reg3_disabled")

    ##### total cost
    cost = yx_cost + beta1 * reg1 + beta2 * reg2 + beta3 *reg3

    total_cost = cost
    partial_costs = [yx_cost, reg1, reg2, reg3]
    tf_placeholders = [beta1, beta2, beta3, kappa]
    return total_cost, partial_costs, tf_placeholders, yx_gauss

def generate_optimizer(cost, eta0 = 0.05, m = 0.8, eta_decay=0.99):
    """
    cost - the cost function
    eta0 - initial learning rate
    m - momentum
    """
    
    global_step = tf.Variable(0, trainable=False)
    current_eta = tf.train.exponential_decay(eta0, global_step, 10, eta_decay, staircase=False)

    opt = tf.train.RMSPropOptimizer(current_eta, momentum = m).minimize(cost, global_step = global_step)
    return opt

def save_map(sess, y, b1, b2, b3, i, outdir="", save_interval=0, quiet=False, suffix=''):
    if (outdir!="") and save_interval>0:
        if i%save_interval==0:
            filename = checkpoint_stem((b1, b2, b3), i)
            zz = sess.run(y)
            
            if (not quiet):
                print("saving {filename}".format(filename=filename))
                
            np.savetxt(full_filename(filename, outdir, ext=".data", suffix=suffix),
                       np.array(zz), fmt = "%f", delimiter=",")

def report_and_log(partial_costs, tf_placeholders, i, k, b1, b2, b3, sess, reportP=True, logP=True, logfile='log'):
    if not reportP and not logP:
        return

    yx_cost = partial_costs[0]
    reg1    = partial_costs[1]
    reg2    = partial_costs[2]
    reg3    = partial_costs[3]
    kappa   = tf_placeholders[3]

    [e0, e1, e2, e3] = sess.run([yx_cost, reg1, reg2, reg3], {kappa: k})
    tt = e0 + b1*e1 + b2*e2 + b3*e3
    if reportP:
        print("%d\t%5f\t%6.4f"%(i, k, tt)) # total cost
        print("\t\t%10.4f\t%10.4f\t%10.4f"%(e0, e1, e2))

    if logP and logfile:
        logstring = "%10.4f,%10.4f,%10.4f,%d,%10.4f,%10.4f,%10.4f,%10.4f,%10.4f,%10.4f\n"%(b1, b2, b3, i, k, tt, e0, e1, e2, e3)
        with open(logfile, 'a') as logfp:
            logfp.write(logstring)

def train(opt, total_cost, partial_costs, tf_placeholders, y, b1, b2, b3, outdir="", save_interval=0, n=1000, k0=30.0, kr=0.005, quiet_save=False, report_iterations = True, log_iterations = True, logfile='log', suffix='', yx_gauss=None):
    """
    b1:              weights for the smoothness term
    b2:              weights for the congruency term
    n:               number of iterations
    k0:              initial annealing parameter
    kr:              reduction of the annealing parameter at each iteration
    outdir:          where to save the results.
    save_intervals:  save map every several iterations. if set to 0, saving is disabled
    """

    yx_cost = partial_costs[0]
    reg1    = partial_costs[1]
    reg2    = partial_costs[2]
    reg3    = partial_costs[3]
    beta1   = tf_placeholders[0]
    beta2   = tf_placeholders[1]
    beta3   = tf_placeholders[2]
    kappa   = tf_placeholders[3]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # save initial map
        #      always save the initial map (by setting save_interval to 1)
        save_map(sess, y, b1, b2, b3, 0, outdir, save_interval=1, quiet=quiet_save, suffix=suffix)

        k = k0
        for i in range(1, n+1):
            # itertaively optimize the total cost function
            sess.run(opt, {kappa: k, beta1: b1, beta2: b2, beta3: b3})

            report_and_log(partial_costs, tf_placeholders, i, k, b1, b2, b3, sess,
                           reportP = report_iterations,
                           logP = log_iterations,
                           logfile=full_filename(logfile, outdir, suffix=suffix))

            if i>0:
                save_map(sess, y, b1, b2, b3, i, outdir, save_interval, quiet=quiet_save, suffix=suffix)

            k = k - k*kr
    
        # Save the final result unless the regular checkpoint cadence just wrote it.
        if save_interval <= 0 or i % save_interval:
            save_map(sess, y, b1, b2, b3, i, outdir, save_interval=1, quiet=quiet_save, suffix=suffix)
    
        report_and_log(partial_costs, tf_placeholders, i, k, b1, b2, b3, sess,
                       reportP = False,
                       logP = log_iterations,
                       logfile = full_filename(logfile, outdir, suffix=suffix))

def optimize(x0, b0, b1, b2, b3, map_h, map_w, y, outdir="", save_interval=0, n=1000, k0=30.0, kr=0.005, eta0 = 0.05, m = 0.8, quiet_save = False, report_iterations = True, log_iterations = True, logfile="log", suffix="", surface_svg=DEFAULT_BORDER_SVG):
    """
    Easy interface to train()
    x0             - visual field prototypes
    b0             - boundary retinotopy
    b1             - model weights
    b2             - model weights
    map_h          - height of the cortical map
    map_w          - width of the cortical map
    y              - the cortical map to learn (tf variable)
    outdir:        directory to save the file to
    save_interval: save map for how many iterations (if 0, do not save intermediate maps)
    n:             how many iterations to update
    quiet_save:    do not print "saving to...."
    report_iterations: report every iteration?
    surface_svg:       two-path cortical boundary SVG; defaults to the Waters
                       conditional-normal tracing
    """
    
    diagnostics_enabled = report_iterations or log_iterations
    (total_cost, partial_costs, tf_placeholders, yx_gauss) = generate_elastic_net(
        x0,
        b0,
        y,
        map_h,
        map_w,
        surface_svg=surface_svg,
        include_reg1=(b1 != 0 or diagnostics_enabled),
        include_reg2=(b2 != 0 or diagnostics_enabled),
        include_reg3=(b3 != 0 or diagnostics_enabled),
    )
    opt = generate_optimizer(total_cost, eta0 = eta0, m = m)
    
    train(opt,
          total_cost, partial_costs, tf_placeholders, y, b1, b2, b3,
          outdir = outdir,
          save_interval = save_interval,
          n = n,
          k0 = k0,
          kr = kr,
          quiet_save = quiet_save,
          report_iterations = report_iterations,
          log_iterations = log_iterations,
          logfile = logfile,
          suffix = suffix,
          yx_gauss = yx_gauss
    )
