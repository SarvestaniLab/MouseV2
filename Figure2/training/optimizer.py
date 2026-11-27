"""Float64 GPU stencil used by the selected Figure 2 run."""

import numpy as np
from elastictopy import elastic_net_mouse as en
from elastictopy.elastic_net_mouse import tf

H, W = 160, 16
N = H * W


def dense_energy(y, geometry, name="reg3_local"):
    # Extract every coefficient, including diagonal, from the actual Mouse D.
    # Keeping the diagonal avoids changing flat-map cancellation behavior by
    # rewriting D@y as weighted neighbor differences.
    indices = geometry["derivative_operator_indices"]
    values = geometry["derivative_operator_values"]
    coeff = np.zeros((2, 3, 3, W, H))
    for (rr, cc), value in zip(indices, values):
        component, node = divmod(int(rr), N)
        col, row = divmod(node, H)
        nc, nr = divmod(int(cc), H)
        dr = (nr - row) % H
        if dr == H - 1:
            dr = -1
        dc = nc - col
        assert dr in (-1, 0, 1) and dc in (-1, 0, 1)
        coeff[component, dr + 1, dc + 1, col, row] += value
    grid = tf.reshape(y, (W, H, 2))
    parts = []
    for component in range(2):
        terms = []
        for dr in (-1, 0, 1):
            shifted = tf.roll(grid, shift=-dr, axis=1)
            for dc in (-1, 0, 1):
                if dc == -1:
                    z = tf.concat([tf.zeros_like(shifted[:1]), shifted[:-1]], axis=0)
                elif dc == 1:
                    z = tf.concat([shifted[1:], tf.zeros_like(shifted[:1])], axis=0)
                else:
                    z = shifted
                terms.append(
                    tf.constant(
                        coeff[component, dr + 1, dc + 1, :, :, None], dtype=tf.float64
                    )
                    * z
                )
        parts.append(tf.reshape(tf.add_n(terms), (N, 2)))
    jac = tf.stack(parts, axis=1)
    metric = tf.matmul(jac, jac, transpose_b=True)
    a, b, c = metric[:, 0, 0], metric[:, 1, 1], metric[:, 0, 1]
    return tf.reduce_sum(
        (tf.square(a - b) + 4 * tf.square(c))
        / (tf.square(a + b) + np.finfo(np.float64).eps)
    )


def config():
    c = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        allow_soft_placement=True,
    )
    c.gpu_options.allow_growth = True
    return c
