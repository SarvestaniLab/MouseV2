"""Equivalence against the original dense objective, including seam edges."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'elastictopy'))
import elastic_net_mouse as net

tf = net.tf


def dense_energy(left, right, mask, name='dense'):
    delta = tf.expand_dims(left, 1) - tf.expand_dims(right, 0)
    return tf.reduce_sum(tf.constant(mask, dtype=tf.float64) *
                         tf.einsum('ijk,ijk->ij', delta, delta))


class EdgeRegularizerTests(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_actual_geometry_costs_and_gradients(self):
        rng = np.random.RandomState(3)
        y = tf.Variable(rng.normal(size=(640, 2)) * 25, dtype=tf.float64)
        b = tf.constant(rng.normal(size=(80, 2)) * 25, dtype=tf.float64)
        masks = [net.make_mask_from_traces(80, 8, fn(80, 8))
                 for fn in (net.neighborhood, net.neighborhood_x, net.neighborhood_y)]
        pairs = [(y, y, m) for m in masks] + [(y[:80], b, net.make_boundary_mask(80, 8))]
        old = [dense_energy(*p) for p in pairs]
        new = [net.weighted_edge_energy(*p) for p in pairs]
        # The absolute directional penalty and its gradient must also agree.
        old.append(tf.abs(old[1]-old[2])); new.append(tf.abs(new[1]-new[2]))
        grads = [[tf.convert_to_tensor(tf.gradients(v, y)[0]) for v in vv] for vv in (old, new)]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            a, b, ga, gb = sess.run([old, new, grads[0], grads[1]])
        np.testing.assert_allclose(a, b, rtol=2e-14, atol=1e-8)
        np.testing.assert_allclose(ga, gb, rtol=2e-12, atol=1e-8)

    def test_empty_mask_and_boundary_diagonal(self):
        a = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float64)
        b = tf.zeros_like(a)
        with tf.Session() as sess:
            empty, diagonal = sess.run([net.weighted_edge_energy(a, b, np.zeros((2, 2))),
                                        net.weighted_edge_energy(a, b, np.eye(2))])
        self.assertEqual(empty, 0.)
        self.assertEqual(diagonal, 30.)

    def test_short_optimizer_trajectory(self):
        rng = np.random.RandomState(9)
        initial = rng.normal(size=(32, 2))
        prototypes = rng.normal(size=(60, 2)) * 10
        boundary = rng.normal(size=(8, 2))
        def trajectory(dense):
            tf.reset_default_graph()
            y = tf.get_variable('y', initializer=initial, dtype=tf.float64)
            implementation = dense_energy if dense else net.weighted_edge_energy
            with patch.object(net, 'weighted_edge_energy', implementation):
                cost, _, placeholders, _ = net.generate_elastic_net(prototypes, boundary, y, 8, 4)
            opt = net.generate_optimizer(cost, eta0=.1)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                k = 30.
                for _ in range(100):
                    sess.run(opt, dict(zip(placeholders, [.00002, 20., .00005, k])))
                    k -= k*.003
                return sess.run(y)
        np.testing.assert_allclose(trajectory(True), trajectory(False), rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
