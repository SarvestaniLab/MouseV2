"""Geometry and runtime selection checks for the optional registered border."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'elastictopy'))
from mouse_surface import (
    DEFAULT_BORDER_SVG, WATERS_CONDITIONAL_BORDER_SVG,
    WATERS_REGISTERED_MOUSE_MEDIAN_BORDER_SVG as CANDIDATE,
    WATERS_INDIVIDUAL_V1_BORDER_SVG,
    generate_mouse_surface_grid, generate_mouse_surface_distances, load_svg_paths,
)


def cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


class RegisteredBorderTests(unittest.TestCase):
    candidate = CANDIDATE

    def test_candidate_preserves_v1_and_has_simple_enclosing_outer_loop(self):
        from matplotlib.path import Path as PolygonPath
        outer, inner = load_svg_paths(self.candidate, step=1)
        _, old_inner = load_svg_paths(WATERS_CONDITIONAL_BORDER_SVG, step=1)
        np.testing.assert_array_equal(inner, old_inner)
        np.testing.assert_array_equal(outer[0], outer[-1])
        self.assertTrue(PolygonPath(outer).contains_points(inner).all())
        # Remove repeated vertices inserted by SVG line sampling.
        outer = outer[np.r_[True, np.any(np.diff(outer, axis=0) != 0, axis=1)]]
        start, end = outer[:-1], outer[1:]
        for i in range(len(start)):
            j = np.arange(i+2, len(start))
            if i == 0:
                j = j[j != len(start)-1]
            a, b = start[i], end[i]
            c, d = start[j], end[j]
            proper_crossing = ((cross(b-a,c-a)*cross(b-a,d-a) < 0)
                               & (cross(d-c,a-c)*cross(d-c,b-c) < 0))
            self.assertFalse(proper_crossing.any())

    def test_option_uses_normal_grid_with_unfolded_cells_and_cyclic_distances(self):
        grid = generate_mouse_surface_grid(self.candidate)
        np.testing.assert_array_equal(grid, generate_mouse_surface_grid(self.candidate, normal_across=True))
        self.assertEqual(grid.shape, (80, 9, 2))
        inner = grid[:, 0]
        radial = grid[:, -1]-inner
        tangent = np.roll(inner,-1,axis=0)-np.roll(inner,1,axis=0)
        np.testing.assert_allclose(np.sum(radial*tangent,axis=1),0,atol=1e-12)
        nxt = np.roll(grid,-1,axis=0)
        # Both triangles of every quadrilateral, including the periodic seam.
        a,b,c,d=grid[:,:-1],nxt[:,:-1],nxt[:,1:],grid[:,1:]
        self.assertTrue((cross(b-a,d-a)<0).all())
        self.assertTrue((cross(d-c,b-c)<0).all())
        self.assertTrue((np.linalg.norm(radial,axis=1)>0).all())
        for distances in generate_mouse_surface_distances(self.candidate).values():
            self.assertTrue(np.isfinite(distances).all())
            self.assertTrue((distances>0).all())


class IndividualV1BorderTests(RegisteredBorderTests):
    candidate = WATERS_INDIVIDUAL_V1_BORDER_SVG

    def test_individual_v1_border_is_runtime_default(self):
        self.assertEqual(DEFAULT_BORDER_SVG, WATERS_INDIVIDUAL_V1_BORDER_SVG)


if __name__ == '__main__':
    unittest.main()
