"""Regression against the archived seed-13 manuscript result."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'elastictopy'))
from fig2.manuscript_partition import normalize_visual, parcellate
from scripts.make_figure2_updated import load_run
from scripts.make_figure2_manuscript import verify_pair
from elastictopy.mouse_surface import generate_mouse_surface_grid


class ManuscriptFigure2Tests(unittest.TestCase):
    def test_selected_runs_and_archived_partition(self):
        freeze = json.loads((ROOT / 'data/publication/figure2/freeze.json').read_text())
        primary = ROOT / freeze['runs']['primary']['directory']
        high = ROOT / freeze['runs']['high_smoothness']['directory']
        manifests = verify_pair(high, primary)
        for manifest, beta1 in zip(manifests, (.002, .00035)):
            self.assertEqual((manifest['seed'], manifest['b1'], manifest['b2'], manifest['b3']),
                             (13, beta1, 20, .001))
            self.assertEqual(manifest['parameters']['n_iterations'], 2000)
            self.assertEqual(manifest['parameters']['magnification'], -.05)
        data = load_run(primary)
        self.assertEqual(data['prototypes'].shape, (2000, 2))
        data['full'] = normalize_visual(data['full'], np.isfinite(data['full'][:, 1:]).all(axis=2))
        surface = generate_mouse_surface_grid(
            svg_path=ROOT / freeze['rendering_surface'], map_h=80, map_w=8, cyclic_along=True).copy()
        surface[:, :, 1] *= -1
        result = parcellate(data, surface, forced_row_cut=None,
                           assignment_sign_threshold=0., merge_overlap_threshold=0.,
                           redundancy_threshold=1.01, post_cut_split=True)
        with np.load(ROOT / 'data/publication/figure2/partition-reference.npz') as reference:
            np.testing.assert_array_equal(result['labels'], reference['labels'])
            np.testing.assert_allclose(result['field_sign'], reference['field_sign'], rtol=0, atol=1e-12)
        self.assertEqual(len(result['centers']), 8)


if __name__ == '__main__':
    unittest.main()
