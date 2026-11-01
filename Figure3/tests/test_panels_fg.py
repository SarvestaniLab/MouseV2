import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from panels_fg import AREA_ORDER, analyze_metric, load_cells


DATA = ROOT / "data" / "han2022_figure3fg_legacy_registration.csv"
EXPECTED_SHA256 = "e9704ac3c81f977bb02370ede617e59c456afdc0f4f8f0316a2a6e1eb5eb7247"


class Figure3PublicRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells = load_cells(DATA)

    def test_input_table_hash_and_row_count(self):
        self.assertEqual(hashlib.sha256(DATA.read_bytes()).hexdigest(), EXPECTED_SHA256)
        self.assertEqual(len(self.cells), 26046)

    def test_only_six_manuscript_hvas_enter_analysis(self):
        result = analyze_metric(self.cells, "spatial_frequency_octaves", fit_method="legacy")
        self.assertEqual(set(result.all_cells["area"]), set(AREA_ORDER))
        self.assertNotIn("V1", set(result.all_cells["area"]))
        self.assertNotIn("P", set(result.all_cells["area"]))

    def test_historical_plane_coefficients(self):
        spatial = analyze_metric(self.cells, "spatial_frequency_octaves", fit_method="legacy")
        temporal = analyze_metric(self.cells, "temporal_frequency_octaves", fit_method="legacy")
        np.testing.assert_allclose(
            (spatial.fit.eccentricity_coefficient, spatial.fit.elevation_coefficient,
             spatial.fit.intercept, spatial.fit.sigma),
            (0.011080628683448756, 0.01677061277459344,
             1.4225905737886158, 0.8775593568872725),
            rtol=0.0, atol=1e-12,
        )
        np.testing.assert_allclose(
            (temporal.fit.eccentricity_coefficient, temporal.fit.elevation_coefficient,
             temporal.fit.intercept, temporal.fit.sigma),
            (-0.016026605453224364, -0.029264038563212247,
             2.831226910427385, 0.856183708104914),
            rtol=0.0, atol=1e-12,
        )

    def test_historical_within_bound_counts(self):
        spatial = analyze_metric(self.cells, "spatial_frequency_octaves", fit_method="legacy")
        temporal = analyze_metric(self.cells, "temporal_frequency_octaves", fit_method="legacy")
        self.assertEqual(len(spatial.uncensored_cells), 12105)
        self.assertEqual(len(temporal.uncensored_cells), 10291)

    def test_maintained_tobit_coefficients_with_portable_tolerance(self):
        spatial = analyze_metric(self.cells, "spatial_frequency_octaves")
        temporal = analyze_metric(self.cells, "temporal_frequency_octaves")
        np.testing.assert_allclose(
            (spatial.fit.eccentricity_coefficient, spatial.fit.elevation_coefficient,
             spatial.fit.intercept, spatial.fit.sigma),
            (0.01574860379374427, 0.022509919879725775,
             1.0931017431502608, 1.521737496417735),
            rtol=0.0, atol=2e-6,
        )
        np.testing.assert_allclose(
            (temporal.fit.eccentricity_coefficient, temporal.fit.elevation_coefficient,
             temporal.fit.intercept, temporal.fit.sigma),
            (-0.025500367825849957, -0.04472769677459344,
             3.2532481897389527, 1.99253527884545),
            rtol=0.0, atol=2e-6,
        )


if __name__ == "__main__":
    unittest.main()
