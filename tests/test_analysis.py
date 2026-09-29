import math
import unittest

import pandas as pd

from medintel.analysis import survey_ratio


class SurveyEstimatorTests(unittest.TestCase):
    def test_equal_weight_ratio_and_linearized_se(self):
        frame = pd.DataFrame({
            "WTPH2YR": [1, 1, 1, 1],
            "SDMVSTRA": [1, 1, 2, 2],
            "SDMVPSU": [1, 2, 1, 2],
        })
        result = survey_ratio(frame, pd.Series([0, 1, 0, 1]))
        self.assertEqual(result["unweighted_n"], 4)
        self.assertAlmostEqual(result["estimate"], 0.5)
        self.assertAlmostEqual(result["standard_error"], math.sqrt(0.125))
        self.assertEqual(result["design_df"], 2)

    def test_domain_keeps_full_design_structure(self):
        frame = pd.DataFrame({
            "WTPH2YR": [1, 1, 1, 1],
            "SDMVSTRA": [1, 1, 2, 2],
            "SDMVPSU": [1, 2, 1, 2],
        })
        result = survey_ratio(
            frame, pd.Series([1, 0, 0, 1]), pd.Series([True, False, False, True])
        )
        self.assertEqual(result["unweighted_n"], 2)
        self.assertAlmostEqual(result["estimate"], 1.0)


if __name__ == "__main__":
    unittest.main()
