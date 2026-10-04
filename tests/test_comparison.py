import unittest

import numpy as np
import pandas as pd

from medintel.comparison import bootstrap_intervals


class ComparisonTests(unittest.TestCase):
    def test_bootstrap_is_reproducible_and_reports_paired_deltas(self):
        target = pd.Series([0, 0, 0, 1, 1, 1])
        logistic = np.array([0.1, 0.2, 0.4, 0.5, 0.7, 0.8])
        boosted = np.array([0.05, 0.15, 0.3, 0.6, 0.8, 0.9])
        first = bootstrap_intervals(target, logistic, boosted, repetitions=30)
        second = bootstrap_intervals(target, logistic, boosted, repetitions=30)
        self.assertEqual(first, second)
        self.assertIn("delta_boosted_minus_logistic", first["roc_auc"])
        self.assertEqual(len(first["average_precision"]["boosted"]), 2)


if __name__ == "__main__":
    unittest.main()
