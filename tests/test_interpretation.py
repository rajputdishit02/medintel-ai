import unittest

import numpy as np
import pandas as pd

from medintel.interpretation import select_high_sensitivity_threshold, threshold_metrics


class ThresholdTests(unittest.TestCase):
    def test_selection_meets_training_sensitivity_target(self):
        target = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        probability = np.array([0.05, 0.1, 0.2, 0.4, 0.3, 0.5, 0.7, 0.9])
        result = select_high_sensitivity_threshold(target, probability, 0.75)
        self.assertGreaterEqual(result["training_sensitivity"], 0.75)
        metrics = threshold_metrics(target, probability, result["threshold"])
        self.assertGreaterEqual(metrics["sensitivity"], 0.75)


if __name__ == "__main__":
    unittest.main()
