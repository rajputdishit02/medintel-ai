import unittest

import pandas as pd

from medintel.modeling import FEATURES, prepare_model_frame


class ModelingDataTests(unittest.TestCase):
    def test_unknown_targets_are_excluded_and_codes_are_cleaned(self):
        rows = []
        for seqn, outcome, smoking in [(1, "positive", 1), (2, "negative", 2), (3, "unknown", 9)]:
            row = {feature: 1.0 for feature in FEATURES if feature != "ever_smoked"}
            row.update({"SEQN": seqn, "cvd_history": outcome, "SMQ020": smoking})
            rows.append(row)
        features, target, identifiers = prepare_model_frame(pd.DataFrame(rows))
        self.assertEqual(identifiers.tolist(), [1, 2])
        self.assertEqual(target.tolist(), [1, 0])
        self.assertEqual(features["ever_smoked"].tolist(), ["yes", "no"])


if __name__ == "__main__":
    unittest.main()
