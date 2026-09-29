import unittest

import pandas as pd

from medintel.cohort import derive_cvd_history


class OutcomeTests(unittest.TestCase):
    def test_positive_if_any_item_is_yes(self):
        frame = self._frame([[2, 2, 1, 2, 9]])
        self.assertEqual(derive_cvd_history(frame).iloc[0], "positive")

    def test_negative_only_if_every_item_is_valid_no(self):
        frame = self._frame([[2, 2, 2, 2, 2], [2, 2, 2, 9, 2]])
        self.assertEqual(derive_cvd_history(frame).tolist(), ["negative", "unknown"])

    def test_missing_is_not_treated_as_negative(self):
        frame = self._frame([[2, 2, None, 2, 2]])
        self.assertEqual(derive_cvd_history(frame).iloc[0], "unknown")

    @staticmethod
    def _frame(values):
        return pd.DataFrame(values, columns=["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"])


if __name__ == "__main__":
    unittest.main()
