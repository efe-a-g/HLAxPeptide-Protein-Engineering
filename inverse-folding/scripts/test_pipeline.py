"""Independent behavioral checks for leakage, ties, coverage and metric aggregation.

Run from the repository root: .venv/bin/python -m unittest discover -s scripts -p test_pipeline.py
"""
import tempfile
import unittest

import numpy as np
import pandas as pd

from evaluation import EPSILON, evaluate_predictions, metric_values
from splits import balanced_subset, load_data, make_splits


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def test_actual_data_split_isolation_and_complete_loao_coverage(self):
        by_id = self.data.set_index("row_id")
        universe = set(by_id.index)
        held_out = []
        for split, fold, train, test in make_splits(self.data):
            self.assertFalse(set(train) & set(test), (split, fold))
            self.assertEqual(set(train) | set(test), universe)
            tr, te = by_id.loc[train], by_id.loc[test]
            if split == "peptide_disjoint":
                self.assertFalse(set(tr.peptide) & set(te.peptide))
            if split == "leave_one_allele_out":
                self.assertEqual(set(te.allele), {fold})
                self.assertFalse(set(tr.allele) & set(te.allele))
                held_out.extend(test)
        self.assertEqual(len(held_out), len(universe))
        self.assertEqual(set(held_out), universe)

    def test_balanced_pilot_and_splits_are_reproducible_without_target(self):
        first = balanced_subset(self.data)
        changed = self.data.assign(thalf_hours=self.data.thalf_hours.iloc[::-1].to_numpy())
        second = balanced_subset(changed)
        self.assertEqual(first.row_id.tolist(), second.row_id.tolist())
        self.assertEqual(len(first), 1000)
        self.assertEqual(first.allele.nunique(), self.data.allele.nunique())
        for a, b in zip(make_splits(first), make_splits(second)):
            self.assertEqual(a[:2], b[:2])
            np.testing.assert_array_equal(a[2], b[2])
            np.testing.assert_array_equal(a[3], b[3])


class MetricTests(unittest.TestCase):
    def test_tied_spearman_uses_average_ranks_and_log_pearson(self):
        frame = pd.DataFrame({"thalf_hours": [0, 0, 1, 3, 3], "prediction": [0, 2, 1, 3, 4]})
        result, n = metric_values(frame)
        expected_ranks = [1.5, 1.5, 3, 4.5, 4.5]
        expected = np.corrcoef(expected_ranks, [1, 3, 2, 4, 5])[0, 1]
        self.assertAlmostEqual(result["spearman"], expected)
        self.assertEqual(n, 5)
        exact_log = frame.assign(prediction=np.log(frame.thalf_hours + EPSILON))
        result, _ = metric_values(exact_log)
        self.assertAlmostEqual(result["pearson_log"], 1)
        self.assertAlmostEqual(result["spearman"], 1)

    def test_constant_predictions_operational_zero_and_missing_coverage(self):
        frame = pd.DataFrame({"thalf_hours": [0, 1, 3, 4], "prediction": [2., 2., 2., np.nan]})
        result, n = metric_values(frame)
        self.assertEqual(n, 3)
        self.assertEqual(result["spearman"], 0.)
        self.assertEqual(result["pearson_log"], 0.)
        self.assertEqual(result["auc_gt_2h"], .5)
        result, _ = metric_values(frame.assign(thalf_hours=1))
        self.assertTrue(np.isnan(result["spearman"]))

    def test_macro_weighted_nonzero_and_unavailable_cells(self):
        # Four perfectly ordered points in A, six perfectly reversed points in B.
        frame = pd.DataFrame({"row_id": range(10), "allele": ["A"]*4+["B"]*6,
                              "thalf_hours": [0, 1, 3, 4, 0, 1, 2, 3, 4, 5],
                              "prediction": [0, 1, 3, 4, 5, 4, 3, 2, 1, 0],
                              "representation": "synthetic", "split": "test", "loss": "mse", "fold": "holdout"})
        with tempfile.TemporaryDirectory() as directory:
            results, per = evaluate_predictions(frame, directory)
        def value(subset, aggregation):
            return results[(results.subset == subset) & (results.aggregation == aggregation) & (results.metric == "spearman")].iloc[0]
        self.assertAlmostEqual(value("all", "macro").value, 0)
        self.assertAlmostEqual(value("all", "size_weighted").value, -.2)
        self.assertAlmostEqual(value("nonzero", "macro").value, 0)
        self.assertAlmostEqual(value("nonzero", "size_weighted").value, -.25)
        self.assertEqual(value("all", "pooled").n_total, 10)
        self.assertEqual(value("nonzero", "pooled").n_total, 8)
        with tempfile.TemporaryDirectory() as directory:
            unavailable, _ = evaluate_predictions(frame.assign(prediction=np.nan), directory)
        self.assertTrue(unavailable.value.isna().all())
        self.assertTrue((unavailable.status == "unavailable").all())
        self.assertTrue((unavailable.n == 0).all())


if __name__ == "__main__":
    unittest.main()
