"""Run with `uv run python -m unittest test_metrics` (no model download required)."""

import unittest

import numpy as np

from train import compute_metrics, tune_thresholds


class MetricsTests(unittest.TestCase):
    def assert_scores(self, metrics, prefix, precision, recall, f1):
        for name, expected in (("precision", precision), ("recall", recall), ("f1", f1)):
            with self.subTest(metric=f"{prefix}_{name}"):
                self.assertAlmostEqual(metrics[f"{prefix}_{name}"], expected)

    def test_single_label_reports_positive_class(self):
        # Issue #119: 30 positives, 12 negatives; TP=23, FN=7, FP=9, TN=3.
        target = np.array([1] * 30 + [0] * 12).reshape(-1, 1)
        logits = np.array([2] * 23 + [-2] * 7 + [2] * 9 + [-2] * 3).reshape(-1, 1)

        metrics = compute_metrics(logits, target, ("mylabel",))

        for prefix in ("micro", "macro", "label_mylabel"):
            self.assert_scores(metrics, prefix, 23 / 32, 23 / 30, 46 / 62)
        self.assertEqual(metrics["label_mylabel_support"], 30)
        self.assertAlmostEqual(metrics["exact_match_accuracy"], 26 / 42)
        self.assertAlmostEqual(metrics["hamming_loss"], 16 / 42)
        expected_ap = (23 / 30) * (23 / 32) + (7 / 30) * (30 / 42)
        for prefix in ("micro", "macro", "label_mylabel"):
            self.assertAlmostEqual(metrics[f"{prefix}_average_precision"], expected_ap)

    def test_single_label_degenerate_predictions(self):
        cases = (
            ([0, 0], [0, 0], 0.0),
            ([0, 0], [1, 1], 0.0),
            ([1, 1], [0, 0], 0.0),
            ([1, 1], [1, 1], 1.0),
        )
        for truth, predicted, score in cases:
            with self.subTest(truth=truth, predicted=predicted):
                target = np.array(truth).reshape(-1, 1)
                logits = 2 * np.array(predicted).reshape(-1, 1) - 1
                metrics = compute_metrics(logits, target, ("mylabel",))
                for prefix in ("micro", "macro", "label_mylabel"):
                    self.assert_scores(metrics, prefix, score, score, score)
                self.assertEqual(metrics["label_mylabel_support"], sum(truth))

    def test_single_label_threshold_reports_use_positive_f1(self):
        probabilities = np.array([[0.9], [0.405], [0.6], [0.1]])
        logits = np.log(probabilities / (1 - probabilities))
        target = np.array([[1], [1], [0], [0]])

        reports = tune_thresholds(logits, target, ("mylabel",))

        for name in ("fixed_0_5", "global", "per_label", "selected"):
            with self.subTest(strategy=name):
                report = reports[name]
                tuned = name != "fixed_0_5"
                self.assertAlmostEqual(report["thresholds"][0], 0.4 if tuned else 0.5)
                for prefix in ("micro", "macro", "label_mylabel"):
                    self.assert_scores(
                        report["metrics"],
                        prefix,
                        2 / 3 if tuned else 0.5,
                        1.0 if tuned else 0.5,
                        0.8 if tuned else 0.5,
                    )

    def test_multiple_labels_keep_indicator_metrics(self):
        target = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
        predicted = np.array([[1, 0], [0, 1], [1, 0], [0, 0]])
        logits = 2 * predicted - 1

        for threshold in (0.5, [0.5, 0.5]):
            with self.subTest(threshold=threshold):
                metrics = compute_metrics(logits, target, ("a", "b"), threshold)
                self.assert_scores(metrics, "micro", 2 / 3, 0.5, 4 / 7)
                self.assert_scores(metrics, "macro", 0.75, 0.5, 7 / 12)
                self.assert_scores(metrics, "label_a", 0.5, 0.5, 0.5)
                self.assert_scores(metrics, "label_b", 1.0, 0.5, 2 / 3)
                self.assertEqual(metrics["label_a_support"], 2)
                self.assertEqual(metrics["label_b_support"], 2)
                self.assertEqual(metrics["exact_match_accuracy"], 0.5)
                self.assertEqual(metrics["hamming_loss"], 3 / 8)
                self.assertAlmostEqual(metrics["micro_average_precision"], 7 / 12)
                self.assertEqual(metrics["macro_average_precision"], 0.625)
                self.assertEqual(metrics["label_a_average_precision"], 0.5)
                self.assertEqual(metrics["label_b_average_precision"], 0.75)


if __name__ == "__main__":
    unittest.main()
