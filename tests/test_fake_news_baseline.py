import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "fake_news"))

import baseline_common
import evaluate_baseline
import predict_baseline
import train_baseline


class FakeNewsBaselineTests(unittest.TestCase):
    def test_validation_selection_uses_fake_f1(self):
        scores = {
            "logistic_regression": {"f1": 0.81},
            "multinomial_nb": {"f1": 0.92},
            "linear_svm": {"f1": 0.88},
        }
        self.assertEqual(evaluate_baseline.choose_best_model(scores), "multinomial_nb")

    def test_hindi_vectorizer_keeps_devanagari_features(self):
        vectorizer = baseline_common.create_vectorizer("hindi")
        self.assertEqual(vectorizer.analyzer, "char_wb")
        self.assertFalse(vectorizer.lowercase)
        matrix = vectorizer.fit_transform(
            [
                "\u0926\u093f\u0932\u094d\u0932\u0940 \u092e\u0947\u0902 "
                "\u0938\u092e\u093e\u091a\u093e\u0930 \u0906\u092f\u093e",
                "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0916\u092c\u0930 "
                "\u0906\u091c",
                "\u0938\u092e\u093e\u091a\u093e\u0930 \u0914\u0930 "
                "\u0916\u092c\u0930",
            ]
        )
        self.assertGreater(matrix.shape[1], 0)
        feature_names = vectorizer.get_feature_names_out()
        self.assertTrue(
            any(
                any(0x0900 <= ord(character) <= 0x097F for character in feature)
                for feature in feature_names
            )
        )

    def test_metrics_include_per_class_results(self):
        metrics = baseline_common.calculate_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0, 1, 1, 1]),
        )
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertIn("REAL", metrics["class_metrics"])
        self.assertIn("FAKE", metrics["class_metrics"])
        self.assertAlmostEqual(metrics["f1"], 0.8)

    def test_training_reads_only_training_split_and_prediction_uses_artifacts(self):
        texts = [
            "verified council approves community plan alpha",
            "verified council approves public plan beta",
            "official city office releases public notice",
            "trusted agency confirms new policy",
            "fabricated secret cure cures every illness alpha",
            "fabricated viral cure promises impossible wealth beta",
            "fake viral story promises instant wealth",
            "false report invents a dramatic conspiracy",
        ]
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        with tempfile.TemporaryDirectory() as temporary:
            model_root = Path(temporary)
            calls = []

            def training_split(language, split):
                calls.append((language, split))
                if split != "train":
                    raise AssertionError(f"Training tried to read {split}")
                return texts, labels

            with (
                patch.object(baseline_common, "MODELS_DIR", model_root),
                patch.object(train_baseline, "MODELS_DIR", model_root),
                patch.object(train_baseline, "read_split", side_effect=training_split),
            ):
                metadata = train_baseline.train_language("english")
                self.assertEqual(calls, [("english", "train")])
                self.assertEqual(len(metadata["models"]), 3)

                prediction = predict_baseline.predict(
                    "english", "logistic_regression", texts[0]
                )
                self.assertIn(prediction["label"], (0, 1))
                self.assertIn(prediction["prediction"], ("REAL", "FAKE"))


if __name__ == "__main__":
    unittest.main()
