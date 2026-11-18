import math
import unittest

from modules.fake_news_detection import clear_prediction_cache, predict_news


class FakeNewsDetectionTests(unittest.TestCase):
    def setUp(self):
        clear_prediction_cache()

    def tearDown(self):
        clear_prediction_cache()

    def test_linear_svm_predicts_for_both_languages_and_returns_raw_scores(self):
        samples = {
            "english": [
                "The city council approved its annual budget after a public hearing.",
                "A public report describes the latest policy announcement.",
            ],
            "hindi": [
                "नगर परिषद ने सार्वजनिक सुनवाई के बाद वार्षिक बजट को मंजूरी दी।",
                "चमत्कारी इलाज से रातोंरात अमीर बनने की गारंटी, ऐसा दावा किया गया।",
            ],
        }

        for language, texts in samples.items():
            for text in texts:
                with self.subTest(language=language, text=text[:30]):
                    result = predict_news(language, text)
                    self.assertEqual(result["language"], language)
                    self.assertEqual(result["model"], "TF-IDF + Linear SVM")
                    self.assertIn(result["label"], (0, 1))
                    self.assertEqual(
                        result["prediction"], {0: "REAL", 1: "FAKE"}[result["label"]]
                    )
                    self.assertTrue(math.isfinite(result["decision_score"]))
                    self.assertEqual(result["decision_score"] > 0, result["label"] == 1)
                    self.assertIn(
                        "not a calibrated probability",
                        result["decision_score_explanation"],
                    )

    def test_language_is_normalized_and_unsupported_language_is_rejected(self):
        result = predict_news(
            " ENGLISH ",
            "The city council approved its annual budget after a public hearing.",
        )
        self.assertEqual(result["language"], "english")
        with self.assertRaisesRegex(ValueError, "Unsupported language"):
            predict_news("french", "A news article.")

    def test_empty_and_non_string_text_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            predict_news("english", " \t\n")
        with self.assertRaisesRegex(TypeError, "must be a string"):
            predict_news("english", None)

    def test_featureless_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no known features"):
            predict_news("english", "🧪🧪🧪")


if __name__ == "__main__":
    unittest.main()
