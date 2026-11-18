"""Language-specific Linear SVM prediction service for fake-news text."""

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "models" / "fake_news" / "baseline"
LANGUAGES = ("english", "hindi")
LABEL_NAMES = {0: "REAL", 1: "FAKE"}
MODEL_NAME = "TF-IDF + Linear SVM"
DECISION_SCORE_EXPLANATION = (
    "Raw signed LinearSVC decision score, not a calibrated probability. "
    "Positive values favor FAKE; negative values favor REAL. "
    "The magnitude is not a probability or a cross-language confidence measure."
)


class FakeNewsPrediction(TypedDict):
    language: str
    model: str
    label: int
    prediction: str
    decision_score: float
    decision_score_explanation: str


def _normalize_language(language: str) -> str:
    if not isinstance(language, str):
        raise TypeError("Language must be a string: 'english' or 'hindi'.")
    normalized = language.strip().casefold()
    if normalized not in LANGUAGES:
        raise ValueError(
            f"Unsupported language {language!r}; choose 'english' or 'hindi'."
        )
    return normalized


@lru_cache(maxsize=2)
def _load_artifacts(language: str) -> tuple[TfidfVectorizer, LinearSVC]:
    model_dir = MODEL_ROOT / language
    vectorizer_path = model_dir / "tfidf_vectorizer.joblib"
    classifier_path = model_dir / "linear_svm.joblib"
    missing = [path for path in (vectorizer_path, classifier_path) if not path.is_file()]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Required {language} fake-news artifact(s) missing: {missing_paths}"
        )

    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError(
            f"Unexpected vectorizer artifact type for {language}: "
            f"{type(vectorizer).__name__}"
        )
    if not isinstance(classifier, LinearSVC):
        raise TypeError(
            f"Unexpected classifier artifact type for {language}: "
            f"{type(classifier).__name__}"
        )
    if classifier.classes_.tolist() != [0, 1]:
        raise ValueError(
            f"{language} Linear SVM classes must be ordered as REAL=0, FAKE=1; "
            f"got {classifier.classes_.tolist()}."
        )
    if classifier.n_features_in_ != len(vectorizer.vocabulary_):
        raise ValueError(
            f"{language} model/vectorizer feature mismatch: "
            f"{classifier.n_features_in_} != {len(vectorizer.vocabulary_)}."
        )
    return vectorizer, classifier


def clear_prediction_cache() -> None:
    """Clear loaded model objects, primarily for tests or artifact replacement."""
    _load_artifacts.cache_clear()


def predict_news(language: str, text: str) -> FakeNewsPrediction:
    """Predict REAL/FAKE for one English or Hindi news article.

    The returned decision score is a raw signed Linear SVM margin, not a
    probability. Positive values favor FAKE and negative values favor REAL.
    """
    normalized_language = _normalize_language(language)
    if not isinstance(text, str):
        raise TypeError("News text must be a string.")
    if not text.strip():
        raise ValueError("News text must not be empty or whitespace-only.")

    vectorizer, classifier = _load_artifacts(normalized_language)
    features = vectorizer.transform([text])
    if features.nnz == 0:
        raise ValueError(
            f"The {normalized_language} vectorizer found no known features in the input."
        )

    decision_score = float(classifier.decision_function(features)[0])
    label = int(classifier.predict(features)[0])
    expected_label = 1 if decision_score > 0 else 0
    if label != expected_label:
        raise RuntimeError(
            f"{normalized_language} Linear SVM prediction disagrees with its decision score."
        )

    return {
        "language": normalized_language,
        "model": MODEL_NAME,
        "label": label,
        "prediction": LABEL_NAMES[label],
        "decision_score": decision_score,
        "decision_score_explanation": DECISION_SCORE_EXPLANATION,
    }
