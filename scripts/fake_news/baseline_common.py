"""Shared configuration and utilities for the fake-news baseline models."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "data" / "fake_news" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "fake_news" / "baseline"
RESULTS_DIR = PROJECT_ROOT / "results" / "fake_news" / "baseline"
LANGUAGES = ("english", "hindi")
MODEL_NAMES = ("logistic_regression", "multinomial_nb", "linear_svm")
MODEL_LABELS = {
    "logistic_regression": "TF-IDF + Logistic Regression",
    "multinomial_nb": "TF-IDF + Multinomial Naive Bayes",
    "linear_svm": "TF-IDF + Linear SVM",
}
LABEL_NAMES = {0: "REAL", 1: "FAKE"}
RANDOM_STATE = 42


def split_path(language: str, split: str) -> Path:
    if language not in LANGUAGES or split not in ("train", "validation", "test"):
        raise ValueError(f"Unsupported language/split: {language}/{split}")
    return SPLITS_DIR / f"{language}_{split}.csv"


def read_split(language: str, split: str) -> tuple[list[str], np.ndarray]:
    path = split_path(language, split)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    frame = pd.read_csv(path, encoding="utf-8-sig")
    expected_columns = ["text", "label", "language"]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"{path} columns must be {expected_columns}, got {list(frame.columns)}")
    if frame.empty:
        raise ValueError(f"Dataset split is empty: {path}")
    if frame["text"].isna().any() or frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError(f"Dataset split contains missing or blank text: {path}")
    if not frame["language"].eq(language).all():
        raise ValueError(f"Dataset split contains the wrong language: {path}")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError(f"Dataset split contains labels outside REAL=0 / FAKE=1: {path}")

    return frame["text"].astype(str).tolist(), frame["label"].astype("int8").to_numpy()


def create_vectorizer(language: str) -> TfidfVectorizer:
    if language == "english":
        return TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=200_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
    if language == "hindi":
        return TfidfVectorizer(
            analyzer="char_wb",
            lowercase=False,
            strip_accents=None,
            ngram_range=(2, 5),
            min_df=2,
            max_features=150_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
    raise ValueError(f"Unsupported language: {language}")


def create_classifier(model_name: str):
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )
    if model_name == "multinomial_nb":
        return MultinomialNB(alpha=1.0)
    if model_name == "linear_svm":
        return LinearSVC(C=1.0, random_state=RANDOM_STATE)
    raise ValueError(f"Unsupported model: {model_name}")


def vectorizer_path(language: str) -> Path:
    return MODELS_DIR / language / "tfidf_vectorizer.joblib"


def classifier_path(language: str, model_name: str) -> Path:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {model_name}")
    return MODELS_DIR / language / f"{model_name}.joblib"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["REAL", "FAKE"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": report,
        "class_metrics": {
            "REAL": report["REAL"],
            "FAKE": report["FAKE"],
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": ["REAL", "FAKE"],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
