"""Predict REAL/FAKE for one text using a saved language-specific baseline."""

import argparse
import json
from pathlib import Path

import joblib

from baseline_common import (
    LANGUAGES,
    MODEL_LABELS,
    MODEL_NAMES,
    classifier_path,
    vectorizer_path,
)


def predict(language: str, model_name: str, text: str) -> dict:
    if not text.strip():
        raise ValueError("Input text must not be blank.")
    vectorizer_file = vectorizer_path(language)
    model_file = classifier_path(language, model_name)
    if not vectorizer_file.is_file() or not model_file.is_file():
        raise FileNotFoundError(
            f"Trained artifacts not found for {language}/{model_name}; "
            "run train_baseline.py first."
        )
    vectorizer = joblib.load(vectorizer_file)
    classifier = joblib.load(model_file)
    predicted_label = int(classifier.predict(vectorizer.transform([text]))[0])
    return {
        "language": language,
        "model": MODEL_LABELS[model_name],
        "label": predicted_label,
        "prediction": "FAKE" if predicted_label == 1 else "REAL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=LANGUAGES, required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    text_input = parser.add_mutually_exclusive_group(required=True)
    text_input.add_argument("--text", help="News text to classify.")
    text_input.add_argument(
        "--text-file",
        type=Path,
        help="UTF-8 file containing the news text (useful for Hindi input).",
    )
    args = parser.parse_args()
    text = (
        args.text
        if args.text is not None
        else args.text_file.read_text(encoding="utf-8")
    )
    result = predict(args.language, args.model, text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
