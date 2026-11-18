"""Train TF-IDF + classical ML baselines using training data only."""

import argparse
import json
import time

import joblib

from baseline_common import (
    LANGUAGES,
    MODEL_LABELS,
    MODEL_NAMES,
    MODELS_DIR,
    create_classifier,
    create_vectorizer,
    read_split,
    vectorizer_path,
    classifier_path,
)


def train_language(language: str) -> dict:
    train_texts, train_labels = read_split(language, "train")
    output_dir = MODELS_DIR / language
    output_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = create_vectorizer(language)
    vectorizer_start = time.perf_counter()
    train_matrix = vectorizer.fit_transform(train_texts)
    vectorizer_fit_seconds = time.perf_counter() - vectorizer_start

    saved_vectorizer = vectorizer_path(language)
    joblib.dump(vectorizer, saved_vectorizer)
    feature_count = int(train_matrix.shape[1])
    model_reports = {}

    for model_name in MODEL_NAMES:
        classifier = create_classifier(model_name)
        start = time.perf_counter()
        classifier.fit(train_matrix, train_labels)
        fit_seconds = time.perf_counter() - start

        model_path = classifier_path(language, model_name)
        joblib.dump(classifier, model_path)
        model_reports[model_name] = {
            "display_name": MODEL_LABELS[model_name],
            "vectorizer_fit_seconds": vectorizer_fit_seconds,
            "classifier_fit_seconds": fit_seconds,
            "total_training_seconds": vectorizer_fit_seconds + fit_seconds,
            "training_rows": len(train_texts),
            "features": feature_count,
            "vectorizer_file": saved_vectorizer.name,
            "vectorizer_size_bytes": saved_vectorizer.stat().st_size,
            "model_file": model_path.name,
            "model_size_bytes": model_path.stat().st_size,
        }
        print(
            f"{language} / {MODEL_LABELS[model_name]}: "
            f"vectorizer fit {vectorizer_fit_seconds:.2f}s, "
            f"classifier fit {fit_seconds:.2f}s, {feature_count:,} features"
        )

    metadata = {
        "language": language,
        "label_mapping": {"REAL": 0, "FAKE": 1},
        "training_split": "train",
        "validation_or_test_used_for_training": False,
        "vectorizer_fit_on": "training data only",
        "vectorizer": {
            "analyzer": "word" if language == "english" else "char_wb",
            "ngram_range": [1, 2] if language == "english" else [2, 5],
            "lowercase": language == "english",
            "english_stop_words": language == "english",
            "hindi_devanagari_preserved": language == "hindi",
            "feature_count": feature_count,
            "fit_seconds": vectorizer_fit_seconds,
            "file": saved_vectorizer.name,
            "size_bytes": saved_vectorizer.stat().st_size,
        },
        "models": model_reports,
    }
    metadata_path = output_dir / "training_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as output:
        json.dump(metadata, output, ensure_ascii=False, indent=2)
    print(f"Saved {language} baseline artifacts to {output_dir}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("all", *LANGUAGES), default="all")
    args = parser.parse_args()
    languages = LANGUAGES if args.language == "all" else (args.language,)
    for language in languages:
        train_language(language)


if __name__ == "__main__":
    main()
