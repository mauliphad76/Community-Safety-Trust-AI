"""Train and evaluate a Gaussian Naive Bayes currency image baseline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = PROJECT_ROOT / "dataset_metadata" / "train.csv"
TEST_CSV = PROJECT_ROOT / "dataset_metadata" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "currency_naive_bayes.joblib"
IMAGE_SIZE = (32, 32)
EXPECTED_COLUMNS = {"relative_path", "label", "class_name"}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or not EXPECTED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"Manifest is missing required columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def extract_features(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    return pixels.reshape(-1)


def load_split(rows: list[dict[str, str]], split_name: str) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for row_number, row in enumerate(rows, start=1):
        image_path = PROJECT_ROOT / row["relative_path"]
        if not image_path.is_file():
            raise FileNotFoundError(
                f"{split_name} row {row_number} points to a missing image: {image_path}"
            )
        try:
            features.append(extract_features(image_path))
            label = int(row["label"])
        except Exception as error:
            raise ValueError(
                f"Could not load {split_name} row {row_number}: {image_path} ({error})"
            ) from error
        if label not in (0, 1):
            raise ValueError(f"Unexpected label {label} in {split_name} row {row_number}.")
        labels.append(label)
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def train_and_evaluate() -> GaussianNB:
    train_rows = read_manifest(TRAIN_CSV)
    test_rows = read_manifest(TEST_CSV)
    train_features, train_labels = load_split(train_rows, "training")
    test_features, test_labels = load_split(test_rows, "test")

    classifier = GaussianNB(var_smoothing=1e-8)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    print(f"Test accuracy: {accuracy_score(test_labels, predictions):.2%}")
    print(
        classification_report(
            test_labels,
            predictions,
            labels=[0, 1],
            target_names=["REAL", "FAKE"],
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=actual, columns=predicted; REAL, FAKE):")
    print(confusion_matrix(test_labels, predictions, labels=[0, 1]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": classifier, "image_size": IMAGE_SIZE}, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")
    return classifier


def predict_image(image_path: Path) -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Train the model first; not found: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    classifier = artifact["model"]
    features = extract_features(image_path).reshape(1, -1)
    probabilities = classifier.predict_proba(features)[0]
    class_ids = classifier.classes_
    predicted_class = int(class_ids[np.argmax(probabilities)])
    confidence = float(np.max(probabilities))
    fake_index = int(np.flatnonzero(class_ids == 1)[0])

    print(f"Prediction: {'FAKE' if predicted_class == 1 else 'REAL'}")
    print(f"Prediction confidence: {confidence:.1%}")
    print(f"Estimated FAKE probability: {probabilities[fake_index]:.1%}")
    print("This experimental image-based result is not definitive counterfeit verification.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predict",
        type=Path,
        help="Classify an image using the saved model instead of training.",
    )
    arguments = parser.parse_args()

    if arguments.predict:
        predict_image(arguments.predict.resolve())
    else:
        train_and_evaluate()


if __name__ == "__main__":
    main()