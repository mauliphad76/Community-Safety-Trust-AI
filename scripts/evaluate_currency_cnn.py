"""Evaluate the saved currency CNN on the unseen test manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_ROOT / "models" / "currency_cnn_best.keras"
TEST_CSV = PROJECT_ROOT / "dataset_metadata" / "test.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "currency_test_results.json"
CONFUSION_PLOT = RESULTS_DIR / "currency_confusion_matrix.png"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CSV_COLUMNS = [
    "relative_path",
    "label",
    "class_name",
    "denomination",
    "extension",
    "width",
    "height",
    "color_mode",
    "sha256",
]


def read_test_manifest() -> list[dict[str, str]]:
    with TEST_CSV.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"Unexpected columns in {TEST_CSV}: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError("The test manifest is empty.")
    return rows


def load_image(relative_path: str) -> np.ndarray:
    image_path = PROJECT_ROOT / Path(relative_path)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def load_test_images(rows: list[dict[str, str]]) -> np.ndarray:
    images = []
    for index, row in enumerate(rows, start=1):
        image_path = PROJECT_ROOT / Path(row["relative_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Test row {index} points to missing image: {image_path}")
        try:
            images.append(load_image(row["relative_path"]))
        except Exception as error:
            raise ValueError(f"Could not load test image {image_path}: {error}") from error
    return np.stack(images)


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        confusion[int(actual), int(predicted)] += 1

    true_negative, false_positive = confusion[0]
    false_negative, true_positive = confusion[1]
    total = len(labels)
    correct = int(true_negative + true_positive)
    incorrect = int(false_positive + false_negative)
    accuracy = correct / total if total else 0.0
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "test_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "correct_predictions": correct,
        "incorrect_predictions": incorrect,
    }
    return metrics, confusion


def save_confusion_plot(confusion: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title="Currency CNN Confusion Matrix",
    )
    threshold = confusion.max() / 2 if confusion.size else 0
    for row_index in range(2):
        for column_index in range(2):
            color = "white" if confusion[row_index, column_index] > threshold else "black"
            axis.text(
                column_index,
                row_index,
                str(confusion[row_index, column_index]),
                ha="center",
                va="center",
                color=color,
                fontsize=14,
                fontweight="bold",
            )
    figure.tight_layout()
    figure.savefig(CONFUSION_PLOT, dpi=150)
    plt.close(figure)


def main() -> None:
    rows = read_test_manifest()
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    images = load_test_images(rows)

    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    probabilities = model.predict(images, batch_size=BATCH_SIZE, verbose=1).reshape(-1)
    predictions = (probabilities >= 0.5).astype(np.int32)
    metrics, confusion = calculate_metrics(labels, predictions)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": MODEL_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "test_manifest": TEST_CSV.relative_to(PROJECT_ROOT).as_posix(),
        "preprocessing": {
            "color": "RGB",
            "resize": "224x224",
            "normalization": "pixel values divided by 255.0 to [0, 1]",
        },
        "test_samples": len(rows),
        **metrics,
        "confusion_matrix": {
            "labels": ["REAL", "FAKE"],
            "rows_actual": [
                [int(confusion[0, 0]), int(confusion[0, 1])],
                [int(confusion[1, 0]), int(confusion[1, 1])],
            ],
        },
    }
    with RESULTS_FILE.open("w", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2)
        result_file.write("\n")
    save_confusion_plot(confusion)

    print("FINAL CURRENCY CNN TEST REPORT")
    print(f"Test samples: {len(rows)}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print("\n                 Predicted")
    print("              REAL     FAKE")
    print(f"Actual REAL   {confusion[0, 0]:4d}     {confusion[0, 1]:4d}")
    print(f"Actual FAKE   {confusion[1, 0]:4d}     {confusion[1, 1]:4d}")
    print(f"Correct predictions: {metrics['correct_predictions']}")
    print(f"Incorrect predictions: {metrics['incorrect_predictions']}")
    print(f"Results: {RESULTS_FILE.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Confusion plot: {CONFUSION_PLOT.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
