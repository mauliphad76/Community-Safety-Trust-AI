"""Compare REAL/FAKE decision thresholds on the validation manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_ROOT / "models" / "currency_cnn_best.keras"
VALIDATION_CSV = PROJECT_ROOT / "dataset_metadata" / "validation.csv"
RESULTS_FILE = PROJECT_ROOT / "results" / "currency_threshold_analysis.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CURRENT_THRESHOLD = 0.50
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
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
RESULT_COLUMNS = [
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "false_positives",
    "false_negatives",
    "true_negatives",
    "true_positives",
    "confusion_matrix_actual_by_predicted",
]


def read_validation_manifest() -> list[dict[str, str]]:
    with VALIDATION_CSV.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(
                f"Unexpected columns in {VALIDATION_CSV}: {reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("The validation manifest is empty.")
    return rows


def load_image(relative_path: str) -> np.ndarray:
    image_path = PROJECT_ROOT / Path(relative_path)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def load_validation_images(rows: list[dict[str, str]]) -> np.ndarray:
    images = []
    for index, row in enumerate(rows, start=1):
        image_path = PROJECT_ROOT / Path(row["relative_path"])
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Validation row {index} points to missing image: {image_path}"
            )
        try:
            images.append(load_image(row["relative_path"]))
        except Exception as error:
            raise ValueError(
                f"Could not load validation image {image_path}: {error}"
            ) from error
    return np.stack(images)


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
) -> dict[str, float | int | str]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        confusion[int(actual), int(predicted)] += 1

    true_negative, false_positive = confusion[0]
    false_negative, true_positive = confusion[1]
    total = len(labels)
    accuracy = float((true_negative + true_positive) / total)
    precision = float(
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = float(
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_negatives": int(true_negative),
        "true_positives": int(true_positive),
        "confusion_matrix_actual_by_predicted": json.dumps(
            confusion.tolist(), separators=(",", ":")
        ),
    }


def format_percent(value: float | int) -> str:
    return f"{float(value) * 100:.2f}%"


def print_threshold_result(result: dict[str, float | int | str]) -> None:
    print(
        f"Threshold {float(result['threshold']):.2f}: "
        f"Accuracy={format_percent(result['accuracy'])}, "
        f"Precision={format_percent(result['precision'])}, "
        f"Recall={format_percent(result['recall'])}, "
        f"F1={format_percent(result['f1_score'])}, "
        f"FP={result['false_positives']}, FN={result['false_negatives']}"
    )
    matrix = json.loads(str(result["confusion_matrix_actual_by_predicted"]))
    print("  Confusion Matrix (rows=actual REAL,FAKE; columns=predicted REAL,FAKE):")
    print(f"    [[{matrix[0][0]}, {matrix[0][1]}], [{matrix[1][0]}, {matrix[1][1]}]]")


def print_comparison_table(results: list[dict[str, float | int | str]]) -> None:
    headings = ("Threshold", "Accuracy", "Precision", "Recall", "F1", "FP", "FN", "Confusion Matrix")
    print("\nTHRESHOLD COMPARISON (FAKE is the positive class)")
    print(
        f"{headings[0]:>9} {headings[1]:>9} {headings[2]:>9} "
        f"{headings[3]:>9} {headings[4]:>9} {headings[5]:>5} "
        f"{headings[6]:>5}  {headings[7]}"
    )
    for result in results:
        print(
            f"{float(result['threshold']):>9.2f} "
            f"{format_percent(result['accuracy']):>9} "
            f"{format_percent(result['precision']):>9} "
            f"{format_percent(result['recall']):>9} "
            f"{format_percent(result['f1_score']):>9} "
            f"{int(result['false_positives']):>5} "
            f"{int(result['false_negatives']):>5}  "
            f"{result['confusion_matrix_actual_by_predicted']}"
        )


def recommend_threshold(
    results: list[dict[str, float | int | str]],
) -> tuple[dict[str, float | int | str], str]:
    current = next(
        result
        for result in results
        if float(result["threshold"]) == CURRENT_THRESHOLD
    )
    candidates = [
        result
        for result in results
        if int(result["false_negatives"]) < int(current["false_negatives"])
    ]
    if not candidates:
        return (
            current,
            "No tested threshold reduced FAKE-to-REAL errors below the current "
            "0.50 threshold, so the current threshold is retained as the "
            "validation-based recommendation.",
        )

    recommended = max(
        candidates,
        key=lambda result: (
            float(result["f1_score"]),
            float(result["accuracy"]),
            -int(result["false_negatives"]),
            float(result["threshold"]),
        ),
    )
    explanation = (
        "Among tested thresholds with fewer FAKE-to-REAL errors than 0.50, "
        "this threshold has the highest F1 score. F1 balances FAKE precision "
        "and recall, while the lower false-negative count addresses the "
        "application's priority error. This recommendation is based only on "
        "the validation set and does not change app.py."
    )
    return recommended, explanation


def save_results(results: list[dict[str, float | int | str]]) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    rows = read_validation_manifest()
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Validation labels must use 0=REAL and 1=FAKE.")

    images = load_validation_images(rows)
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    fake_probabilities = model.predict(
        images,
        batch_size=BATCH_SIZE,
        verbose=1,
    ).reshape(-1)

    results = []
    for threshold in THRESHOLDS:
        predictions = (fake_probabilities >= threshold).astype(np.int32)
        result = calculate_metrics(labels, predictions, threshold)
        results.append(result)
        print_threshold_result(result)

    save_results(results)
    recommended, reason = recommend_threshold(results)
    print_comparison_table(results)
    print(f"\nValidation image count: {len(rows)}")
    print(f"Current threshold: {CURRENT_THRESHOLD:.2f}")
    print(f"Recommended threshold: {float(recommended['threshold']):.2f}")
    print(f"Recommendation: {reason}")
    print(f"Complete results saved to: {RESULTS_FILE.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()