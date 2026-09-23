"""Train a binary REAL-versus-FAKE currency CNN from CSV image manifests."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_ROOT / "dataset_metadata"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
TRAIN_CSV = METADATA_DIR / "train.csv"
VALIDATION_CSV = METADATA_DIR / "validation.csv"
TEST_CSV = METADATA_DIR / "test.csv"
BEST_MODEL = MODEL_DIR / "currency_cnn_best.keras"
HISTORY_CSV = RESULTS_DIR / "currency_training_history.csv"
PLOT_FILE = RESULTS_DIR / "currency_training_plot.png"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_EPOCHS = 20
SEED = 42
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


def set_reproducible_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def validate_manifest_images(rows: list[dict[str, str]], split_name: str) -> None:
    for index, row in enumerate(rows, start=1):
        image_path = PROJECT_ROOT / Path(row["relative_path"])
        if not image_path.is_file():
            raise FileNotFoundError(
                f"{split_name} row {index} points to missing image: {image_path}"
            )
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as error:
            raise ValueError(
                f"{split_name} row {index} cannot be opened: {image_path} ({error})"
            ) from error


def load_image(relative_path: str) -> np.ndarray:
    image_path = PROJECT_ROOT / Path(relative_path)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array


class CurrencySequence(tf.keras.utils.Sequence):
    def __init__(self, rows: list[dict[str, str]], shuffle: bool) -> None:
        self.rows = rows
        self.shuffle = shuffle
        self.indices = np.arange(len(rows))
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / BATCH_SIZE)

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        start = batch_index * BATCH_SIZE
        batch_indices = self.indices[start : start + BATCH_SIZE]
        batch_rows = [self.rows[index] for index in batch_indices]
        images = np.stack([load_image(row["relative_path"]) for row in batch_rows])
        labels = np.asarray([int(row["label"]) for row in batch_rows], dtype=np.float32)
        return images, labels

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)


def calculate_class_weights(rows: list[dict[str, str]]) -> dict[int, float]:
    labels = [int(row["label"]) for row in rows]
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training data must contain both REAL and FAKE classes.")
    total = len(labels)
    return {class_id: total / (2.0 * int(count)) for class_id, count in enumerate(counts)}


def build_model() -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.03, fill_mode="reflect"),
            tf.keras.layers.RandomZoom(height_factor=(-0.05, 0.08), width_factor=(-0.05, 0.08), fill_mode="reflect"),
            tf.keras.layers.RandomTranslation(0.04, 0.04, fill_mode="reflect"),
        ],
        name="training_augmentation",
    )

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = augmentation(inputs)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="fake_probability")(x)

    model = tf.keras.Model(inputs, outputs, name="currency_real_fake_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def save_history(history: tf.keras.callbacks.History) -> None:
    keys = list(history.history)
    with HISTORY_CSV.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(["epoch", *keys])
        for epoch_index in range(len(history.history[keys[0]])):
            writer.writerow([epoch_index + 1, *[history.history[key][epoch_index] for key in keys]])


def save_plot(history: tf.keras.callbacks.History) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history.history["loss"]) + 1)
    axes[0].plot(epochs, history.history["accuracy"], label="Training")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history.history["loss"], label="Training")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(PLOT_FILE, dpi=150)
    plt.close(figure)


def main() -> None:
    set_reproducible_seed()
    train_rows = read_manifest(TRAIN_CSV)
    validation_rows = read_manifest(VALIDATION_CSV)
    test_rows = read_manifest(TEST_CSV)

    print("Verifying all train, validation, and test image paths...")
    validate_manifest_images(train_rows, "train")
    validate_manifest_images(validation_rows, "validation")
    validate_manifest_images(test_rows, "test")
    print("All CSV paths are present and readable. Test data will remain unused during training.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    class_weights = calculate_class_weights(train_rows)
    train_sequence = CurrencySequence(train_rows, shuffle=True)
    validation_sequence = CurrencySequence(validation_rows, shuffle=False)

    model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            BEST_MODEL, monitor="val_loss", save_best_only=True, mode="min", verbose=1
        ),
    ]
    history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    save_history(history)
    save_plot(history)
    best_epoch_index = int(np.argmin(history.history["val_loss"]))
    best_epoch = best_epoch_index + 1
    print(f"Training samples: {len(train_rows)}")
    print(f"Validation samples: {len(validation_rows)}")
    print(f"Test samples: {len(test_rows)}")
    print(f"Class weights: {class_weights}")
    print(f"Best epoch: {best_epoch}")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Best training accuracy: {history.history['accuracy'][best_epoch_index]:.4f}")
    print(f"Best validation accuracy: {history.history['val_accuracy'][best_epoch_index]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Best validation loss: {history.history['val_loss'][best_epoch_index]:.4f}")
    print(f"Best model: {BEST_MODEL.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"History: {HISTORY_CSV.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Plot: {PLOT_FILE.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
