"""Build a read-only metadata index for the currency image dataset."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "dataset_metadata"
OUTPUT_FILE = OUTPUT_DIR / "all_images.csv"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".avif", ".webp"}
CLASS_INFO = {
    "real": (0, "REAL"),
    "fake": (1, "FAKE"),
}
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


def sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path, label: int, class_name: str) -> dict[str, str | int]:
    with Image.open(path) as image:
        image.verify()

    with Image.open(path) as image:
        width, height = image.size
        color_mode = image.mode

    return {
        "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "label": label,
        "class_name": class_name,
        "denomination": path.parent.name,
        "extension": path.suffix.lower(),
        "width": width,
        "height": height,
        "color_mode": color_mode,
        "sha256": sha256_for_file(path),
    }


def main() -> None:
    valid_rows: list[dict[str, str | int]] = []
    unreadable: list[tuple[Path, str]] = []

    for class_directory, (label, class_name) in CLASS_INFO.items():
        class_root = DATA_ROOT / class_directory
        image_paths = sorted(
            path
            for path in class_root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        for path in image_paths:
            try:
                valid_rows.append(inspect_image(path, label, class_name))
            except Exception as error:
                unreadable.append((path, f"{type(error).__name__}: {error}"))

    valid_rows.sort(key=lambda row: str(row["relative_path"]))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(valid_rows)

    class_counts = Counter(str(row["class_name"]) for row in valid_rows)
    denomination_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in valid_rows:
        denomination_counts[str(row["denomination"])][str(row["class_name"])] += 1

    hash_counts = Counter(str(row["sha256"]) for row in valid_rows)
    duplicate_files = sum(count - 1 for count in hash_counts.values() if count > 1)

    print(f"Index created: {OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Total valid images: {len(valid_rows)}")
    print(f"REAL: {class_counts['REAL']}")
    print(f"FAKE: {class_counts['FAKE']}")
    print(f"Unreadable: {len(unreadable)}")
    print(f"Duplicates: {duplicate_files}")
    print("Counts by denomination:")
    for denomination in sorted(denomination_counts, key=lambda value: int(value)):
        counts = denomination_counts[denomination]
        print(
            f"  {denomination}: "
            f"REAL={counts['REAL']}, FAKE={counts['FAKE']}, "
            f"TOTAL={sum(counts.values())}"
        )

    if unreadable:
        print("Unreadable files:")
        for path, error in unreadable:
            print(f"  {path.relative_to(PROJECT_ROOT).as_posix()} [{error}]")


if __name__ == "__main__":
    main()
