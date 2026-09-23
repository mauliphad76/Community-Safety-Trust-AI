"""Create reproducible, leakage-safe stratified dataset splits."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_ROOT / "dataset_metadata"
INPUT_FILE = METADATA_DIR / "all_images.csv"
SEED = 42
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
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


def target_counts(total: int) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in SPLIT_RATIOS.items()}
    counts = {name: int(value) for name, value in raw.items()}
    for name in sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True):
        if sum(counts.values()) < total:
            counts[name] += 1
    return counts


def read_rows() -> list[dict[str, str]]:
    with INPUT_FILE.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(
                f"Unexpected CSV columns: {reader.fieldnames}; expected {CSV_COLUMNS}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError("The input index is empty.")
    if any(not row["sha256"] for row in rows):
        raise ValueError("Every row must have a SHA-256 hash.")
    if any(row["class_name"] not in {"REAL", "FAKE"} for row in rows):
        raise ValueError("Every row must have class_name REAL or FAKE.")
    return rows


def assign_groups(rows: list[dict[str, str]]) -> dict[str, str]:
    groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["sha256"]].append(row)

    grouped_by_class: defaultdict[str, list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
    for sha256, group_rows in groups.items():
        labels = {row["class_name"] for row in group_rows}
        if len(labels) != 1:
            raise ValueError(
                f"SHA-256 group {sha256} contains multiple class labels: {sorted(labels)}"
            )
        grouped_by_class[next(iter(labels))].append((sha256, group_rows))

    assignment: dict[str, str] = {}
    rng = random.Random(SEED)
    for class_name, class_groups in sorted(grouped_by_class.items()):
        rng.shuffle(class_groups)
        class_total = sum(len(group_rows) for _, group_rows in class_groups)
        targets = target_counts(class_total)
        current = Counter()

        # Place larger groups first so duplicate groups stay intact while counts remain close.
        class_groups.sort(key=lambda item: len(item[1]), reverse=True)
        for sha256, group_rows in class_groups:
            size = len(group_rows)
            scores = {}
            for split_name in SPLIT_RATIOS:
                projected = current.copy()
                projected[split_name] += size
                scores[split_name] = sum(
                    abs(projected[name] - targets[name]) for name in SPLIT_RATIOS
                )
            best_score = min(scores.values())
            choices = [name for name, score in scores.items() if score == best_score]
            split_name = rng.choice(choices)
            assignment[sha256] = split_name
            current[split_name] += size

    return assignment


def validate_splits(
    split_rows: dict[str, list[dict[str, str]]],
    assignment: dict[str, str],
) -> None:
    path_splits: dict[str, str] = {}
    hash_splits: dict[str, str] = {}
    for split_name, rows in split_rows.items():
        if not {row["class_name"] for row in rows} == {"REAL", "FAKE"}:
            raise ValueError(f"{split_name} must contain both REAL and FAKE rows.")
        for row in rows:
            path = row["relative_path"]
            sha256 = row["sha256"]
            if path in path_splits and path_splits[path] != split_name:
                raise ValueError(f"Path appears in multiple splits: {path}")
            if sha256 in hash_splits and hash_splits[sha256] != split_name:
                raise ValueError(f"SHA-256 appears in multiple splits: {sha256}")
            path_splits[path] = split_name
            hash_splits[sha256] = split_name
            if assignment[sha256] != split_name:
                raise ValueError(f"Hash group was split incorrectly: {sha256}")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["relative_path"]))


def main() -> None:
    rows = read_rows()
    assignment = assign_groups(rows)
    split_rows = {
        split_name: [row for row in rows if assignment[row["sha256"]] == split_name]
        for split_name in SPLIT_RATIOS
    }
    validate_splits(split_rows, assignment)

    for split_name, rows_for_split in split_rows.items():
        write_csv(METADATA_DIR / f"{split_name}.csv", rows_for_split)

    hash_counts = Counter(row["sha256"] for row in rows)
    summary = {
        "random_seed": SEED,
        "split_percentages": {name: ratio * 100 for name, ratio in SPLIT_RATIOS.items()},
        "total_images": len(rows),
        "duplicate_group_count": sum(count > 1 for count in hash_counts.values()),
        "duplicate_file_count_beyond_first": sum(
            count - 1 for count in hash_counts.values() if count > 1
        ),
        "splits": {},
    }
    for split_name, rows_for_split in split_rows.items():
        class_counts = Counter(row["class_name"] for row in rows_for_split)
        summary["splits"][split_name] = {
            "total": len(rows_for_split),
            "REAL": class_counts["REAL"],
            "FAKE": class_counts["FAKE"],
        }

    with (METADATA_DIR / "split_summary.json").open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
        json_file.write("\n")

    print(f"Random seed: {SEED}")
    print(f"Total images: {len(rows)}")
    print(f"Duplicate groups: {summary['duplicate_group_count']}")
    print(f"Duplicate files beyond first: {summary['duplicate_file_count_beyond_first']}")
    for split_name in SPLIT_RATIOS:
        counts = summary["splits"][split_name]
        print(
            f"{split_name}: total={counts['total']}, "
            f"REAL={counts['REAL']}, FAKE={counts['FAKE']}"
        )
    print("Verified: paths and SHA-256 groups do not cross splits.")


if __name__ == "__main__":
    main()
