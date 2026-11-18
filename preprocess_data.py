import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from fake_news_data import (
    PROJECT_ROOT,
    discover_dataset_files,
    iter_dataset_records,
    text_fingerprint,
)


OUTPUT_DIR = PROJECT_ROOT / "data" / "fake_news" / "processed"
REPORT_DIR = PROJECT_ROOT / "results" / "fake_news"


def load_unique_records(files: list[Path]) -> tuple[list[dict[str, object]], dict[str, int]]:
    labels_by_fingerprint: dict[str, int] = {}
    source_rows = 0
    missing_labels = 0
    for source in files:
        for record in iter_dataset_records(source):
            source_rows += 1
            text = str(record["text"])
            if not text:
                continue
            label = record["label"]
            if label is None:
                missing_labels += 1
                continue
            fingerprint = text_fingerprint(text)
            previous_label = labels_by_fingerprint.setdefault(fingerprint, int(label))
            if previous_label != int(label):
                raise ValueError(
                    "Duplicate text has conflicting labels; resolve the source labels "
                    "before training. Conflicts are not silently discarded."
                )

    unique_records: list[dict[str, object]] = []
    seen: set[str] = set()
    for source in files:
        for record in iter_dataset_records(source):
            text = str(record["text"])
            label = record["label"]
            if not text or label is None:
                continue
            fingerprint = text_fingerprint(text)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique_records.append(
                {
                    "text": text,
                    "label": int(label),
                    "language": str(record["language"]),
                    "fingerprint": fingerprint,
                }
            )
    return unique_records, {
        "source_rows": source_rows,
        "unique_records": len(unique_records),
        "missing_labels": missing_labels,
        "duplicate_text_rows_removed": source_rows - len(unique_records) - missing_labels,
    }


def make_splits(
    records: list[dict[str, object]], random_state: int
) -> dict[str, set[int]]:
    indices = list(range(len(records)))
    stratify = [
        f"{record['language']}:{record['label']}"
        for record in records
    ]
    train_indices, remaining_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=random_state,
        stratify=stratify,
    )
    remaining_strata = [stratify[index] for index in remaining_indices]
    validation_indices, test_indices = train_test_split(
        remaining_indices,
        test_size=0.50,
        random_state=random_state,
        stratify=remaining_strata,
    )
    return {
        "train": set(train_indices),
        "validation": set(validation_indices),
        "test": set(test_indices),
    }


def summarize_split(
    records: list[dict[str, object]], indices: set[int]
) -> dict[str, object]:
    labels: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    for index in indices:
        record = records[index]
        labels[str(record["label"])] += 1
        languages[str(record["language"])] += 1
    return {
        "records": len(indices),
        "labels": dict(labels),
        "languages": dict(languages),
        "language_label_counts": {
            f"{language}_{label}": sum(
                1
                for index in indices
                if records[index]["language"] == language
                and records[index]["label"] == label
            )
            for language in ("english", "hindi")
            for label in (0, 1)
        },
    }


def write_splits(
    records: list[dict[str, object]],
    splits: dict[str, set[int]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "combined.csv"
    split_files = {
        name: (output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="")
        for name in splits
    }
    combined_file = combined_path.open("w", encoding="utf-8", newline="")
    try:
        combined_writer = csv.writer(combined_file)
        combined_writer.writerow(["text", "label", "language"])
        writers = {}
        for name, stream in split_files.items():
            writers[name] = csv.writer(stream)
            writers[name].writerow(["text", "label", "language"])
        for index, record in enumerate(records):
            row = [record["text"], record["label"], record["language"]]
            combined_writer.writerow(row)
            for name, indices in splits.items():
                if index in indices:
                    writers[name].writerow(row)
                    break
    finally:
        combined_file.close()
        for stream in split_files.values():
            stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean, deduplicate, and split English/Hindi news datasets."
    )
    parser.add_argument("--search-dir", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    files = discover_dataset_files(args.search_dir)
    if not files:
        raise FileNotFoundError("No CSV or TSV news datasets found.")

    records, source_summary = load_unique_records(files)
    if not records:
        raise ValueError("No labeled, non-empty news records remain after cleaning.")
    splits = make_splits(records, args.random_state)
    assignments = list(splits.values())
    if any(assignments[left] & assignments[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("Split indices overlap; refusing to write leakage-prone splits.")

    split_summary = {
        "random_state": args.random_state,
        "label_encoding": {"0": "FAKE", "1": "REAL"},
        "source_files": [str(path) for path in files],
        "source": source_summary,
        "splits": {
            name: summarize_split(records, indices) for name, indices in splits.items()
        },
        "duplicate_text_leakage": False,
        "method": (
            "Normalize and deduplicate text across all sources before a deterministic "
            "stratified split; stratification uses the language/label combination."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_splits(records, splits, args.output_dir)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.report_dir / "split_summary.json"
    summary_path.write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(split_summary, ensure_ascii=False, indent=2))
    print(f"Processed CSVs: {args.output_dir}")
    print(f"Split summary: {summary_path}")


if __name__ == "__main__":
    main()
