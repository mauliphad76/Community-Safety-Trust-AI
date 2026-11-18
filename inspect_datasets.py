import argparse
import json
from collections import Counter
from pathlib import Path

from fake_news_data import (
    PROJECT_ROOT,
    detect_columns,
    discover_dataset_files,
    iter_dataset_records,
    text_fingerprint,
)


REPORT_DIR = PROJECT_ROOT / "results" / "fake_news"


def inspect_file(source: Path) -> dict[str, object]:
    import csv

    delimiter = "\t" if source.suffix.casefold() == ".tsv" else ","
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter, restkey="_extra_fields")
        headers = reader.fieldnames or []
        text_columns, label_column = detect_columns(headers, source)
        missing = Counter()
        row_count = 0
        extra_field_rows = 0
        for row in reader:
            row_count += 1
            for column in headers:
                if not str(row.get(column) or "").strip():
                    missing[column] += 1
            if row.get("_extra_fields"):
                extra_field_rows += 1

    label_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    empty_texts = 0
    fingerprints: set[str] = set()
    duplicate_texts = 0
    for record in iter_dataset_records(source):
        text = str(record["text"])
        if not text:
            empty_texts += 1
            continue
        fingerprint = text_fingerprint(text)
        if fingerprint in fingerprints:
            duplicate_texts += 1
        fingerprints.add(fingerprint)
        label = record["label"]
        label_counts["missing" if label is None else str(label)] += 1
        language_counts[str(record["language"])] += 1

    return {
        "file": str(source),
        "size_bytes": source.stat().st_size,
        "columns": headers,
        "detected_text_columns": text_columns,
        "detected_label_column": label_column,
        "rows": row_count,
        "missing_values_by_column": dict(missing),
        "empty_text_rows": empty_texts,
        "duplicate_text_rows": duplicate_texts,
        "duplicate_text_groups": len(fingerprints),
        "extra_field_rows": extra_field_rows,
        "labels": dict(label_counts),
        "languages": dict(language_counts),
    }


def build_report(files: list[Path]) -> dict[str, object]:
    if not files:
        raise FileNotFoundError(
            "No CSV or TSV datasets found. Expected files in Downloads/English, "
            "Downloads/hindi, datasets/, or data/fake_news/."
        )
    per_file = [inspect_file(source) for source in files]
    total_labels: Counter[str] = Counter()
    total_languages: Counter[str] = Counter()
    globally_seen: set[str] = set()
    global_duplicates = 0
    total_records = 0
    empty_rows = 0
    label_conflicts: dict[str, set[int]] = {}
    for source in files:
        for record in iter_dataset_records(source):
            total_records += 1
            text = str(record["text"])
            if not text:
                empty_rows += 1
                continue
            fingerprint = text_fingerprint(text)
            if fingerprint in globally_seen:
                global_duplicates += 1
            globally_seen.add(fingerprint)
            label = record["label"]
            if label is None:
                total_labels["missing"] += 1
            else:
                total_labels[str(label)] += 1
                label_conflicts.setdefault(fingerprint, set()).add(int(label))
            total_languages[str(record["language"])] += 1
    conflicts = sum(len(labels) > 1 for labels in label_conflicts.values())

    return {
        "label_encoding": {"0": "REAL", "1": "FAKE"},
        "label_mapping_evidence": (
            "WELFake source labels map as 0=FAKE and 1=REAL; project labels are "
            "REAL=0 and FAKE=1. Hindi labels are assigned from the explicit "
            "hindi_fake_news (1) and hindi_true_news (0) filenames."
        ),
        "files": per_file,
        "total_source_rows": total_records,
        "empty_text_rows": empty_rows,
        "duplicate_text_rows_across_files_including_within_files": global_duplicates,
        "unique_nonempty_texts": len(globally_seen),
        "label_counts": dict(total_labels),
        "language_counts": dict(total_languages),
        "conflicting_duplicate_label_groups": conflicts,
    }


def render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Fake News Dataset Inspection",
        "",
        f"- Source rows: {report['total_source_rows']:,}",
        f"- Unique non-empty texts: {report['unique_nonempty_texts']:,}",
        f"- Empty text rows: {report['empty_text_rows']:,}",
        f"- Duplicate text rows (including within files): "
        f"{report['duplicate_text_rows_across_files_including_within_files']:,}",
        f"- Conflicting duplicate-label groups: {report['conflicting_duplicate_label_groups']}",
        f"- Labels: {report['label_counts']}",
        f"- Languages: {report['language_counts']}",
        "",
        "Label encoding: `0 = REAL`, `1 = FAKE`.",
        "",
        str(report["label_mapping_evidence"]),
        "",
        "| File | Rows | Columns | Text columns | Label column | Missing values | Duplicates |",
        "|---|---:|---|---|---|---|---:|",
    ]
    for item in report["files"]:
        missing = item["missing_values_by_column"] or "none"
        lines.append(
            f"| `{Path(item['file']).name}` | {item['rows']:,} | "
            f"`{', '.join(item['columns'])}` | "
            f"`{', '.join(item['detected_text_columns'])}` | "
            f"`{item['detected_label_column'] or 'filename'}` | "
            f"`{missing}` | {item['duplicate_text_rows']:,} |"
        )
    lines.extend(
        [
            "",
            "Duplicate text is normalized for Unicode compatibility, whitespace, and case "
            "before fingerprinting. No article text is copied into this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect English and Hindi news datasets.")
    parser.add_argument(
        "--search-dir",
        action="append",
        type=Path,
        help="Additional directory to search recursively for CSV/TSV files.",
    )
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    files = discover_dataset_files(args.search_dir)
    report = build_report(files)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / "dataset_inspection.json"
    markdown_path = args.report_dir / "dataset_inspection.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
