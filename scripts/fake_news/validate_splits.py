"""Validate fake-news preprocessing outputs, stratification, leakage, and Unicode."""

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "fake_news" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "fake_news" / "splits"
METADATA_DIR = PROJECT_ROOT / "dataset_metadata" / "fake_news"
EXPECTED_COLUMNS = ["text", "label", "language"]
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def has_devanagari(text: str) -> bool:
    return any(0x0900 <= ord(character) <= 0x097F for character in text)


def inspect_csv(path: Path, expected_language: str) -> dict:
    result = {
        "rows": 0,
        "real": 0,
        "fake": 0,
        "blank": 0,
        "null": 0,
        "bad_language": 0,
        "invalid_label": 0,
        "devanagari": 0,
        "fingerprints": set(),
        "duplicates": 0,
        "columns": [],
    }

    for chunk in pd.read_csv(path, encoding="utf-8-sig", chunksize=5000):
        if not result["columns"]:
            result["columns"] = list(chunk.columns)
        if list(chunk.columns) != EXPECTED_COLUMNS:
            break
        result["rows"] += len(chunk)
        result["null"] += int(chunk.isnull().any(axis=1).sum())
        text = chunk["text"].fillna("").astype(str)
        result["blank"] += int(text.str.strip().eq("").sum())
        result["bad_language"] += int(chunk["language"].ne(expected_language).sum())
        result["real"] += int(chunk["label"].eq(0).sum())
        result["fake"] += int(chunk["label"].eq(1).sum())
        result["invalid_label"] += int((~chunk["label"].isin([0, 1])).sum())
        if expected_language == "hindi":
            result["devanagari"] += int(text.map(has_devanagari).sum())

        for value in text:
            fingerprint = hashlib.sha256(value.encode("utf-8")).digest()
            if fingerprint in result["fingerprints"]:
                result["duplicates"] += 1
            result["fingerprints"].add(fingerprint)
    return result


def validate_source_labels(language: str, clean_path: Path) -> tuple[bool, dict]:
    if language == "english":
        source = pd.read_csv(
            PROJECT_ROOT / "data" / "fake_news" / "raw" / "english"
            / "WELFake_Dataset.csv",
            usecols=["title", "text", "label"],
        )
        title = source["title"].fillna("").astype(str).str.strip()
        body = source["text"].fillna("").astype(str).str.strip()
        text = title.where(title.eq("") | body.eq(""), title + "\n\n" + body)
        text = text.where(title.ne(""), body)
        text = text.where(body.ne(""), title)
        source_label = source["label"]
        expected_map = {0: 1, 1: 0}
        source_rows = pd.DataFrame(
            {"text": text, "source_label": source_label}
        )
        source_rows = source_rows[source_rows["text"].str.strip().ne("")].copy()
        source_rows["label"] = source_rows["source_label"].map(expected_map)
        source_counts = {
            "source_label_0": int(source_label.eq(0).sum()),
            "source_label_1": int(source_label.eq(1).sum()),
            "REAL": int((source_rows["label"] == 0).sum()),
            "FAKE": int((source_rows["label"] == 1).sum()),
        }
        source_label_mapping = {"0": 1, "1": 0}
    else:
        fake_path = (
            PROJECT_ROOT / "data" / "fake_news" / "raw" / "hindi"
            / "hindi_fake_news.csv"
        )
        true_path = (
            PROJECT_ROOT / "data" / "fake_news" / "raw" / "hindi"
            / "hindi_true_news.csv"
        )
        fake = pd.read_csv(fake_path, encoding="utf-8-sig")
        if "Unnamed: 1" in fake.columns:
            fake = fake.drop(columns=["Unnamed: 1"])
        true = pd.read_csv(true_path, encoding="utf-8-sig")
        source_rows = pd.concat(
            [
                pd.DataFrame(
                    {
                        "text": fake["Articles"].fillna("").astype(str).str.strip(),
                        "label": 1,
                    }
                ),
                pd.DataFrame(
                    {
                        "text": true["Articles"].fillna("").astype(str).str.strip(),
                        "label": 0,
                    }
                ),
            ],
            ignore_index=True,
        )
        source_rows = source_rows[source_rows["text"].str.strip().ne("")].copy()
        source_counts = {
            "source_fake_rows": len(fake),
            "source_true_rows": len(true),
            "REAL": int((source_rows["label"] == 0).sum()),
            "FAKE": int((source_rows["label"] == 1).sum()),
        }
        source_label_mapping = {"hindi_fake_news.csv": 1, "hindi_true_news.csv": 0}

    conflicting_groups = int(
        (source_rows.groupby("text")["label"].nunique() > 1).sum()
    )
    expected = source_rows.drop_duplicates(subset=["text"], keep="first")
    actual = pd.read_csv(clean_path, encoding="utf-8-sig")
    expected_pairs = set(zip(expected["text"], expected["label"].astype(int)))
    actual_pairs = set(zip(actual["text"], actual["label"].astype(int)))
    correct = (
        expected_pairs == actual_pairs
        and conflicting_groups == 0
        and actual["language"].eq(language).all()
    )
    return correct, {
        "source_counts": source_counts,
        "source_label_mapping": source_label_mapping,
        "conflicting_groups": conflicting_groups,
        "cleaned_real": int(actual["label"].eq(0).sum()),
        "cleaned_fake": int(actual["label"].eq(1).sum()),
        "source_rows_match_cleaned": expected_pairs == actual_pairs,
    }


def validate() -> bool:
    print("FAKE NEWS PREPROCESSING VALIDATION")
    all_passed = True
    data = {}

    for language in ("english", "hindi"):
        clean_path = PROCESSED_DIR / f"{language}_clean.csv"
        paths = {
            "clean": clean_path,
            **{
                split: SPLITS_DIR / f"{language}_{split}.csv"
                for split in SPLIT_NAMES
            },
        }
        data[language] = {}
        for split, path in paths.items():
            if not path.is_file():
                print(f"FAIL missing output: {path}")
                all_passed = False
                continue
            stats = inspect_csv(path, language)
            data[language][split] = stats
            valid = (
                stats["columns"] == EXPECTED_COLUMNS
                and stats["null"] == 0
                and stats["blank"] == 0
                and stats["bad_language"] == 0
                and stats["invalid_label"] == 0
            )
            print(
                f"{'PASS' if valid else 'FAIL'} {language}_{split}: "
                f"{stats['rows']:,} rows; REAL={stats['real']:,}; "
                f"FAKE={stats['fake']:,}; columns={stats['columns']}"
            )
            all_passed &= valid

    summary_path = METADATA_DIR / "split_summary.json"
    if not summary_path.is_file():
        print(f"FAIL missing metadata: {summary_path}")
        return False
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)

    expected_mapping = {"REAL": 0, "FAKE": 1}
    if summary.get("label_mapping") != expected_mapping:
        print(f"FAIL incorrect metadata label mapping: {summary.get('label_mapping')}")
        all_passed = False
    expected_source_mappings = {
        "english": {"0": 1, "1": 0},
        "hindi": {"hindi_fake_news.csv": 1, "hindi_true_news.csv": 0},
    }

    for language in ("english", "hindi"):
        language_data = data[language]
        if not all(name in language_data for name in ("clean", *SPLIT_NAMES)):
            all_passed = False
            continue
        clean = language_data["clean"]
        split_stats = [language_data[name] for name in SPLIT_NAMES]
        dataset_meta = summary.get("datasets", {}).get(language, {})

        source_ok, source_report = validate_source_labels(
            language, PROCESSED_DIR / f"{language}_clean.csv"
        )
        mapping_ok = (
            dataset_meta.get("source_label_mapping")
            == expected_source_mappings[language]
        )
        print(
            f"{'PASS' if source_ok and mapping_ok else 'FAIL'} {language} source-label "
            f"mapping: {source_report['source_label_mapping']}; "
            f"cleaned REAL={source_report['cleaned_real']:,}, "
            f"FAKE={source_report['cleaned_fake']:,}; "
            f"source rows reconcile={source_report['source_rows_match_cleaned']}; "
            f"conflicting groups={source_report['conflicting_groups']}"
        )
        all_passed &= source_ok and mapping_ok

        if clean["duplicates"] != 0:
            print(f"FAIL {language}: duplicate articles remain in cleaned data")
            all_passed = False
        if sum(item["rows"] for item in split_stats) != clean["rows"]:
            print(f"FAIL {language}: split row counts do not equal cleaned row count")
            all_passed = False

        split_union = set().union(*(item["fingerprints"] for item in split_stats))
        if split_union != clean["fingerprints"]:
            print(f"FAIL {language}: split contents do not match cleaned dataset")
            all_passed = False
        overlaps = {
            "train-validation": len(
                language_data["train"]["fingerprints"]
                & language_data["validation"]["fingerprints"]
            ),
            "train-test": len(
                language_data["train"]["fingerprints"]
                & language_data["test"]["fingerprints"]
            ),
            "validation-test": len(
                language_data["validation"]["fingerprints"]
                & language_data["test"]["fingerprints"]
            ),
        }
        leakage_ok = not any(overlaps.values())
        print(f"{'PASS' if leakage_ok else 'FAIL'} {language} duplicate leakage: {overlaps}")
        all_passed &= leakage_ok

        total_rows = clean["rows"]
        for split_name, stats in zip(SPLIT_NAMES, split_stats):
            expected_rows = total_rows * SPLIT_RATIOS[split_name]
            size_ok = abs(stats["rows"] - expected_rows) <= 1
            if stats["rows"]:
                clean_real_pct = clean["real"] / total_rows
                split_real_pct = stats["real"] / stats["rows"]
                distribution_ok = abs(clean_real_pct - split_real_pct) <= 0.002
            else:
                distribution_ok = False
            print(
                f"{'PASS' if size_ok and distribution_ok else 'FAIL'} "
                f"{language} {split_name}: {stats['rows']:,}/{total_rows:,} rows "
                f"({stats['rows'] / total_rows:.2%}); REAL={stats['real']:,} "
                f"({stats['real'] / stats['rows']:.2%}); FAKE={stats['fake']:,} "
                f"({stats['fake'] / stats['rows']:.2%})"
            )
            all_passed &= size_ok and distribution_ok

        counts_consistent = (
            dataset_meta.get("cleaned_count") == clean["rows"]
            and dataset_meta.get("real_count") == clean["real"]
            and dataset_meta.get("fake_count") == clean["fake"]
            and dataset_meta.get("original_count")
            == dataset_meta.get("cleaned_count", 0)
            + dataset_meta.get("blank_removed", 0)
            + dataset_meta.get("duplicates_removed", 0)
        )
        print(f"{'PASS' if counts_consistent else 'FAIL'} {language} metadata counts")
        all_passed &= counts_consistent

        for split_name, stats in zip(SPLIT_NAMES, split_stats):
            recorded = dataset_meta.get("splits", {}).get(split_name, {})
            recorded_ok = (
                recorded.get("total") == stats["rows"]
                and recorded.get("real") == stats["real"]
                and recorded.get("fake") == stats["fake"]
            )
            all_passed &= recorded_ok
            if not recorded_ok:
                print(f"FAIL {language} {split_name} counts differ from metadata")

    if "hindi" in data and "clean" in data["hindi"]:
        hindi_clean = data["hindi"]["clean"]
        unicode_ok = (
            hindi_clean["devanagari"] > 0
            and hindi_clean["devanagari"]
            == summary.get("datasets", {}).get("hindi", {}).get(
                "unicode_devanagari_count"
            )
        )
        print(
            f"{'PASS' if unicode_ok else 'FAIL'} Hindi Devanagari preservation: "
            f"{hindi_clean['devanagari']:,}/{hindi_clean['rows']:,} rows"
        )
        all_passed &= unicode_ok

    print("ALL VALIDATION CHECKS PASSED" if all_passed else "VALIDATION FAILED")
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if validate() else 1)
