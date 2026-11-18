"""
Preprocessing and leakage-safe dataset splitting pipeline for Fake News.
Handles English (WELFake) and Hindi datasets.
Standard labels: REAL = 0, FAKE = 1.
"""

from pathlib import Path
import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_REAL = 0
LABEL_FAKE = 1
ENGLISH_LABEL_MAP = {0: LABEL_FAKE, 1: LABEL_REAL}

# Input Paths
RAW_ENG_PATH = PROJECT_ROOT / "data" / "fake_news" / "raw" / "english" / "WELFake_Dataset.csv"
RAW_HIN_FAKE_PATH = PROJECT_ROOT / "data" / "fake_news" / "raw" / "hindi" / "hindi_fake_news.csv"
RAW_HIN_TRUE_PATH = PROJECT_ROOT / "data" / "fake_news" / "raw" / "hindi" / "hindi_true_news.csv"

# Output Paths
PROCESSED_DIR = PROJECT_ROOT / "data" / "fake_news" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "fake_news" / "splits"
METADATA_DIR = PROJECT_ROOT / "dataset_metadata" / "fake_news"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


def is_devanagari(text: str) -> bool:
    """Check whether text contains Devanagari Unicode characters (0x0900 - 0x097F)."""
    return any(0x0900 <= ord(c) <= 0x097F for c in str(text))


def preprocess_english() -> dict:
    print("\n--- Processing English (WELFake) Dataset ---")
    df_raw = pd.read_csv(RAW_ENG_PATH)
    original_count = len(df_raw)
    print(f"Original row count: {original_count}")

    # Verify original label values
    raw_labels = df_raw["label"].value_counts(dropna=False).to_dict()
    print(f"Raw label distribution: {raw_labels}")
    unknown_labels = set(df_raw["label"].dropna().unique()) - set(ENGLISH_LABEL_MAP)
    observed_labels = set(df_raw["label"].dropna().unique())
    if unknown_labels or observed_labels != set(ENGLISH_LABEL_MAP) or df_raw["label"].isna().any():
        raise ValueError(f"Unexpected, incomplete, or missing WELFake labels: {raw_labels}")

    # Combine title and text
    def combine_title_text(row):
        title = str(row["title"]).strip() if pd.notna(row["title"]) else ""
        text = str(row["text"]).strip() if pd.notna(row["text"]) else ""
        if title and text:
            return f"{title}\n\n{text}"
        elif title:
            return title
        elif text:
            return text
        return ""

    df_raw["text"] = df_raw.apply(combine_title_text, axis=1)
    
    # Filter empty / blank text
    blank_mask = df_raw["text"].str.strip() == ""
    blank_count = int(blank_mask.sum())
    print(f"Blank/empty text rows removed: {blank_count}")
    df_valid = df_raw[~blank_mask].copy()

    # Deduplication
    dup_mask = df_valid.duplicated(subset=["text"], keep="first")
    dup_count = int(dup_mask.sum())
    print(f"Duplicate text rows removed: {dup_count}")
    df_clean = df_valid[~dup_mask].copy()

    # Format columns
    df_clean["label"] = df_clean["label"].map(ENGLISH_LABEL_MAP).astype(int)
    df_clean["language"] = "english"
    df_clean = df_clean[["text", "label", "language"]].reset_index(drop=True)

    cleaned_count = len(df_clean)
    real_count = int((df_clean["label"] == 0).sum())
    fake_count = int((df_clean["label"] == 1).sum())
    print(f"Cleaned row count: {cleaned_count} (REAL: {real_count}, FAKE: {fake_count})")

    # Save cleaned
    clean_csv_path = PROCESSED_DIR / "english_clean.csv"
    df_clean.to_csv(clean_csv_path, index=False, encoding="utf-8")
    print(f"Saved: {clean_csv_path}")

    # Stratified Split (70% train, 15% val, 15% test)
    train_df, temp_df = train_test_split(
        df_clean,
        test_size=0.30,
        stratify=df_clean["label"],
        random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=42,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save splits
    train_path = SPLITS_DIR / "english_train.csv"
    val_path = SPLITS_DIR / "english_validation.csv"
    test_path = SPLITS_DIR / "english_test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    # Leakage verification
    train_set = set(train_df["text"])
    val_set = set(val_df["text"])
    test_set = set(test_df["text"])

    leakage_train_val = len(train_set.intersection(val_set))
    leakage_train_test = len(train_set.intersection(test_set))
    leakage_val_test = len(val_set.intersection(test_set))
    total_leakage = leakage_train_val + leakage_train_test + leakage_val_test
    if total_leakage:
        raise RuntimeError(f"English duplicate leakage detected: {total_leakage}")

    stats = {
        "dataset": "WELFake_Dataset",
        "language": "english",
        "source_files": [str(RAW_ENG_PATH.relative_to(PROJECT_ROOT))],
        "original_count": original_count,
        "original_label_counts": {
            "source_label_0_fake": int((df_raw["label"] == 0).sum()),
            "source_label_1_real": int((df_raw["label"] == 1).sum()),
        },
        "source_label_mapping": {"0": LABEL_FAKE, "1": LABEL_REAL},
        "blank_removed": blank_count,
        "duplicates_removed": dup_count,
        "cleaned_count": cleaned_count,
        "real_count": real_count,
        "fake_count": fake_count,
        "splits": {
            "train": {
                "total": len(train_df),
                "real": int((train_df["label"] == 0).sum()),
                "fake": int((train_df["label"] == 1).sum()),
                "real_pct": round(float((train_df["label"] == 0).mean()) * 100, 2),
                "fake_pct": round(float((train_df["label"] == 1).mean()) * 100, 2),
            },
            "validation": {
                "total": len(val_df),
                "real": int((val_df["label"] == 0).sum()),
                "fake": int((val_df["label"] == 1).sum()),
                "real_pct": round(float((val_df["label"] == 0).mean()) * 100, 2),
                "fake_pct": round(float((val_df["label"] == 1).mean()) * 100, 2),
            },
            "test": {
                "total": len(test_df),
                "real": int((test_df["label"] == 0).sum()),
                "fake": int((test_df["label"] == 1).sum()),
                "real_pct": round(float((test_df["label"] == 0).mean()) * 100, 2),
                "fake_pct": round(float((test_df["label"] == 1).mean()) * 100, 2),
            },
        },
        "leakage": {
            "train_val_overlap": leakage_train_val,
            "train_test_overlap": leakage_train_test,
            "val_test_overlap": leakage_val_test,
            "total_leakage": total_leakage,
        },
    }
    return stats


def preprocess_hindi() -> dict:
    print("\n--- Processing Hindi Dataset ---")
    df_fake_raw = pd.read_csv(RAW_HIN_FAKE_PATH, encoding="utf-8-sig")
    df_true_raw = pd.read_csv(RAW_HIN_TRUE_PATH, encoding="utf-8-sig")

    raw_fake_count = len(df_fake_raw)
    raw_true_count = len(df_true_raw)
    original_count = raw_fake_count + raw_true_count
    print(f"Original row count: {original_count} (Fake: {raw_fake_count}, True: {raw_true_count})")

    # Drop Unnamed: 1 if present
    if "Unnamed: 1" in df_fake_raw.columns:
        df_fake_raw = df_fake_raw.drop(columns=["Unnamed: 1"])

    # Prepare Fake (label 1)
    df_fake = pd.DataFrame()
    df_fake["text"] = df_fake_raw["Articles"].fillna("").astype(str).str.strip()
    df_fake["label"] = LABEL_FAKE
    df_fake["language"] = "hindi"

    # Prepare True (label 0)
    df_true = pd.DataFrame()
    df_true["text"] = df_true_raw["Articles"].fillna("").astype(str).str.strip()
    df_true["label"] = LABEL_REAL
    df_true["language"] = "hindi"

    # Combine
    df_combined = pd.concat([df_fake, df_true], ignore_index=True)

    # Filter empty / blank text
    blank_mask = df_combined["text"].str.strip() == ""
    blank_count = int(blank_mask.sum())
    print(f"Blank/empty text rows removed: {blank_count}")
    df_valid = df_combined[~blank_mask].copy()

    # Deduplication
    dup_mask = df_valid.duplicated(subset=["text"], keep="first")
    dup_count = int(dup_mask.sum())
    print(f"Duplicate text rows removed: {dup_count}")
    df_clean = df_valid[~dup_mask].copy().reset_index(drop=True)

    cleaned_count = len(df_clean)
    real_count = int((df_clean["label"] == 0).sum())
    fake_count = int((df_clean["label"] == 1).sum())
    print(f"Cleaned row count: {cleaned_count} (REAL: {real_count}, FAKE: {fake_count})")

    # Unicode preservation check
    devanagari_count = int(df_clean["text"].apply(is_devanagari).sum())
    devanagari_pct = round(devanagari_count / cleaned_count * 100, 2)
    print(f"Devanagari script preserved: {devanagari_count}/{cleaned_count} ({devanagari_pct}%)")

    # Save cleaned
    clean_csv_path = PROCESSED_DIR / "hindi_clean.csv"
    df_clean.to_csv(clean_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {clean_csv_path}")

    # Stratified Split (70% train, 15% val, 15% test)
    train_df, temp_df = train_test_split(
        df_clean,
        test_size=0.30,
        stratify=df_clean["label"],
        random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=42,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save splits
    train_path = SPLITS_DIR / "hindi_train.csv"
    val_path = SPLITS_DIR / "hindi_validation.csv"
    test_path = SPLITS_DIR / "hindi_test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    # Leakage verification
    train_set = set(train_df["text"])
    val_set = set(val_df["text"])
    test_set = set(test_df["text"])

    leakage_train_val = len(train_set.intersection(val_set))
    leakage_train_test = len(train_set.intersection(test_set))
    leakage_val_test = len(val_set.intersection(test_set))
    total_leakage = leakage_train_val + leakage_train_test + leakage_val_test
    if total_leakage:
        raise RuntimeError(f"Hindi duplicate leakage detected: {total_leakage}")

    stats = {
        "dataset": "hindi_fake_and_true_news",
        "language": "hindi",
        "source_files": [
            str(RAW_HIN_FAKE_PATH.relative_to(PROJECT_ROOT)),
            str(RAW_HIN_TRUE_PATH.relative_to(PROJECT_ROOT)),
        ],
        "source_label_mapping": {
            "hindi_fake_news.csv": LABEL_FAKE,
            "hindi_true_news.csv": LABEL_REAL,
        },
        "original_count": original_count,
        "original_fake": raw_fake_count,
        "original_true": raw_true_count,
        "blank_removed": blank_count,
        "duplicates_removed": dup_count,
        "cleaned_count": cleaned_count,
        "real_count": real_count,
        "fake_count": fake_count,
        "unicode_devanagari_count": devanagari_count,
        "unicode_devanagari_pct": devanagari_pct,
        "splits": {
            "train": {
                "total": len(train_df),
                "real": int((train_df["label"] == 0).sum()),
                "fake": int((train_df["label"] == 1).sum()),
                "real_pct": round(float((train_df["label"] == 0).mean()) * 100, 2),
                "fake_pct": round(float((train_df["label"] == 1).mean()) * 100, 2),
            },
            "validation": {
                "total": len(val_df),
                "real": int((val_df["label"] == 0).sum()),
                "fake": int((val_df["label"] == 1).sum()),
                "real_pct": round(float((val_df["label"] == 0).mean()) * 100, 2),
                "fake_pct": round(float((val_df["label"] == 1).mean()) * 100, 2),
            },
            "test": {
                "total": len(test_df),
                "real": int((test_df["label"] == 0).sum()),
                "fake": int((test_df["label"] == 1).sum()),
                "real_pct": round(float((test_df["label"] == 0).mean()) * 100, 2),
                "fake_pct": round(float((test_df["label"] == 1).mean()) * 100, 2),
            },
        },
        "leakage": {
            "train_val_overlap": leakage_train_val,
            "train_test_overlap": leakage_train_test,
            "val_test_overlap": leakage_val_test,
            "total_leakage": total_leakage,
        },
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess and split fake-news datasets.")
    parser.add_argument(
        "--language",
        choices=("all", "english", "hindi"),
        default="all",
        help="Dataset to rebuild; selecting one language leaves the other outputs untouched.",
    )
    args = parser.parse_args()

    print("Starting Preprocessing & Splitting Pipeline...")
    summary_file = METADATA_DIR / "split_summary.json"
    if args.language == "all":
        summary = {
            "pipeline": "Fake News Preprocessing and Stratified Splitting",
            "random_state": 42,
            "split_ratio": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "label_mapping": {"REAL": 0, "FAKE": 1},
            "label_mapping_verification": {},
            "datasets": {},
        }
    elif summary_file.is_file():
        with summary_file.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
    else:
        raise FileNotFoundError(
            f"Cannot rebuild only {args.language}: existing metadata is missing at "
            f"{summary_file}. Run with --language all to create it."
        )

    summary["pipeline"] = "Fake News Preprocessing and Stratified Splitting"
    summary["random_state"] = 42
    summary["split_ratio"] = {"train": 0.70, "validation": 0.15, "test": 0.15}
    summary["label_mapping"] = {"REAL": LABEL_REAL, "FAKE": LABEL_FAKE}
    summary.setdefault("label_mapping_verification", {})
    summary.setdefault("datasets", {})
    summary["label_mapping_verification"]["hindi"] = (
        "hindi_fake_news.csv maps to FAKE label 1; hindi_true_news.csv maps "
        "to REAL label 0."
    )
    if "hindi" in summary["datasets"] and args.language != "hindi":
        summary["datasets"]["hindi"]["source_label_mapping"] = {
            "hindi_fake_news.csv": LABEL_FAKE,
            "hindi_true_news.csv": LABEL_REAL,
        }

    if args.language in ("all", "english"):
        summary["datasets"]["english"] = preprocess_english()
        summary["label_mapping_verification"]["english"] = (
            "WELFake source label 0 maps to project FAKE label 1; source label 1 "
            "maps to project REAL label 0."
        )
    if args.language in ("all", "hindi"):
        summary["datasets"]["hindi"] = preprocess_hindi()
        summary["label_mapping_verification"]["hindi"] = (
            "hindi_fake_news.csv maps to FAKE label 1; hindi_true_news.csv maps "
            "to REAL label 0."
        )

    summary_file = METADATA_DIR / "split_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata summary saved to: {summary_file}")
    report_file = METADATA_DIR / "preprocessing_report.md"
    report_file.write_text(render_report(summary), encoding="utf-8")
    print(f"Preprocessing report saved to: {report_file}")


def render_report(summary: dict) -> str:
    lines = [
        "# Fake News Preprocessing Report",
        "",
        "- Label mapping: REAL = 0, FAKE = 1",
        "- Split proportions: 70% train, 15% validation, 15% test",
        "- Random state: 42",
        "",
    ]
    for language, stats in summary["datasets"].items():
        lines.extend(
            [
                f"## {language.title()}",
                "",
                f"- Source files: {', '.join(stats['source_files'])}",
                f"- Original rows: {stats['original_count']:,}",
                f"- Cleaned rows: {stats['cleaned_count']:,}",
                f"- Removed missing/blank text: {stats['blank_removed']:,}",
                f"- Removed exact duplicates: {stats['duplicates_removed']:,}",
                f"- REAL: {stats['real_count']:,}",
                f"- FAKE: {stats['fake_count']:,}",
                "",
                "| Split | Rows | REAL | FAKE | REAL % | FAKE % |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for split_name in ("train", "validation", "test"):
            split = stats["splits"][split_name]
            lines.append(
                f"| {split_name.title()} | {split['total']:,} | {split['real']:,} | "
                f"{split['fake']:,} | {split['real_pct']:.2f}% | "
                f"{split['fake_pct']:.2f}% |"
            )
        if language == "english":
            source_counts = stats["original_label_counts"]
            lines.extend(
                [
                    "",
                    "Source label mapping: original 0 (FAKE) -> project 1; "
                    "original 1 (REAL) -> project 0.",
                    f"Raw source-label counts: 0={source_counts['source_label_0_fake']:,}, "
                    f"1={source_counts['source_label_1_real']:,}.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Source label mapping: hindi_fake_news.csv -> 1 (FAKE); "
                    "hindi_true_news.csv -> 0 (REAL).",
                ]
            )
        leakage = stats["leakage"]
        lines.extend(
            [
                "",
                "Duplicate leakage across splits: "
                f"{leakage['total_leakage']} "
                f"(train-validation {leakage['train_val_overlap']}, "
                f"train-test {leakage['train_test_overlap']}, "
                f"validation-test {leakage['val_test_overlap']}).",
                "",
            ]
        )
        if language == "hindi":
            lines.append(
                "Devanagari-containing rows: "
                f"{stats['unicode_devanagari_count']:,} "
                f"({stats['unicode_devanagari_pct']:.2f}%)."
            )
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
