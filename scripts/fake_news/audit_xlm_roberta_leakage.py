"""Audit exact split overlap and visible source/topic cues without training."""

import hashlib
import json
from pathlib import Path
import re

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "data" / "fake_news" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results" / "fake_news" / "xlm_roberta"
LANGUAGES = ("english", "hindi")
SPLITS = ("train", "validation", "test")

SOURCE_CUES = {
    "english": {
        "Reuters": r"\bReuters\b",
        "Associated Press": r"\bAssociated Press\b",
        "BBC": r"\bBBC\b",
        "CNN": r"\bCNN\b",
        "New York Times": r"\bNew York Times\b",
        "Washington Post": r"\bWashington Post\b",
        "Fox News": r"\bFox News\b",
        "Breitbart": r"\bBreitbart\b",
        "Infowars": r"\bInfowars\b",
        "Sputnik": r"\bSputnik\b",
        "The Guardian": r"\bThe Guardian\b",
    },
    "hindi": {
        "PTI": r"\bPTI\b",
        "ANI": r"\bANI\b",
        "PIB": r"\bPIB\b",
        "Alt News": r"\bAlt News\b",
        "BOOM": r"\bBOOM\b",
        "India Today": r"\bIndia Today\b",
        "NDTV": r"\bNDTV\b",
        "Aaj Tak": r"\bAaj Tak\b",
        "The Quint": r"\bThe Quint\b",
    },
}
TOPIC_CUES = {
    "english": (
        "election", "vote", "president", "congress", "immigration",
        "brexit", "covid", "vaccine", "muslim", "israel", "russia", "economy",
    ),
    "hindi": (
        "चुनाव", "वोट", "मोदी", "कांग्रेस", "भाजपा", "सरकार", "कोरोना",
        "टीका", "मुस्लिम", "पाकिस्तान", "रूस", "अर्थव्यवस्था",
    ),
}


def audit_split(language: str, split: str) -> dict:
    path = SPLITS_DIR / f"{language}_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split: {path}")

    counts = {
        "rows": 0,
        "class_counts": {"REAL": 0, "FAKE": 0},
        "source_cues": {
            cue: {"REAL": 0, "FAKE": 0}
            for cue in SOURCE_CUES[language]
        },
        "topic_cues": {
            cue: {"REAL": 0, "FAKE": 0}
            for cue in TOPIC_CUES[language]
        },
        "fingerprints": set(),
    }

    for frame in pd.read_csv(
        path, encoding="utf-8-sig", usecols=["text", "label", "language"], chunksize=1000
    ):
        if not frame["language"].eq(language).all():
            raise ValueError(f"Unexpected language label in {path}")
        if not frame["label"].isin([0, 1]).all():
            raise ValueError(f"Unexpected class label in {path}")
        text = frame["text"].fillna("").astype(str)
        labels = frame["label"].astype(int)
        counts["rows"] += len(frame)
        for label, name in ((0, "REAL"), (1, "FAKE")):
            counts["class_counts"][name] += int(labels.eq(label).sum())

        for cue, pattern in SOURCE_CUES[language].items():
            present = text.str.contains(pattern, case=False, regex=True, na=False)
            for label, name in ((0, "REAL"), (1, "FAKE")):
                counts["source_cues"][cue][name] += int(
                    (present & labels.eq(label)).sum()
                )
        for cue in TOPIC_CUES[language]:
            present = text.str.contains(re.escape(cue), case=False, regex=True, na=False)
            for label, name in ((0, "REAL"), (1, "FAKE")):
                counts["topic_cues"][cue][name] += int(
                    (present & labels.eq(label)).sum()
                )
        for value in text:
            counts["fingerprints"].add(
                hashlib.sha256(value.encode("utf-8")).digest()
            )

    counts["unique_texts"] = len(counts["fingerprints"])
    counts["within_split_duplicate_rows"] = counts["rows"] - counts["unique_texts"]
    return counts


def audit() -> dict:
    report = {
        "purpose": "Diagnostic only; no model training, tuning, or checkpoint selection.",
        "label_mapping": {"REAL": 0, "FAKE": 1},
        "source_metadata_available": False,
        "limitations": [
            "The prepared split schema has text, label, and language only; source, "
            "publisher, publication date, and topic identifiers are unavailable.",
            "Publisher-name mentions are heuristic text cues, not verified publication "
            "provenance; articles can quote or discuss a publisher.",
            "Topic terms are a small manually chosen diagnostic list, not an exhaustive "
            "topic annotation or topic-disjoint split.",
            "Existing stratified random splits may distribute related events/topics "
            "across splits; no temporal or event-grouped holdout is available.",
        ],
        "languages": {},
    }

    for language in LANGUAGES:
        split_results = {
            split: audit_split(language, split)
            for split in SPLITS
        }
        overlaps = {}
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlaps[f"{left}_{right}"] = len(
                split_results[left]["fingerprints"]
                & split_results[right]["fingerprints"]
            )
        for value in split_results.values():
            value.pop("fingerprints")
        report["languages"][language] = {
            "splits": split_results,
            "exact_duplicate_overlap": overlaps,
        }

    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# XLM-RoBERTa Dataset Leakage Audit",
        "",
        "Diagnostic only; this audit does not train models or select checkpoints.",
        "",
        "## Exact duplicate overlap",
        "",
        "| Language | Train-validation | Train-test | Validation-test |",
        "|---|---:|---:|---:|",
    ]
    for language, data in report["languages"].items():
        overlap = data["exact_duplicate_overlap"]
        lines.append(
            f"| {language.title()} | {overlap['train_validation']} | "
            f"{overlap['train_test']} | {overlap['validation_test']} |"
        )

    for language, data in report["languages"].items():
        lines.extend(
            [
                "",
                f"## {language.title()} source/topic cue counts",
                "",
                "Counts are articles containing a cue (REAL/FAKE); cues can overlap.",
                "",
                "| Split | Rows | Source cue | REAL | FAKE | Topic cue | REAL | FAKE |",
                "|---|---:|---|---:|---:|---|---:|---:|",
            ]
        )
        for split, stats in data["splits"].items():
            source_cues = list(stats["source_cues"].items())
            topic_cues = list(stats["topic_cues"].items())
            for index in range(max(len(source_cues), len(topic_cues))):
                source = source_cues[index] if index < len(source_cues) else ("", {})
                topic = topic_cues[index] if index < len(topic_cues) else ("", {})
                lines.append(
                    f"| {split.title() if index == 0 else ''} | "
                    f"{stats['rows']:,}" if index == 0 else "|  |"
                )
                lines[-1] += (
                    f" | {source[0]} | {source[1].get('REAL', '')} | "
                    f"{source[1].get('FAKE', '')} | {topic[0]} | "
                    f"{topic[1].get('REAL', '')} | {topic[1].get('FAKE', '')} |"
                )

    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = audit()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "leakage_audit.json"
    markdown_path = RESULTS_DIR / "leakage_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print("Leakage audit completed without model training.")
    for language, data in report["languages"].items():
        print(f"{language}: exact duplicate overlaps {data['exact_duplicate_overlap']}")
    print(f"JSON audit: {json_path}")
    print(f"Markdown audit: {markdown_path}")


if __name__ == "__main__":
    main()
