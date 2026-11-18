import csv
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parent
TEXT_COLUMN_NAMES = {"text", "content", "article", "articles", "news", "body"}
TITLE_COLUMN_NAMES = {"title", "headline"}
LABEL_COLUMN_NAMES = {"label", "class", "target", "category", "verdict", "isfake"}

csv.field_size_limit(2**31 - 1)


def discover_dataset_files(search_dirs: list[Path] | None = None) -> list[Path]:
    if search_dirs is None:
        downloads = Path.home() / "Downloads"
        search_dirs = [
            PROJECT_ROOT,
            PROJECT_ROOT / "datasets",
            PROJECT_ROOT / "data" / "fake_news",
            downloads / "English",
            downloads / "hindi",
        ]

    files: set[Path] = set()
    for directory in search_dirs:
        if not directory.exists():
            continue
        recursive = directory.resolve() != PROJECT_ROOT.resolve()
        files.update(
            path.resolve()
            for pattern in ("*.csv", "*.tsv")
            for path in (directory.rglob(pattern) if recursive else directory.glob(pattern))
            if path.is_file()
        )
    return sorted(files)


def normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def detect_columns(headers: list[str], source: Path) -> tuple[list[str], str | None]:
    columns = {header: normalize_column_name(header) for header in headers if header.strip()}
    text_columns = [
        header
        for header, normalized in columns.items()
        if normalized in TITLE_COLUMN_NAMES | TEXT_COLUMN_NAMES
    ]
    label_columns = [
        header for header, normalized in columns.items() if normalized in LABEL_COLUMN_NAMES
    ]
    if len(label_columns) > 1:
        raise ValueError(f"{source}: multiple possible label columns: {label_columns}")
    if not text_columns:
        if len(columns) == 1 and not label_columns:
            text_columns = list(columns)
        else:
            raise ValueError(f"{source}: unable to identify a text column from {headers}")
    title_columns = [c for c in text_columns if normalize_column_name(c) in TITLE_COLUMN_NAMES]
    body_columns = [c for c in text_columns if c not in title_columns]
    ordered_text_columns = title_columns + body_columns
    return ordered_text_columns, label_columns[0] if label_columns else None


def infer_language(source: Path, text: str) -> str:
    path_parts = {part.casefold() for part in source.parts}
    stem = source.stem.casefold()
    if "hindi" in path_parts or "hindi" in stem:
        return "hindi"
    if "english" in path_parts or "english" in stem or "welfake" in stem:
        return "english"
    devanagari = sum("\u0900" <= char <= "\u097f" for char in text)
    letters = sum(char.isalpha() for char in text)
    if letters and devanagari / letters >= 0.2:
        return "hindi"
    if letters:
        return "english"
    raise ValueError(f"{source}: unable to infer language from path or text")


def _label_from_filename(source: Path) -> int | None:
    tokens = set(re.split(r"[^a-z0-9]+", source.stem.casefold()))
    fake_tokens = tokens & {"fake", "false", "hoax"}
    real_tokens = tokens & {"real", "true", "genuine"}
    if fake_tokens and real_tokens:
        raise ValueError(f"{source}: filename contains conflicting class indicators")
    if fake_tokens:
        return 1
    if real_tokens:
        return 0
    return None


def map_label(value: str, source: Path, has_label_column: bool) -> int | None:
    normalized = re.sub(r"[^a-z0-9.]", "", value.casefold())
    if not normalized:
        return None
    if normalized in {"real", "true", "genuine", "verified"}:
        return 0
    if normalized in {"fake", "false", "hoax", "misinformation"}:
        return 1
    if normalized in {"0", "1"}:
        if "welfake" not in source.stem.casefold():
            raise ValueError(
                f"{source}: numeric label {normalized} is ambiguous; "
                "use a documented label mapping"
            )
        return 1 if normalized == "0" else 0
    if not has_label_column:
        inferred = _label_from_filename(source)
        if inferred is not None:
            return inferred
    raise ValueError(f"{source}: unrecognized label value {value!r}")


def iter_dataset_records(source: Path) -> Iterator[dict[str, object]]:
    delimiter = "\t" if source.suffix.casefold() == ".tsv" else ","
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter, restkey="_extra_fields")
        headers = reader.fieldnames or []
        text_columns, label_column = detect_columns(headers, source)
        filename_label = _label_from_filename(source) if label_column is None else None
        for row_number, row in enumerate(reader, start=2):
            parts = [str(row.get(column) or "").strip() for column in text_columns]
            text = " ".join(part for part in parts if part)
            if label_column is None:
                label = filename_label
            else:
                raw_label = str(row.get(label_column) or "").strip()
                label = map_label(raw_label, source, has_label_column=True)
            yield {
                "source": str(source),
                "row_number": row_number,
                "text": text,
                "label": label,
                "language": infer_language(source, text),
            }


def normalize_text_for_deduplication(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split()).casefold()


def text_fingerprint(text: str) -> str:
    normalized = normalize_text_for_deduplication(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
