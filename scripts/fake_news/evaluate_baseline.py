"""Compare baselines on validation data, then test only each selected model."""

import argparse
import json
from pathlib import Path
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from baseline_common import (
    LANGUAGES,
    MODEL_LABELS,
    MODEL_NAMES,
    MODELS_DIR,
    RESULTS_DIR,
    calculate_metrics,
    classifier_path,
    read_split,
    vectorizer_path,
    write_json,
)


def save_confusion_matrix(matrix: list, labels: list, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(matrix),
        display_labels=labels,
    )
    display.plot(cmap="Blues", values_format="d", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def choose_best_model(validation_results: dict) -> str:
    """Choose by FAKE-class F1; ties use MODEL_NAMES order."""
    if set(validation_results) != set(MODEL_NAMES):
        raise ValueError("Validation results must contain all three baseline models.")
    return max(MODEL_NAMES, key=lambda name: validation_results[name]["f1"])


def evaluate_language(language: str) -> dict:
    language_dir = RESULTS_DIR / language
    language_dir.mkdir(parents=True, exist_ok=True)
    validation_texts, validation_labels = read_split(language, "validation")
    vectorizer_file = vectorizer_path(language)
    if not vectorizer_file.is_file():
        raise FileNotFoundError(f"Train baselines first; missing vectorizer: {vectorizer_file}")
    vectorizer = joblib.load(vectorizer_file)
    validation_matrix = vectorizer.transform(validation_texts)

    validation_results = {}
    evaluation_times = {}
    for model_name in MODEL_NAMES:
        model_file = classifier_path(language, model_name)
        if not model_file.is_file():
            raise FileNotFoundError(f"Train baselines first; missing model: {model_file}")
        classifier = joblib.load(model_file)
        start = time.perf_counter()
        predictions = classifier.predict(validation_matrix)
        metrics = calculate_metrics(validation_labels, predictions)
        evaluation_seconds = time.perf_counter() - start
        evaluation_times[model_name] = evaluation_seconds
        validation_results[model_name] = metrics
        metrics["evaluation_seconds"] = evaluation_seconds
        metrics["selection_metric"] = "FAKE-class F1 (label 1)"
        metrics_path = language_dir / f"validation_{model_name}.json"
        write_json(metrics_path, metrics)
        save_confusion_matrix(
            metrics["confusion_matrix"],
            metrics["confusion_matrix_labels"],
            language_dir / f"validation_{model_name}_confusion_matrix.png",
            f"{language.title()} validation: {MODEL_LABELS[model_name]}",
        )
        print(
            f"{language} validation / {MODEL_LABELS[model_name]}: "
            f"accuracy={metrics['accuracy']:.4f}, FAKE F1={metrics['f1']:.4f}, "
            f"evaluation={evaluation_seconds:.2f}s"
        )

    selected_model = choose_best_model(validation_results)
    print(
        f"{language} selected by validation FAKE F1: "
        f"{MODEL_LABELS[selected_model]} ({validation_results[selected_model]['f1']:.4f})"
    )

    # Test data is loaded only after validation-based model selection is complete.
    test_texts, test_labels = read_split(language, "test")
    test_matrix = vectorizer.transform(test_texts)
    selected_classifier = joblib.load(classifier_path(language, selected_model))
    start = time.perf_counter()
    test_predictions = selected_classifier.predict(test_matrix)
    test_metrics = calculate_metrics(test_labels, test_predictions)
    test_evaluation_seconds = time.perf_counter() - start
    test_metrics.update(
        {
            "language": language,
            "selected_model": selected_model,
            "selected_by": "validation FAKE-class F1 (label 1)",
            "validation_f1": validation_results[selected_model]["f1"],
            "evaluation_seconds": test_evaluation_seconds,
        }
    )
    write_json(language_dir / "selected_model_test_report.json", test_metrics)
    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        test_metrics["confusion_matrix_labels"],
        language_dir / f"test_{selected_model}_confusion_matrix.png",
        f"{language.title()} test: selected {MODEL_LABELS[selected_model]}",
    )

    result = {
        "language": language,
        "selected_model": selected_model,
        "validation_results": validation_results,
        "validation_evaluation_seconds": evaluation_times,
        "test_result": test_metrics,
    }
    write_json(language_dir / "evaluation_summary.json", result)
    return result


def build_comparison(results: list[dict]) -> None:
    rows = []
    markdown = [
        "# Fake News Baseline Comparison",
        "",
        "Selection uses validation FAKE-class F1 (label 1). Test metrics are shown "
        "only for the model selected for that language; test data was not used "
        "for tuning or selection.",
        "",
        "| Language | Model | Validation accuracy | Validation precision | Validation recall | "
        "Validation F1 (FAKE) | Train time (s) | Validation eval (s) | "
        "Test eval (s)* | Test accuracy* | Test precision* | Test recall* | Test F1* | "
        "Vectorizer bytes | Classifier bytes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        language = result["language"]
        training_path = MODELS_DIR / language / "training_metadata.json"
        with training_path.open(encoding="utf-8") as source:
            training = json.load(source)
        vectorizer_bytes = training["vectorizer"]["size_bytes"]
        for model_name in MODEL_NAMES:
            val = result["validation_results"][model_name]
            model_training = training["models"][model_name]
            selected = model_name == result["selected_model"]
            test = result["test_result"] if selected else {}
            row = {
                "language": language,
                "model": model_name,
                "validation_accuracy": val["accuracy"],
                "validation_precision_fake": val["precision"],
                "validation_recall_fake": val["recall"],
                "validation_f1_fake": val["f1"],
                "selected_by_validation": selected,
                "training_seconds_including_vectorizer": model_training[
                    "total_training_seconds"
                ],
                "classifier_fit_seconds": model_training["classifier_fit_seconds"],
                "vectorizer_fit_seconds": training["vectorizer"]["fit_seconds"],
                "validation_evaluation_seconds": val["evaluation_seconds"],
                "test_accuracy": test.get("accuracy"),
                "test_precision_fake": test.get("precision"),
                "test_recall_fake": test.get("recall"),
                "test_f1_fake": test.get("f1"),
                "test_evaluation_seconds": test.get("evaluation_seconds"),
                "vectorizer_file_size_bytes": vectorizer_bytes,
                "classifier_file_size_bytes": model_training["model_size_bytes"],
            }
            rows.append(row)
            show_test = lambda key: f"{test[key]:.4f}" if key in test else "n/a"
            markdown.append(
                f"| {language.title()} | {MODEL_LABELS[model_name]}"
                f"{' **(selected)**' if selected else ''} | "
                f"{val['accuracy']:.4f} | {val['precision']:.4f} | "
                f"{val['recall']:.4f} | {val['f1']:.4f} | "
                f"{model_training['total_training_seconds']:.2f} | "
                f"{val['evaluation_seconds']:.2f} | "
                f"{show_test('evaluation_seconds')} | {show_test('accuracy')} | "
                f"{show_test('precision')} | {show_test('recall')} | "
                f"{show_test('f1')} | {vectorizer_bytes:,} | "
                f"{model_training['model_size_bytes']:,} |"
            )
    markdown.extend(["", "## Validation classification details", ""])
    for result in results:
        language = result["language"]
        markdown.extend([f"### {language.title()}", ""])
        for model_name in MODEL_NAMES:
            metrics = result["validation_results"][model_name]
            markdown.extend(
                [
                    f"#### {MODEL_LABELS[model_name]}",
                    "",
                    f"- Accuracy: {metrics['accuracy']:.4f}",
                    f"- FAKE precision: {metrics['precision']:.4f}",
                    f"- FAKE recall: {metrics['recall']:.4f}",
                    f"- FAKE F1: {metrics['f1']:.4f}",
                    f"- Macro F1: {metrics['macro_f1']:.4f}",
                    "",
                    "| Class | Precision | Recall | F1 | Support |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for class_name in ("REAL", "FAKE"):
                class_result = metrics["class_metrics"][class_name]
                markdown.append(
                    f"| {class_name} | {class_result['precision']:.4f} | "
                    f"{class_result['recall']:.4f} | {class_result['f1-score']:.4f} | "
                    f"{int(class_result['support']):,} |"
                )
            markdown.extend(
                [
                    "",
                    "Confusion matrix (rows=actual, columns=predicted; REAL, FAKE): "
                    f"`{metrics['confusion_matrix']}`",
                    "",
                ]
            )
        test = result["test_result"]
        selected_name = result["selected_model"]
        markdown.extend(
            [
                f"#### Selected test evaluation: {MODEL_LABELS[selected_name]}",
                "",
                f"- Selected by validation FAKE F1: "
                f"{result['validation_results'][selected_name]['f1']:.4f}",
                f"- Accuracy: {test['accuracy']:.4f}",
                f"- FAKE precision: {test['precision']:.4f}",
                f"- FAKE recall: {test['recall']:.4f}",
                f"- FAKE F1: {test['f1']:.4f}",
                f"- Macro F1: {test['macro_f1']:.4f}",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for class_name in ("REAL", "FAKE"):
            class_result = test["class_metrics"][class_name]
            markdown.append(
                f"| {class_name} | {class_result['precision']:.4f} | "
                f"{class_result['recall']:.4f} | {class_result['f1-score']:.4f} | "
                f"{int(class_result['support']):,} |"
            )
        markdown.extend(
            [
                "",
                "Confusion matrix (rows=actual, columns=predicted; REAL, FAKE): "
                f"`{test['confusion_matrix']}`",
                "",
            ]
        )
    markdown.extend(
        [
            "",
            "*Test columns are populated only for validation-selected models.*",
            "",
            "Detailed per-model classification reports and validation confusion "
            "matrices are saved in the language-specific result directories. "
            "Test classification reports and confusion matrices are saved only "
            "for the selected model in each language.",
            "",
            "Evaluation timings include classifier prediction and metric calculation; "
            "the shared TF-IDF transformation is performed once per split and is not "
            "included in each model's timing.",
            "",
        ]
    )

    comparison_dir = RESULTS_DIR
    comparison_dir.mkdir(parents=True, exist_ok=True)
    csv_path = comparison_dir / "comparison.csv"
    import pandas as pd

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    (comparison_dir / "comparison.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    write_json(comparison_dir / "comparison.json", {"models": rows})
    print(f"Saved comparison reports to {comparison_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("all", *LANGUAGES), default="all")
    args = parser.parse_args()
    languages = LANGUAGES if args.language == "all" else (args.language,)
    results = [evaluate_language(language) for language in languages]
    build_comparison(results)


if __name__ == "__main__":
    main()
