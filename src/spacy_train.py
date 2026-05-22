from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.data_loader import load_dataset
from src.spacy_preprocessing import normalize_texts_spacy
from src.train import (
    prepare_dataframe,
    optimize_hybrid_logistic_regression,
    evaluate_and_save,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"


def resolve_split_paths():
    train_candidates = [
        DATA_DIR / "train_split.xlsx",
        DATA_DIR / "data_train_split.xlsx",
    ]
    val_candidates = [
        DATA_DIR / "validation_split.xlsx",
        DATA_DIR / "data_validation_split.xlsx",
    ]

    train_path = next((p for p in train_candidates if p.exists()), None)
    val_path = next((p for p in val_candidates if p.exists()), None)

    if train_path is None or val_path is None:
        raise FileNotFoundError("No se encontraron los archivos de train/validation split.")

    return train_path, val_path


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_model_results(model_name: str, metrics: dict):
    print(f"\nModelo: {model_name}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print("Matriz de confusión:")
    print(metrics["confusion_matrix"])

    if "best_params" in metrics:
        print("Mejores parámetros encontrados:")
        for key, value in metrics["best_params"].items():
            print(f"  {key}: {value}")

    if "best_cv_score" in metrics:
        print(f"Mejor score CV: {metrics['best_cv_score']:.4f}")

    if "classification_report" in metrics:
        print("Classification report:")
        report = metrics["classification_report"]
        for label, values in report.items():
            if isinstance(values, dict):
                print(
                    f"  {label}: "
                    f"precision={values.get('precision', 0):.4f}, "
                    f"recall={values.get('recall', 0):.4f}, "
                    f"f1-score={values.get('f1-score', 0):.4f}, "
                    f"support={values.get('support', 0)}"
                )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_path, val_path = resolve_split_paths()
    train_df = prepare_dataframe(load_dataset(str(train_path)))
    val_df = prepare_dataframe(load_dataset(str(val_path)))

    print("Aplicando normalización con spaCy...")
    train_df["clean_text"] = normalize_texts_spacy(train_df["clean_text"].tolist())
    val_df["clean_text"] = normalize_texts_spacy(val_df["clean_text"].tolist())

    y_train = train_df["label"]
    y_val = val_df["label"]

    print("Entrenando Logistic Regression híbrida con texto normalizado por spaCy...")
    grid = optimize_hybrid_logistic_regression(train_df, y_train)
    best_model = grid.best_estimator_

    metrics = evaluate_and_save(
        "logistic_regression_hybrid_spacy",
        best_model,
        val_df[["clean_text"]],
        y_val,
        val_df,
        RESULTS_DIR
    )

    metrics["best_params"] = grid.best_params_
    metrics["best_cv_score"] = grid.best_score_

    print("\n=== RESULTADOS LOGISTIC REGRESSION HYBRID + SPACY ===")
    print_model_results("logistic_regression_hybrid_spacy", metrics)

    joblib.dump(best_model, RESULTS_DIR / "logistic_regression_hybrid_spacy.joblib")
    save_json(metrics, RESULTS_DIR / "logistic_regression_hybrid_spacy_metrics.json")

    print("\nSe guardaron los resultados de spaCy en la carpeta 'results/'.")


if __name__ == "__main__":
    main()