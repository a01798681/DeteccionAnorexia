import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from src.data_loader import load_dataset, validate_dataset
from src.llm_classifier import classify_text
from src.train import prepare_dataframe


def save_json(data, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/validation_split.xlsx",
        help="Ruta al conjunto de validación."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Carpeta donde se guardarán resultados."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Número máximo de filas a evaluar (útil para pruebas iniciales)."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.input)
    validate_dataset(df)

    prepared = prepare_dataframe(df)

    if args.limit is not None:
        prepared = prepared.head(args.limit).copy()

    results = []

    for idx, row in prepared.iterrows():
        text = row["tweet_text"] if "tweet_text" in row else row.get("text", "")
        true_label = row["label"]

        llm_result = classify_text(text)

        true_label_num = int(row["label"])
        true_label_text = "anorexia" if true_label_num == 1 else "control"

        results.append({
            "row_id": idx,
            "text": text,
            "clean_text": row.get("clean_text", ""),
            "true_label": true_label_num,
            "true_label_text": true_label_text,
            "predicted_label": llm_result["label"],
            "risk_score": float(llm_result["risk_score"]),
            "reason": llm_result["reason"],
            "model_id": llm_result.get("model_id", ""),
            "raw_response": llm_result.get("raw_response", "")
        })

    results_df = pd.DataFrame(results)

    # Mapear etiquetas a binario
    # y_true ya viene numérico desde prepare_dataframe()
    y_true = results_df["true_label"].astype(int)

    # predicted_label sí viene como texto desde el LLM
    y_pred = results_df["predicted_label"].map({"control": 0, "anorexia": 1}).astype(int)

    y_score = results_df["risk_score"].astype(float)

    metrics = {
        "n_samples": int(len(results_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "model_id": results_df["model_id"].iloc[0] if len(results_df) > 0 else ""
    }

    results_path = output_dir / "phase3_llm_validation_predictions.csv"
    metrics_path = output_dir / "phase3_llm_metrics.json"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    save_json(metrics, metrics_path)

    print("\n=== RESULTADOS FASE 3 (LLM) ===")
    print(f"Muestras evaluadas: {metrics['n_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("Matriz de confusión:")
    print(metrics["confusion_matrix"])
    print(f"\nResultados guardados en: {results_path}")
    print(f"Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()