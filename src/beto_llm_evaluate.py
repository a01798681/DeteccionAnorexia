# Author: Andrés Cabrera Alvarado - A01798681
# Fecha de creación: 10/05/2026
# Archivo: src/beto_llm_evaluate.py
# Descripción general: Script de evaluación de los métodos híbridos BETO + LLM
#   (cascade y ensemble) sobre el conjunto de validación. Carga el clasificador
#   BETO entrenado, construye un callback con caché para el LLM, ejecuta cada
#   método sobre cada fila del dataset, calcula métricas y guarda predicciones
#   y resultados en archivos CSV y JSON.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.beto_embeddings import BETOEmbedder
from src.beto_llm_methods import (
    predict_text_beto_llm_cascade,
    predict_text_beto_llm_ensemble,
)
from src.data_loader import load_dataset
from src.evaluate import compute_metrics
from src.llm_classifier import classify_text
from src.train import prepare_dataframe


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"


# Busca los archivos de train/validation split con nombres alternativos.
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


# Guarda un diccionario como archivo JSON con formato legible.
def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Lee el caché de respuestas del LLM desde un archivo JSON.
def load_llm_cache(cache_path: Path):
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


# Persiste el caché de respuestas del LLM en disco.
def save_llm_cache(cache: dict, cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# Crea un callback que usa caché para evitar llamadas repetidas al LLM.
def build_cached_llm_callback(cache_path: Path):
    cache = load_llm_cache(cache_path)

    def callback(text: str):
        key = str(text).strip()
        if key not in cache:
            cache[key] = classify_text(key)
            save_llm_cache(cache, cache_path)
        return cache[key]

    return callback


# Ejecuta un método (cascade o ensemble) sobre todo el dataset de validación
# y devuelve el DataFrame de predicciones junto con las métricas calculadas.
def run_method(df_val, classifier, embedder, method_name, llm_callback, args):
    records = []

    for _, row in df_val.iterrows():
        text = row["tweet_text"]

        if method_name == "cascade":
            result = predict_text_beto_llm_cascade(
                classifier=classifier,
                embedder=embedder,
                text=text,
                beto_low=args.beto_low,
                beto_high=args.beto_high,
                anorexia_threshold=args.anorexia_threshold,
                control_threshold=args.control_threshold,
                min_words=args.min_words,
                llm_callback=llm_callback,
            )
        elif method_name == "ensemble":
            result = predict_text_beto_llm_ensemble(
                classifier=classifier,
                embedder=embedder,
                text=text,
                alpha=args.alpha,
                beta=args.beta,
                anorexia_threshold=args.anorexia_threshold,
                control_threshold=args.control_threshold,
                min_words=args.min_words,
                llm_callback=llm_callback,
            )
        else:
            raise ValueError(f"Método no soportado: {method_name}")

        records.append({
            "tweet_text": text,
            "true_label_num": int(row["label"]),
            "true_label": row["class"],
            **result,
        })

    pred_df = pd.DataFrame(records)

    y_true = pred_df["true_label_num"].astype(int).tolist()
    y_pred = pred_df["hard_numeric_label"].astype(int).tolist()
    y_score = pred_df["final_score"].astype(float).tolist()

    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics["method"] = method_name

    return pred_df, metrics


# Imprime las métricas de evaluación en consola con formato legible.
def print_metrics(title: str, metrics: dict):
    print(f"\n=== RESULTADOS {title.upper()} ===")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print("Matriz de confusión:")
    print(metrics["confusion_matrix"])

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


# Función principal: parsea argumentos, carga datos y modelo, ejecuta la
# evaluación de los métodos seleccionados y guarda resultados.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["cascade", "ensemble", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.70)
    parser.add_argument("--beta", type=float, default=0.30)
    parser.add_argument("--beto-low", type=float, default=0.20)
    parser.add_argument("--beto-high", type=float, default=0.80)
    parser.add_argument("--anorexia-threshold", type=float, default=0.70)
    parser.add_argument("--control-threshold", type=float, default=0.35)
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--cache-path", type=str, default=str(RESULTS_DIR / "llm_cache.json"))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _, val_path = resolve_split_paths()
    val_df = prepare_dataframe(load_dataset(str(val_path)))

    if args.limit is not None:
        val_df = val_df.head(args.limit).copy()

    beto_model_path = RESULTS_DIR / "beto_logreg.joblib"
    if not beto_model_path.exists():
        raise FileNotFoundError(
            "No se encontró results/beto_logreg.joblib. "
            "Primero corre: python -m src.beto_train"
        )

    classifier = joblib.load(beto_model_path)
    embedder = BETOEmbedder()
    llm_callback = build_cached_llm_callback(Path(args.cache_path))

    methods = ["cascade", "ensemble"] if args.method == "both" else [args.method]

    for method_name in methods:
        pred_df, metrics = run_method(
            df_val=val_df,
            classifier=classifier,
            embedder=embedder,
            method_name=method_name,
            llm_callback=llm_callback,
            args=args,
        )

        print_metrics(f"BETO + LLM ({method_name})", metrics)

        pred_df.to_csv(
            RESULTS_DIR / f"beto_llm_{method_name}_validation_predictions.csv",
            index=False,
            encoding="utf-8-sig"
        )
        save_json(metrics, RESULTS_DIR / f"beto_llm_{method_name}_metrics.json")

        print(f"\nSe guardaron los resultados de {method_name} en results/.")


if __name__ == "__main__":
    main()