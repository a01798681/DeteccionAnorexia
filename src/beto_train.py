# Author: Andrés Cabrera Alvarado - A01798681
# Fecha de creación: 05/06/2026
# Archivo: src/beto_train.py
# Descripción general: Script para entrenar y evaluar un modelo de regresión logística
#   utilizando embeddings generados por BETO (BERT en español).
#   El script realiza una búsqueda de hiperparámetros (GridSearchCV), calcula métricas
#   de evaluación y exporta el modelo, los resultados y archivos de análisis de errores.

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.beto_embeddings import BETOEmbedder
from src.data_loader import load_dataset
from src.evaluate import compute_metrics
from src.train import prepare_dataframe


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"


# Busca y resuelve las rutas de los archivos de entrenamiento y validación
# dentro del directorio de datos procesados. Lanza FileNotFoundError si no los encuentra.
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


# Guarda un diccionario o estructura de datos en formato JSON.
def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Construye un DataFrame con los detalles de las predicciones, añadiendo las 
# probabilidades, etiquetas legibles y el tipo de predicción (TP, TN, FP, FN).
def build_predictions_df(df_val: pd.DataFrame, y_true, y_pred, y_score):
    result = df_val.copy()
    result["true_label_num"] = y_true
    result["predicted_label_num"] = y_pred
    result["true_label"] = result["true_label_num"].map({0: "control", 1: "anorexia"})
    result["predicted_label"] = result["predicted_label_num"].map({0: "control", 1: "anorexia"})
    result["probability_anorexia"] = y_score
    result["is_correct"] = result["true_label_num"] == result["predicted_label_num"]

    def classify_error(row):
        if row["true_label_num"] == 1 and row["predicted_label_num"] == 1:
            return "true_positive"
        elif row["true_label_num"] == 0 and row["predicted_label_num"] == 0:
            return "true_negative"
        elif row["true_label_num"] == 0 and row["predicted_label_num"] == 1:
            return "false_positive"
        elif row["true_label_num"] == 1 and row["predicted_label_num"] == 0:
            return "false_negative"
        return "unknown"

    result["prediction_type"] = result.apply(classify_error, axis=1)
    return result


# Orquesta el flujo de entrenamiento: carga datos, genera embeddings BETO,
# entrena Logistic Regression (GridSearchCV), evalúa y exporta métricas y modelos.
def main():
    # Crear directorio de resultados si no existe
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar y preparar los datos
    train_path, val_path = resolve_split_paths()
    train_df = prepare_dataframe(load_dataset(str(train_path)))
    val_df = prepare_dataframe(load_dataset(str(val_path)))

    # Inicializar el embedder y generar características para train y validation
    print("Cargando BETO y generando embeddings...")
    embedder = BETOEmbedder()

    X_train = embedder.encode(train_df["clean_text"].tolist(), batch_size=16)
    X_val = embedder.encode(val_df["clean_text"].tolist(), batch_size=16)

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    # Configurar el clasificador base
    clf = LogisticRegression(max_iter=3000, random_state=42)

    # Definir la grilla de hiperparámetros a explorar
    param_grid = {
        "C": [0.1, 1.0, 5.0, 10.0],
        "class_weight": [None, "balanced"]
    }

    # Configurar validación cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Inicializar la búsqueda en grilla
    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    # Entrenar iterando sobre la grilla para encontrar los mejores hiperparámetros
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Realizar predicciones sobre el conjunto de validación
    y_pred = best_model.predict(X_val)
    y_score = best_model.predict_proba(X_val)[:, 1]

    # Calcular y almacenar métricas de desempeño
    metrics = compute_metrics(y_val, y_pred, y_score)
    metrics["best_params"] = grid.best_params_
    metrics["best_cv_score"] = grid.best_score_

    print("\n=== RESULTADOS BETO + LOGISTIC REGRESSION ===")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print("Matriz de confusión:")
    print(metrics["confusion_matrix"])
    print("Mejores parámetros encontrados:")
    for key, value in metrics["best_params"].items():
        print(f"  {key}: {value}")
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

    # Guardar el modelo entrenado y las métricas
    joblib.dump(best_model, RESULTS_DIR / "beto_logreg.joblib")
    save_json(metrics, RESULTS_DIR / "beto_logreg_metrics.json")

    # Generar y exportar reportes detallados de las predicciones para su análisis
    predictions_df = build_predictions_df(val_df, y_val, y_pred, y_score)
    predictions_df.to_csv(RESULTS_DIR / "beto_logreg_validation_predictions.csv", index=False, encoding="utf-8-sig")
    
    # Exportar subconjuntos de errores específicos para análisis manual
    predictions_df[predictions_df["prediction_type"] == "false_positive"].to_csv(
        RESULTS_DIR / "beto_logreg_false_positives.csv", index=False, encoding="utf-8-sig"
    )
    predictions_df[predictions_df["prediction_type"] == "false_negative"].to_csv(
        RESULTS_DIR / "beto_logreg_false_negatives.csv", index=False, encoding="utf-8-sig"
    )
    predictions_df[predictions_df["is_correct"] == False].to_csv(
        RESULTS_DIR / "beto_logreg_incorrect_predictions.csv", index=False, encoding="utf-8-sig"
    )

    print("\nSe guardaron los resultados de BETO en la carpeta 'results/'.")


if __name__ == "__main__":
    main()