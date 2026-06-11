# Author: Andrés Cabrera Alvarado - A01798681
# Author: Andrea Elizabeth Roman Varela - A01749760
# Author: Pablo Alonso Galván - A01748288
# Fecha de creación: 04/06/26
# Archivo: main.py
# Descripción general: ejecuta el flujo principal de entrenamiento y evaluación
# de los modelos base del proyecto usando los splits procesados.

from pathlib import Path

from src.data_loader import load_dataset
from src.train import run_baseline_experiment

# Define las rutas principales usadas para encontrar datos y guardar resultados.
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"


# Busca automáticamente los archivos de train/validation split
# con los nombres esperados dentro de la carpeta 'processed'.
def resolve_split_paths():
    train_candidates = [
        PROCESSED_DIR / "train_split.xlsx",
        PROCESSED_DIR / "data_train_split.xlsx",
    ]

    val_candidates = [
        PROCESSED_DIR / "validation_split.xlsx",
        PROCESSED_DIR / "data_validation_split.xlsx",
    ]

    train_path = next((p for p in train_candidates if p.exists()), None)
    val_path = next((p for p in val_candidates if p.exists()), None)

    if train_path is None:
        raise FileNotFoundError(
            "No se encontró el archivo de entrenamiento. "
            "Se esperaba alguno de estos:\n"
            f"- {train_candidates[0]}\n"
            f"- {train_candidates[1]}"
        )

    if val_path is None:
        raise FileNotFoundError(
            "No se encontró el archivo de validación. "
            "Se esperaba alguno de estos:\n"
            f"- {val_candidates[0]}\n"
            f"- {val_candidates[1]}"
        )

    return train_path, val_path


# Imprime en consola un resumen de las métricas obtenidas (ROC-AUC, Accuracy,
# Precision, Recall, F1 y la matriz de confusión) para un modelo evaluado.
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


# Función principal que orquesta la carga de datos, el entrenamiento de los modelos base,
# la evaluación cruzada y la generación de reportes y archivos de salida.
def main():
    print("Cargando datasets...")

    train_path, val_path = resolve_split_paths()

    train_df = load_dataset(str(train_path))
    val_df = load_dataset(str(val_path))

    print(f"Dataset de entrenamiento cargado desde: {train_path}")
    print(f"Total registros de entrenamiento: {len(train_df)}")

    print(f"Dataset de validación cargado desde: {val_path}")
    print(f"Total registros de validación: {len(val_df)}")

    results = run_baseline_experiment(
        train_df=train_df,
        val_df=val_df,
        output_dir=str(RESULTS_DIR)
    )

    print("\n=== RESULTADOS DEL ENTRENAMIENTO ===")

    for model_name, metrics in results.items():
        print_model_results(model_name, metrics)

    print(f"\nSe guardaron modelos, métricas y archivos de análisis en la carpeta '{RESULTS_DIR.name}/'.")


if __name__ == "__main__":
    main()