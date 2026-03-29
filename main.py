import json
from src.data_loader import load_dataset, validate_dataset
from src.train import run_baseline_experiment


def main():
    data_path = "data/data_train.xlsx"

    df = load_dataset(data_path)
    validate_dataset(df)

    results = run_baseline_experiment(df, output_dir="results")

    print("\n=== RESULTADOS BASELINE ===")
    for model_name, metrics in results.items():
        print(f"\nModelo: {model_name}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Matriz de confusión:")
        print(metrics["confusion_matrix"])

    print("\nSe guardaron modelos y métricas en la carpeta 'results/'.")


if __name__ == "__main__":
    main()