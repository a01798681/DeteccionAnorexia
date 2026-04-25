from src.data_loader import load_dataset, validate_dataset
from src.train import run_baseline_experiment

def main():
    original_data_path = "data/data_train.xlsx"
    train_path = "data/processed/train_split.xlsx"
    validation_path = "data/processed/validation_split.xlsx"

    original_df = load_dataset(original_data_path)
    validate_dataset(original_df)

    train_df = load_dataset(train_path)
    validation_df = load_dataset(validation_path)

    validate_dataset(train_df)
    validate_dataset(validation_df)

    print("Dataset original cargado correctamente.")
    print(f"Total registros originales: {len(original_df)}")
    print(f"Registros de entrenamiento: {len(train_df)}")
    print(f"Registros de validación: {len(validation_df)}")

    results = run_baseline_experiment(
        train_df=train_df,
        val_df=validation_df,
        output_dir="results"
    )

    print("\n=== RESULTADOS DEL ENTRENAMIENTO ===")
    for model_name, metrics in results.items():
        print(f"\nModelo: {model_name}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Matriz de confusión:")
        print(metrics["confusion_matrix"])

        if "best_params" in metrics:
            print("Mejores parámetros encontrados:")
            for key, value in metrics["best_params"].items():
                print(f"  {key}: {value}")
            print(f"Mejor score CV: {metrics['best_cv_score']:.4f}")

    print("\nSe guardaron modelos, métricas y archivos de análisis en la carpeta 'results/'.")


if __name__ == "__main__":
    main()