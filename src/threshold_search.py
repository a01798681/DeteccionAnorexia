# Author: Andrés Cabrera Alvarado - A01798681
# Fecha de creación: 05/06/2026
# Archivo: src/threshold_search.py
# Descripción general: Script auxiliar para buscar y encontrar el mejor umbral (threshold) de decisión 
# a partir de las predicciones de validación, maximizando la precisión (accuracy).

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

INPUT_PATH = Path("results/phase3_llm_validation_predictions.csv")

# Lee las predicciones, itera sobre posibles umbrales de decisión (0.05 a 0.95),
# evalúa el accuracy para cada uno, y muestra el mejor umbral encontrado.
def main():
    df = pd.read_csv(INPUT_PATH)

    y_true = df["true_label"].astype(int)
    y_score = df["risk_score"].astype(float)

    best = None

    for threshold in [x / 100 for x in range(5, 96)]:
        y_pred = (y_score >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred).tolist()

        if best is None or acc > best["accuracy"]:
            best = {
                "threshold": threshold,
                "accuracy": acc,
                "confusion_matrix": cm
            }

    print("=== MEJOR UMBRAL FASE 3 ===")
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Accuracy: {best['accuracy']:.4f}")
    print("Matriz de confusión:")
    print(best["confusion_matrix"])

if __name__ == "__main__":
    main()