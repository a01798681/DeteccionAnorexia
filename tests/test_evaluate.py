# Author: Andrés Cabrera Alvarado - A01798681
# Author: Andrea Elizabeth Roman Varela - A01749760
# Author: Pablo Alonso Galván - A01748288
# Fecha de creación: 05/06/2026
# Archivo: tests/test_evaluate.py
# Descripción general: Pruebas unitarias para el cálculo de métricas de evaluación
# (accuracy, precision, recall, F1, ROC-AUC, matriz de confusión).
# Verifica que los cálculos usando scikit-learn coincidan con los valores esperados.

import numpy as np
import pytest
from src.evaluate import compute_metrics


#Datos de prueba compartidos
Y_TRUE  = np.array([1, 0, 1, 0, 1, 0])
Y_PRED  = np.array([1, 0, 1, 1, 0, 0])
Y_SCORE = np.array([0.9, 0.1, 0.8, 0.7, 0.4, 0.2])

# Función auxiliar que calcula y retorna las métricas usando los datos estáticos de prueba.
def _metrics():
    return compute_metrics(Y_TRUE, Y_PRED, Y_SCORE)

# Verifica que el accuracy se calcule correctamente dado el set de datos de prueba.
def test_accuracy_correct():
    m = _metrics()
    # TP=2, TN=2, FP=1, FN=1  → acc = 4/6
    assert abs(m["accuracy"] - (4 / 6)) < 1e-6

# Verifica que la precisión se calcule correctamente para la clase positiva (anorexia).
def test_precision_correct():
    m = _metrics()
    report = m["classification_report"]
    precision = report["1"]["precision"]
    # TP=2, FP=1 → precision = 2/3
    assert abs(precision - (2 / 3)) < 1e-6

# Verifica que el recall (exhaustividad) se calcule correctamente para la clase positiva.
def test_recall_correct():
    m = _metrics()
    report = m["classification_report"]
    recall = report["1"]["recall"]
    # TP=2, FN=1 → recall = 2/3
    assert abs(recall - (2 / 3)) < 1e-6

# Verifica que el F1-Score coincida con el cálculo manual de la media armónica entre precision y recall.
def test_f1_score_correct():
    m = _metrics()
    report = m["classification_report"]
    f1 = report["1"]["f1-score"]
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    expected_f1 = 2 * precision * recall / (precision + recall)
    assert abs(f1 - expected_f1) < 1e-6

# Comprueba que la matriz de confusión retornada tenga la dimensión esperada (2x2).
def test_confusion_matrix_shape():
    m = _metrics()
    cm = m["confusion_matrix"]
    assert len(cm) == 2
    assert len(cm[0]) == 2
    assert len(cm[1]) == 2

# Asegura que el área bajo la curva ROC (AUC) se calcule como un valor válido (entre 0 y 1).
def test_roc_auc_valid():
    m = _metrics()
    auc = m["roc_auc"]
    assert 0.0 <= auc <= 1.0