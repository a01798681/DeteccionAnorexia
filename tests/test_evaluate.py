import numpy as np
import pytest
from src.evaluate import compute_metrics


#Datos de prueba compartidos
Y_TRUE  = np.array([1, 0, 1, 0, 1, 0])
Y_PRED  = np.array([1, 0, 1, 1, 0, 0])
Y_SCORE = np.array([0.9, 0.1, 0.8, 0.7, 0.4, 0.2])

def _metrics():
    return compute_metrics(Y_TRUE, Y_PRED, Y_SCORE)

#Calcula accuracy correctamente
def test_accuracy_correct():
    m = _metrics()
    # TP=2, TN=2, FP=1, FN=1  → acc = 4/6
    assert abs(m["accuracy"] - (4 / 6)) < 1e-6

#Calcula precision correctamente
def test_precision_correct():
    m = _metrics()
    report = m["classification_report"]
    precision = report["1"]["precision"]
    # TP=2, FP=1 → precision = 2/3
    assert abs(precision - (2 / 3)) < 1e-6

#Calcula recall correctamente
def test_recall_correct():
    m = _metrics()
    report = m["classification_report"]
    recall = report["1"]["recall"]
    # TP=2, FN=1 → recall = 2/3
    assert abs(recall - (2 / 3)) < 1e-6

#Calcula F1-score correctamente
def test_f1_score_correct():
    m = _metrics()
    report = m["classification_report"]
    f1 = report["1"]["f1-score"]
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    expected_f1 = 2 * precision * recall / (precision + recall)
    assert abs(f1 - expected_f1) < 1e-6

#Calcula matriz de confusión con forma 2x2
def test_confusion_matrix_shape():
    m = _metrics()
    cm = m["confusion_matrix"]
    assert len(cm) == 2
    assert len(cm[0]) == 2
    assert len(cm[1]) == 2

#Calcula ROC-AUC con probabilidades válidas
def test_roc_auc_valid():
    m = _metrics()
    auc = m["roc_auc"]
    assert 0.0 <= auc <= 1.0