from typing import Dict
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report,
    confusion_matrix, f1_score, recall_score, precision_score
)

def compute_metrics(y_true, y_pred, y_score) -> Dict:
    # Detectar si es multiclase
    n_classes = len(np.unique(y_true))
    is_multiclass = n_classes > 2

    # AUC con soporte multiclase
    try:
        if is_multiclass:
            auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        else:
            auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = None

    # Parámetro average para métricas multiclase
    avg = "macro" if is_multiclass else "binary"

    return {
        "roc_auc": auc,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }