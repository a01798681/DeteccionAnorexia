from typing import Dict
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix


def compute_metrics(y_true, y_pred, y_score) -> Dict:
    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "roc_auc": auc,
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }