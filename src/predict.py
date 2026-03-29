from typing import Dict, Any
import joblib

from .preprocessing import clean_text


LABEL_MAP = {
    0: "control",
    1: "anorexia"
}


def load_model(model_path: str):
    """
    Carga un modelo entrenado desde un archivo .joblib
    """
    return joblib.load(model_path)


def predict_text(model, text: str) -> Dict[str, Any]:
    """
    Recibe texto crudo, lo limpia y devuelve la predicción.
    """
    cleaned = clean_text(text)

    pred = int(model.predict([cleaned])[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba([cleaned])[0][1])

    return {
        "input_text": text,
        "cleaned_text": cleaned,
        "predicted_label": LABEL_MAP[pred],
        "predicted_numeric_label": pred,
        "probability_anorexia": probability
    }