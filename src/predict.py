from typing import Dict, Any
import joblib
import pandas as pd

from .preprocessing import clean_text


LABEL_MAP = {
    0: "control",
    1: "anorexia"
}


def load_model(model_path: str):
    return joblib.load(model_path)


def predict_text(
    model,
    text: str,
    anorexia_threshold: float = 0.70,
    control_threshold: float = 0.30,
    min_words: int = 4
) -> Dict[str, Any]:
    cleaned = clean_text(text if isinstance(text, str) else "")
    words = cleaned.split()

    if len(words) < min_words:
        return {
            "input_text": text,
            "cleaned_text": cleaned,
            "predicted_label": "insuficiente",
            "predicted_numeric_label": None,
            "probability_anorexia": None,
            "confidence": "baja",
            "message": "El texto es demasiado corto para una clasificación confiable.",
            "observations": "texto corto"
        }

    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba([cleaned])[0][1])

    if prob is None:
        pred = int(model.predict([cleaned])[0])
        label = LABEL_MAP[pred]
        confidence = "desconocida"
        message = "El modelo no proporciona probabilidad."
        observations = "sin probabilidad"
    else:
        if prob >= anorexia_threshold:
            label = "anorexia"
        elif prob <= control_threshold:
            label = "control"
        else:
            label = "incierto"

        if prob >= 0.85 or prob <= 0.15:
            confidence = "alta"
        elif prob >= 0.70 or prob <= 0.30:
            confidence = "media"
        else:
            confidence = "baja"

        message = "Clasificación realizada con umbrales de decisión."

        observations_list = []
        if len(words) < 6:
            observations_list.append("texto breve")
        if label == "incierto":
            observations_list.append("requiere revisión manual")

        observations = ", ".join(observations_list) if observations_list else "sin observaciones"

    return {
        "input_text": text,
        "cleaned_text": cleaned,
        "predicted_label": label,
        "predicted_numeric_label": None,
        "probability_anorexia": prob,
        "confidence": confidence,
        "message": message,
        "observations": observations
    }


def predict_dataframe(
    model,
    df: pd.DataFrame,
    text_column: str,
    anorexia_threshold: float = 0.70,
    control_threshold: float = 0.30,
    min_words: int = 4
) -> pd.DataFrame:
    """
    Clasifica todos los textos de un DataFrame y devuelve uno nuevo con columnas extra.
    """
    if text_column not in df.columns:
        raise ValueError(f"La columna '{text_column}' no existe en el archivo.")

    result_df = df.copy()

    predictions = result_df[text_column].apply(
        lambda x: predict_text(
            model=model,
            text=x,
            anorexia_threshold=anorexia_threshold,
            control_threshold=control_threshold,
            min_words=min_words
        )
    )

    result_df["cleaned_text"] = predictions.apply(lambda x: x["cleaned_text"])
    result_df["predicted_label"] = predictions.apply(lambda x: x["predicted_label"])
    result_df["probability_anorexia"] = predictions.apply(lambda x: x["probability_anorexia"])
    result_df["confidence"] = predictions.apply(lambda x: x["confidence"])
    result_df["message"] = predictions.apply(lambda x: x["message"])
    result_df["observations"] = predictions.apply(lambda x: x["observations"])

    return result_df