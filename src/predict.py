from typing import Dict, Any
import joblib
import pandas as pd

from .preprocessing import clean_text


LABEL_MAP = {
    0: "control",
    1: "anorexia"
}

GENERIC_SAFE_PHRASES = [
    "hola",
    "que tal",
    "qué tal",
    "como estas",
    "cómo estás",
    "todo bien",
    "buenos dias",
    "buenos días",
    "buenas tardes",
    "buenas noches",
    "gracias",
    "saludos",
    "me siento bien",
    "estoy bien",
    "todo tranquilo",
    "con mis amigos",
    "con mi familia"
]

RISK_TERMS = [
    "vomit", "vomitar", "vomitando", "purga", "purging",
    "adelgazar", "bajar de peso", "peso", "gorda", "flaca",
    "abdomen", "cuerpo", "grasa", "ayuno", "ayunas",
    "thinspo", "thinspiration", "proana", "#thinspo", "#thinspiration",
    "#proana", "#ana", "#mia", "dejar de comer", "no quiero comer",
    "quiero ser flaca", "me siento gorda", "anorexia", "bulimia"
]


def load_model(model_path: str):
    return joblib.load(model_path)


def _build_model_input(model, cleaned_text: str):
    if hasattr(model, "named_steps") and "features" in model.named_steps:
        return pd.DataFrame({"clean_text": [cleaned_text]})
    return [cleaned_text]


def _contains_any(text: str, terms) -> bool:
    text = text.lower()
    return any(term in text for term in terms)


def _get_tfidf_vocabulary(model):
    if not hasattr(model, "named_steps"):
        return set()

    # Modelo híbrido
    if "features" in model.named_steps:
        preprocessor = model.named_steps["features"]
        try:
            tfidf = preprocessor.named_transformers_["tfidf"]
            return set(tfidf.vocabulary_.keys())
        except Exception:
            return set()

    # Modelo simple
    if "tfidf" in model.named_steps:
        try:
            return set(model.named_steps["tfidf"].vocabulary_.keys())
        except Exception:
            return set()

    return set()


def _estimate_vocab_coverage(cleaned_text: str, vocabulary: set) -> float:
    tokens = cleaned_text.split()
    if not tokens:
        return 0.0
    covered = sum(1 for token in tokens if token in vocabulary)
    return covered / len(tokens)


def predict_text(
    model,
    text: str,
    anorexia_threshold: float = 0.70,
    control_threshold: float = 0.30,
    min_words: int = 4
) -> Dict[str, Any]:
    cleaned = clean_text(text if isinstance(text, str) else "")
    words = cleaned.split()
    word_count = len(words)

    has_risk_terms = _contains_any(cleaned, RISK_TERMS)
    has_generic_safe_phrase = _contains_any(cleaned, GENERIC_SAFE_PHRASES)

    vocabulary = _get_tfidf_vocabulary(model)
    vocab_coverage = _estimate_vocab_coverage(cleaned, vocabulary)

    # Regla 1: saludo / texto casual sin señales de riesgo
    if has_generic_safe_phrase and not has_risk_terms:
        return {
            "input_text": text,
            "cleaned_text": cleaned,
            "predicted_label": "control",
            "predicted_numeric_label": 0,
            "probability_anorexia": 0.05,
            "confidence": "media",
            "message": "Texto general o casual sin señales claras de riesgo.",
            "observations": "texto casual/general",
            "word_count": word_count,
            "vocab_coverage": vocab_coverage
        }

    # Regla 2: cobertura de vocabulario muy baja y sin riesgo
    if vocab_coverage < 0.20 and not has_risk_terms:
        return {
            "input_text": text,
            "cleaned_text": cleaned,
            "predicted_label": "incierto",
            "predicted_numeric_label": None,
            "probability_anorexia": None,
            "confidence": "baja",
            "message": "El texto tiene muy poca cobertura del vocabulario del modelo.",
            "observations": "fuera de distribución, requiere revisión manual",
            "word_count": word_count,
            "vocab_coverage": vocab_coverage
        }

    model_input = _build_model_input(model, cleaned)

    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(model_input)[0][1])

    if prob is None:
        pred = int(model.predict(model_input)[0])
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
        if word_count < min_words:
            observations_list.append("texto corto")
        if label == "incierto":
            observations_list.append("requiere revisión manual")
        if vocab_coverage < 0.40:
            observations_list.append("cobertura baja del vocabulario")

        observations = ", ".join(observations_list) if observations_list else "sin observaciones"

    predicted_numeric_label = None if label == "incierto" else (1 if label == "anorexia" else 0)

    return {
        "input_text": text,
        "cleaned_text": cleaned,
        "predicted_label": label,
        "predicted_numeric_label": predicted_numeric_label,
        "probability_anorexia": prob,
        "confidence": confidence,
        "message": message,
        "observations": observations,
        "word_count": word_count,
        "vocab_coverage": vocab_coverage
    }


def predict_dataframe(
    model,
    df: pd.DataFrame,
    text_column: str,
    anorexia_threshold: float = 0.70,
    control_threshold: float = 0.30,
    min_words: int = 4
) -> pd.DataFrame:
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
    result_df["word_count"] = predictions.apply(lambda x: x["word_count"])
    result_df["vocab_coverage"] = predictions.apply(lambda x: x["vocab_coverage"])

    return result_df