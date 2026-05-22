from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from .beto_embeddings import BETOEmbedder
from .preprocessing import clean_text
from .term_lexicon import get_term_sets


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


def _contains_any(text: str, terms) -> bool:
    text = text.lower()
    return any(term in text for term in terms)


def predict_text_beto(
    classifier,
    embedder: BETOEmbedder,
    text: str,
    anorexia_threshold: float = 0.70,
    control_threshold: float = 0.35,
    min_words: int = 3
) -> Dict[str, Any]:
    term_sets = get_term_sets()
    risk_terms = term_sets["risk_terms"]
    positive_safe_terms = term_sets["positive_safe_terms"]
    negation_safe_terms = term_sets["negation_safe_terms"]

    cleaned = clean_text(text if isinstance(text, str) else "")
    words = cleaned.split()
    word_count = len(words)

    has_risk_terms = _contains_any(cleaned, risk_terms)
    has_generic_safe_phrase = _contains_any(
        cleaned,
        GENERIC_SAFE_PHRASES + positive_safe_terms + negation_safe_terms
    )

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
            "vocab_coverage": None,
            "model_backend": "BETO"
        }

    X = embedder.encode([cleaned], batch_size=1)
    prob = float(classifier.predict_proba(X)[0][1])

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

    observations_list = []
    if word_count < min_words:
        observations_list.append("texto corto")
    if label == "incierto":
        observations_list.append("requiere revisión manual")
    if has_risk_terms:
        observations_list.append("incluye términos de riesgo")

    observations = ", ".join(observations_list) if observations_list else "sin observaciones"

    predicted_numeric_label = None if label == "incierto" else (1 if label == "anorexia" else 0)

    return {
        "input_text": text,
        "cleaned_text": cleaned,
        "predicted_label": label,
        "predicted_numeric_label": predicted_numeric_label,
        "probability_anorexia": prob,
        "confidence": confidence,
        "message": "Clasificación realizada con BETO + Logistic Regression.",
        "observations": observations,
        "word_count": word_count,
        "vocab_coverage": None,
        "model_backend": "BETO"
    }


def predict_dataframe_beto(
    classifier,
    embedder: BETOEmbedder,
    df: pd.DataFrame,
    text_column: str,
    anorexia_threshold: float = 0.70,
    control_threshold: float = 0.35,
    min_words: int = 3
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"La columna '{text_column}' no existe en el archivo.")

    result_df = df.copy()

    predictions = result_df[text_column].apply(
        lambda x: predict_text_beto(
            classifier=classifier,
            embedder=embedder,
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
    result_df["model_backend"] = predictions.apply(lambda x: x["model_backend"])

    return result_df