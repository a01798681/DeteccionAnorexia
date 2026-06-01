from __future__ import annotations

from typing import Any, Callable, Dict

from .llm_classifier import classify_text
from .predict_beto import predict_text_beto


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _label_from_score(score: float, anorexia_threshold: float, control_threshold: float) -> str:
    if score >= anorexia_threshold:
        return "anorexia"
    if score <= control_threshold:
        return "control"
    return "incierto"


def _numeric_label(label: str):
    if label == "anorexia":
        return 1
    if label == "control":
        return 0
    return None


def _hard_numeric_from_score(score: float) -> int:
    return 1 if score >= 0.5 else 0


def _normalize_llm_result(
    llm_result: Dict[str, Any],
    anorexia_threshold: float,
    control_threshold: float
) -> Dict[str, Any]:
    label = str(llm_result.get("label", "")).strip().lower()
    score = _safe_float(llm_result.get("risk_score"), default=None)
    reason = llm_result.get("reason", "")
    model_id = llm_result.get("model_id", "llm")

    if score is None:
        if label == "anorexia":
            score = 0.95
        elif label == "control":
            score = 0.05
        else:
            score = 0.50

    if label not in {"anorexia", "control", "incierto"}:
        label = _label_from_score(score, anorexia_threshold, control_threshold)

    return {
        "label": label,
        "risk_score": score,
        "reason": reason,
        "model_id": model_id,
    }


def predict_text_beto_llm_cascade(
    classifier,
    embedder,
    text: str,
    beto_low: float = 0.20,
    beto_high: float = 0.80,
    anorexia_threshold: float = 0.70,
    control_threshold: float = 0.35,
    min_words: int = 3,
    llm_callback: Callable[[str], Dict[str, Any]] = classify_text,
) -> Dict[str, Any]:
    """
    Opción 1: cascada BETO -> LLM.
    Si BETO tiene alta confianza, decide.
    Si BETO cae en zona ambigua, consulta al LLM.
    """
    beto_result = predict_text_beto(
        classifier=classifier,
        embedder=embedder,
        text=text,
        anorexia_threshold=anorexia_threshold,
        control_threshold=control_threshold,
        min_words=min_words,
    )

    beto_prob = _safe_float(beto_result.get("probability_anorexia"), default=0.50)

    llm_label = None
    llm_score = None
    llm_reason = None
    llm_model_id = None

    if beto_prob <= beto_low or beto_prob >= beto_high:
        final_score = beto_prob
        final_label = _label_from_score(final_score, anorexia_threshold, control_threshold)
        decision_source = "beto"
        reason = "Decisión tomada por BETO al estar fuera de la zona ambigua."
    else:
        llm_raw = llm_callback(text)
        llm_norm = _normalize_llm_result(
            llm_raw,
            anorexia_threshold=anorexia_threshold,
            control_threshold=control_threshold,
        )

        llm_label = llm_norm["label"]
        llm_score = llm_norm["risk_score"]
        llm_reason = llm_norm["reason"]
        llm_model_id = llm_norm["model_id"]

        final_score = llm_score
        final_label = llm_label
        decision_source = "llm"
        reason = llm_reason or "Decisión tomada por LLM en caso ambiguo."

    predicted_numeric_label = _numeric_label(final_label)
    hard_numeric_label = (
        predicted_numeric_label
        if predicted_numeric_label is not None
        else _hard_numeric_from_score(final_score)
    )

    return {
        "method": "beto_llm_cascade",
        "input_text": text,
        "cleaned_text": beto_result.get("cleaned_text"),
        "word_count": beto_result.get("word_count"),
        "beto_probability": beto_prob,
        "beto_label": beto_result.get("predicted_label"),
        "llm_label": llm_label,
        "llm_risk_score": llm_score,
        "llm_reason": llm_reason,
        "llm_model_id": llm_model_id,
        "final_label": final_label,
        "predicted_numeric_label": predicted_numeric_label,
        "hard_numeric_label": hard_numeric_label,
        "final_score": final_score,
        "decision_source": decision_source,
        "confidence": beto_result.get("confidence") if decision_source == "beto" else "llm",
        "message": reason,
        "observations": beto_result.get("observations"),
    }


def predict_text_beto_llm_ensemble(
    classifier,
    embedder,
    text: str,
    alpha: float = 0.70,
    beta: float = 0.30,
    anorexia_threshold: float = 0.70,
    control_threshold: float = 0.35,
    min_words: int = 3,
    llm_callback: Callable[[str], Dict[str, Any]] = classify_text,
) -> Dict[str, Any]:
    """
    Opción 2: ensamble por puntaje.
    final_score = alpha * BETO + beta * LLM
    """
    beto_result = predict_text_beto(
        classifier=classifier,
        embedder=embedder,
        text=text,
        anorexia_threshold=anorexia_threshold,
        control_threshold=control_threshold,
        min_words=min_words,
    )
    beto_prob = _safe_float(beto_result.get("probability_anorexia"), default=0.50)

    llm_raw = llm_callback(text)
    llm_norm = _normalize_llm_result(
        llm_raw,
        anorexia_threshold=anorexia_threshold,
        control_threshold=control_threshold,
    )
    llm_score = llm_norm["risk_score"]

    total_weight = alpha + beta
    if total_weight == 0:
        raise ValueError("alpha + beta no puede ser 0.")

    final_score = (alpha * beto_prob + beta * llm_score) / total_weight
    final_label = _label_from_score(final_score, anorexia_threshold, control_threshold)

    predicted_numeric_label = _numeric_label(final_label)
    hard_numeric_label = (
        predicted_numeric_label
        if predicted_numeric_label is not None
        else _hard_numeric_from_score(final_score)
    )

    return {
        "method": "beto_llm_ensemble",
        "input_text": text,
        "cleaned_text": beto_result.get("cleaned_text"),
        "word_count": beto_result.get("word_count"),
        "beto_probability": beto_prob,
        "beto_label": beto_result.get("predicted_label"),
        "llm_label": llm_norm["label"],
        "llm_risk_score": llm_score,
        "llm_reason": llm_norm["reason"],
        "llm_model_id": llm_norm["model_id"],
        "final_label": final_label,
        "predicted_numeric_label": predicted_numeric_label,
        "hard_numeric_label": hard_numeric_label,
        "final_score": final_score,
        "decision_source": "ensemble",
        "confidence": "mixta",
        "message": f"Score combinado: {alpha:.2f} * BETO + {beta:.2f} * LLM",
        "observations": beto_result.get("observations"),
    }