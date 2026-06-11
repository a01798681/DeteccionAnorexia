# Author: Andrés Cabrera Alvarado - A01798681
# Author: Andrea Elizabeth Roman Varela - A01749760
# Author: Pablo Alonso Galván - A01748288
# Fecha de creación: 05/06/2026
# Archivo: tests/test_beto_llm_methods.py
# Descripción general: Pruebas unitarias para los métodos híbridos (Cascada y Ensemble)
# que combinan predicciones de BETO y modelos LLM. Verifica la correcta ponderación
# de scores y la delegación de decisiones en casos ambiguos.

import src.beto_llm_methods as combo


# Verifica que el modelo en cascada no llame al LLM si BETO tiene alta confianza.
def test_cascade_uses_beto_when_confident(monkeypatch):
    calls = {"llm": 0}

    def fake_beto(*args, **kwargs):
        return {
            "cleaned_text": "texto limpio",
            "word_count": 5,
            "probability_anorexia": 0.95,
            "predicted_label": "anorexia",
            "confidence": "alta",
            "observations": "incluye términos de riesgo",
        }

    def fake_llm(text):
        calls["llm"] += 1
        return {"label": "control", "risk_score": 0.10, "reason": "LLM"}

    monkeypatch.setattr(combo, "predict_text_beto", fake_beto)

    result = combo.predict_text_beto_llm_cascade(
        classifier=None,
        embedder=None,
        text="quiero dejar de comer",
        llm_callback=fake_llm,
    )

    assert result["decision_source"] == "beto"
    assert result["final_label"] == "anorexia"
    assert calls["llm"] == 0


# Verifica que el modelo en cascada llame al LLM si BETO tiene baja confianza (incierto).
def test_cascade_uses_llm_when_ambiguous(monkeypatch):
    def fake_beto(*args, **kwargs):
        return {
            "cleaned_text": "texto limpio",
            "word_count": 6,
            "probability_anorexia": 0.55,
            "predicted_label": "incierto",
            "confidence": "baja",
            "observations": "requiere revisión manual",
        }

    def fake_llm(text):
        return {
            "label": "anorexia",
            "risk_score": 0.88,
            "reason": "Caso ambiguo resuelto por LLM",
            "model_id": "llm-test",
        }

    monkeypatch.setattr(combo, "predict_text_beto", fake_beto)

    result = combo.predict_text_beto_llm_cascade(
        classifier=None,
        embedder=None,
        text="bodycheck otra vez",
        llm_callback=fake_llm,
    )

    assert result["decision_source"] == "llm"
    assert result["final_label"] == "anorexia"
    assert abs(result["final_score"] - 0.88) < 1e-9


# Verifica que el modelo ensemble calcule correctamente el score final ponderando
# las predicciones de BETO (alpha) y el LLM (beta).
def test_ensemble_combines_scores(monkeypatch):
    def fake_beto(*args, **kwargs):
        return {
            "cleaned_text": "texto limpio",
            "word_count": 7,
            "probability_anorexia": 0.80,
            "predicted_label": "anorexia",
            "confidence": "alta",
            "observations": "incluye términos de riesgo",
        }

    def fake_llm(text):
        return {
            "label": "anorexia",
            "risk_score": 0.60,
            "reason": "LLM",
            "model_id": "llm-test",
        }

    monkeypatch.setattr(combo, "predict_text_beto", fake_beto)

    result = combo.predict_text_beto_llm_ensemble(
        classifier=None,
        embedder=None,
        text="quiero ser flaca",
        alpha=0.7,
        beta=0.3,
        llm_callback=fake_llm,
    )

    expected = 0.7 * 0.80 + 0.3 * 0.60
    assert result["decision_source"] == "ensemble"
    assert abs(result["final_score"] - expected) < 1e-9
    assert result["final_label"] == "anorexia"