# Author: Andrés Cabrera Alvarado - A01798681
# Fecha de creación: 05/06/2026
# Archivo: tests/test_model_runtime_combo.py
# Descripción general: Pruebas unitarias para verificar la correcta integración de
# los modelos híbridos (Ensemble y Cascada) dentro del entorno de ejecución (runtime).
# Asegura que se ruteen correctamente las peticiones a los métodos combinados.

import src.model_runtime as mr


# Verifica que el runtime ejecute correctamente la predicción de un solo texto
# utilizando la configuración de Ensemble (BETO + LLM) y normalice el resultado.
def test_predict_single_with_runtime_ensemble(monkeypatch):
    runtime_bundle = {
        "config": {
            "key": "beto_llm_ensemble",
            "label": "BETO + LLM ensemble",
            "type": "beto_llm_ensemble"
        },
        "model": object(),
        "embedder": object(),
        "llm_callback": lambda text: {"label": "anorexia", "risk_score": 0.9, "reason": "ok"},
    }

    monkeypatch.setattr(
        mr,
        "predict_text_beto_llm_ensemble",
        lambda **kwargs: {
            "input_text": kwargs["text"],
            "cleaned_text": "texto limpio",
            "final_label": "anorexia",
            "predicted_numeric_label": 1,
            "final_score": 0.91,
            "confidence": "mixta",
            "message": "ensemble",
            "observations": "sin observaciones",
            "word_count": 5,
            "decision_source": "ensemble",
            "beto_probability": 0.88,
            "llm_risk_score": 0.95,
            "llm_label": "anorexia",
            "method": "beto_llm_ensemble",
        }
    )

    result = mr.predict_single_with_runtime(runtime_bundle, "quiero dejar de comer")
    assert result["model_label"] == "BETO + LLM ensemble"
    assert result["predicted_label"] == "anorexia"


# Verifica que el runtime ejecute correctamente la predicción de un solo texto
# utilizando la configuración en Cascada (BETO -> LLM) y normalice el resultado.
def test_predict_single_with_runtime_cascade(monkeypatch):
    runtime_bundle = {
        "config": {
            "key": "beto_llm_cascade",
            "label": "BETO + LLM cascade",
            "type": "beto_llm_cascade"
        },
        "model": object(),
        "embedder": object(),
        "llm_callback": lambda text: {"label": "control", "risk_score": 0.1, "reason": "ok"},
    }

    monkeypatch.setattr(
        mr,
        "predict_text_beto_llm_cascade",
        lambda **kwargs: {
            "input_text": kwargs["text"],
            "cleaned_text": "texto limpio",
            "final_label": "control",
            "predicted_numeric_label": 0,
            "final_score": 0.12,
            "confidence": "llm",
            "message": "cascade",
            "observations": "caso ambiguo",
            "word_count": 6,
            "decision_source": "llm",
            "beto_probability": 0.55,
            "llm_risk_score": 0.12,
            "llm_label": "control",
            "method": "beto_llm_cascade",
        }
    )

    result = mr.predict_single_with_runtime(runtime_bundle, "hoy comí con mi familia")
    assert result["model_label"] == "BETO + LLM cascade"
    assert result["predicted_label"] == "control"