# Author: Andrés Cabrera Alvarado - A01798681
# Fecha de creación: 05/06/2026
# Archivo: tests/test_predict.py
# Descripción general: Pruebas unitarias para las funciones de predicción usando modelos clásicos (predict.py). 
# Valida que las predicciones individuales y por lotes (DataFrames) mantengan la estructura de datos correcta
# y no fallen con casos límite (textos vacíos o cortos).

import pandas as pd
import pytest
from src.train import prepare_dataframe, build_logreg_pipeline
from src.predict import predict_text, predict_dataframe

# Helper que entrena un pipeline de regresión logística rápido con un DataFrame simulado pequeño y 
# lo devuelve para ser utilizado en las pruebas.
def _get_model():
    df = pd.DataFrame({
        "user_id":    [1, 2, 3, 4, 5, 6],
        "tweet_id":   [101, 102, 103, 104, 105, 106],
        "tweet_text": [
            "quiero ser flaca y dejar de comer ayuno purga",
            "me siento gorda no quiero comer nada thinspo",
            "salí con mis amigos a comer pizza feliz",
            "me siento bien y tranquila hoy todo bien",
            "vomitar todo después de cenar grasa peso flaca",
            "disfruté mucho la comida con mi familia contenta",
        ],
        "class": ["anorexia", "anorexia", "control", "control", "anorexia", "control"],
    })
    prepared = prepare_dataframe(df)
    model = build_logreg_pipeline()
    model.fit(prepared["clean_text"], prepared["label"])
    return model

# Verifica que la predicción sobre un texto retorne una etiqueta válida.
def test_predict_returns_valid_label():
    model = _get_model()
    result = predict_text(model, "quiero vomitar todo y ser muy flaca thinspo ayuno")
    assert result["predicted_label"] in ("anorexia", "control", "incierto")

# Comprueba que la probabilidad retornada por el modelo clásico esté en el rango de [0.0, 1.0].
def test_predict_returns_valid_probability():
    model = _get_model()
    result = predict_text(model, "quiero vomitar todo y ser flaca purga ayuno thinspo")
    prob = result["probability_anorexia"]
    if prob is not None:
        assert 0.0 <= prob <= 1.0

# Asegura que el diccionario de resultados contenga todas las métricas y atributos esperados.
def test_predict_result_structure():
    model = _get_model()
    result = predict_text(model, "hoy fui al gym y comí bien")
    expected_keys = {
        "input_text", "cleaned_text", "predicted_label",
        "predicted_numeric_label", "probability_anorexia",
        "confidence", "message", "observations",
        "word_count", "vocab_coverage",
    }
    assert expected_keys.issubset(result.keys())

# Verifica que la función de predicción maneje correctamente un texto vacío sin arrojar excepciones.
def test_predict_empty_text():
    model = _get_model()
    result = predict_text(model, "")
    assert "predicted_label" in result

# Verifica que la función de predicción maneje un texto excesivamente corto (1 palabra)
# retornando correctamente un resultado.
def test_predict_very_short_text():
    model = _get_model()
    result = predict_text(model, "ok")
    assert "predicted_label" in result

# Comprueba la predicción en lote procesando un DataFrame completo y
# verificando que se anexe la columna de predicciones correctamente.
def test_predict_dataframe_multiple_texts():
    model = _get_model()
    df = pd.DataFrame({
        "tweet_text": [
            "quiero ser flaca y dejar de comer",
            "me siento bien con mi cuerpo",
            "vomitar purga ayuno thinspo",
        ]
    })
    result_df = predict_dataframe(model, df, "tweet_text")
    assert len(result_df) == 3
    assert "predicted_label" in result_df.columns