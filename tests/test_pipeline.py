# Author: Andrés Cabrera Alvarado - A01798681
# Fecha de creación: 05/06/2026
# Archivo: tests/test_pipeline.py
# Descripción general: Pruebas unitarias para la creación y entrenamiento del pipeline clásico de aprendizaje automático (Logistic Regression). 
# Verifica que el pipeline se construya, entrene y genere predicciones válidas sin errores.

import pandas as pd
import pytest
from src.train import prepare_dataframe, build_logreg_pipeline

# Helper que genera un DataFrame mínimo reutilizable para simular un dataset
# balanceado y probar el entrenamiento del pipeline.
def _mini_df():
    return pd.DataFrame({
        "user_id":    [1, 2, 3, 4, 5, 6],
        "tweet_id":   [101, 102, 103, 104, 105, 106],
        "tweet_text": [
            "quiero ser flaca y dejar de comer ayuno purga",
            "me siento gorda no quiero comer nada thinspo",
            "salí con mis amigos a comer pizza",
            "me siento bien y tranquila hoy",
            "vomitar todo después de cenar grasa peso",
            "disfruté mucho la comida con mi familia",
        ],
        "class": ["anorexia", "anorexia", "control", "control", "anorexia", "control"],
    })


# Helper que prepara el dataset de prueba, inicializa el pipeline y
# ajusta (fit) el modelo. Devuelve el modelo entrenado y los datos preparados (X).
def _trained_model():
    df = _mini_df()
    prepared = prepare_dataframe(df)
    X = prepared["clean_text"]
    y = prepared["label"]
    model = build_logreg_pipeline()
    model.fit(X, y)
    return model, X

# Verifica que el pipeline se entrene correctamente usando un dataset pequeño
# y que el objeto resultante tenga el método 'predict'.
def test_pipeline_trains_small_dataset():
    model, X = _trained_model()
    assert hasattr(model, "predict")

# Verifica que las predicciones generadas por el pipeline pertenezcan al
# conjunto de clases válidas (0 o 1).
def test_pipeline_predicts_valid_classes():
    model, X = _trained_model()
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})

# Comprueba que el método predict_proba retorne probabilidades válidas
# dentro del rango de [0.0, 1.0].
def test_pipeline_proba_range():
    model, X = _trained_model()
    probs = model.predict_proba(X)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

# Asegura que la cantidad de predicciones coincida con el número de instancias ingresadas al modelo.
def test_pipeline_output_length():
    model, X = _trained_model()
    preds = model.predict(X)
    assert len(preds) == len(X)

# Verifica que el modelo no arroje errores al intentar predecir
# sobre un texto neutral, retornando una clase válida.
def test_pipeline_neutral_text():
    model, _ = _trained_model()
    neutral = pd.Series(["hoy fui al parque con mis amigos y comimos helado"])
    preds = model.predict(neutral)
    assert preds[0] in (0, 1)

# Verifica que el modelo pueda predecir sobre un texto explícito de
# alto riesgo sin fallar, retornando una clase válida.
def test_pipeline_risk_text():
    model, _ = _trained_model()
    risk = pd.Series(["quiero vomitar todo ayuno thinspo proana grasa flaca"])
    preds = model.predict(risk)
    assert preds[0] in (0, 1)