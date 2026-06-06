# Author: Andrés Cabrera Alvarado - A01798681
# Fecha de creación: 05/06/2026
# Archivo: tests/test_random_forest.py
# Descripción general: Pruebas unitarias para la creación y entrenamiento del pipeline basado en Random Forest. 
# Verifica que el modelo se ajuste correctamente, produzca predicciones y probabilidades válidas.

import pandas as pd
from src.train import prepare_dataframe, build_random_forest_pipeline


# Helper que genera un DataFrame de prueba balanceado y estructurado, garantizando suficientes ejemplos para entrenar el modelo.
def _mini_df():
    return pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "tweet_id": [101, 102, 103, 104, 105, 106, 107, 108],
        "tweet_text": [
            "quiero ser flaca y dejar de comer ayuno purga",
            "me siento gorda no quiero comer nada thinspo",
            "vomitar todo después de cenar grasa peso",
            "ayuno y bodycheck todo el día",
            "salí con mis amigos a comer pizza",
            "me siento bien y tranquila hoy",
            "disfruté mucho la comida con mi familia",
            "todo bien hoy descansé y comí rico"
        ],
        "class": [
            "anorexia", "anorexia", "anorexia", "anorexia",
            "control", "control", "control", "control"
        ],
    })


# Helper que inicializa y ajusta (fit) el pipeline de Random Forest utilizando el DataFrame mínimo.
def _trained_model():
    df = _mini_df()
    prepared = prepare_dataframe(df)
    X = prepared[["clean_text"]]
    y = prepared["label"]
    model = build_random_forest_pipeline()
    model.fit(X, y)
    return model, X


# Verifica que el pipeline de Random Forest se inicialice y entrene correctamente, exponiendo el método 'predict'.
def test_random_forest_pipeline_trains():
    model, X = _trained_model()
    assert hasattr(model, "predict")


# Asegura que el modelo Random Forest genere predicciones dentro del espacio de clases esperadas (0 o 1).
def test_random_forest_predicts_valid_classes():
    model, X = _trained_model()
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})


# Comprueba que las probabilidades calculadas por el Random Forest estén limitadas al rango [0.0, 1.0].
def test_random_forest_predict_proba_range():
    model, X = _trained_model()
    probs = model.predict_proba(X)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


# Verifica que la salida del modelo tenga exactamente la misma cantidad
# de predicciones que las instancias proporcionadas en la entrada.
def test_random_forest_output_length_matches_input():
    model, X = _trained_model()
    preds = model.predict(X)
    assert len(preds) == len(X)


# Asegura que el modelo entrenado pueda predecir correctamente un
# texto nuevo sin fallar, usando el pipeline de transformación completo.
def test_random_forest_handles_new_text():
    model, _ = _trained_model()
    sample = pd.DataFrame({
        "clean_text": ["quiero dejar de comer y hacer ayuno porque me siento gorda"]
    })
    preds = model.predict(sample)
    assert preds[0] in (0, 1)