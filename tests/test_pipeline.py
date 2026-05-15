import pandas as pd
import pytest
from src.train import prepare_dataframe, build_logreg_pipeline

#Dataset mínimo reutilizable
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


def _trained_model():
    df = _mini_df()
    prepared = prepare_dataframe(df)
    X = prepared["clean_text"]
    y = prepared["label"]
    model = build_logreg_pipeline()
    model.fit(X, y)
    return model, X

#El pipeline entrena con dataset pequeño 
def test_pipeline_trains_small_dataset():
    model, X = _trained_model()
    assert hasattr(model, "predict")

#El pipeline predice clases válidas 
def test_pipeline_predicts_valid_classes():
    model, X = _trained_model()
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})

#predict_proba devuelve valores entre 0 y 1
def test_pipeline_proba_range():
    model, X = _trained_model()
    probs = model.predict_proba(X)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

#Longitud de predicciones coincide con entrada
def test_pipeline_output_length():
    model, X = _trained_model()
    preds = model.predict(X)
    assert len(preds) == len(X)

#El pipeline no rompe con texto neutral
def test_pipeline_neutral_text():
    model, _ = _trained_model()
    neutral = pd.Series(["hoy fui al parque con mis amigos y comimos helado"])
    preds = model.predict(neutral)
    assert preds[0] in (0, 1)

#El pipeline no rompe con texto de riesgo explícito
def test_pipeline_risk_text():
    model, _ = _trained_model()
    risk = pd.Series(["quiero vomitar todo ayuno thinspo proana grasa flaca"])
    preds = model.predict(risk)
    assert preds[0] in (0, 1)