import pandas as pd
import pytest
from src.train import prepare_dataframe, build_logreg_pipeline
from src.predict import predict_text, predict_dataframe

#Fixture: modelo entrenado rápido
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

#Devuelve clase válida
def test_predict_returns_valid_label():
    model = _get_model()
    result = predict_text(model, "quiero vomitar todo y ser muy flaca thinspo ayuno")
    assert result["predicted_label"] in ("anorexia", "control", "incierto")

#Devuelve probabilidad válida (entre 0 y 1)
def test_predict_returns_valid_probability():
    model = _get_model()
    result = predict_text(model, "quiero vomitar todo y ser flaca purga ayuno thinspo")
    prob = result["probability_anorexia"]
    if prob is not None:
        assert 0.0 <= prob <= 1.0

#Devuelve estructura esperada del resultado
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

#No rompe con texto vacío o muy corto
def test_predict_empty_text():
    model = _get_model()
    result = predict_text(model, "")
    assert "predicted_label" in result

def test_predict_very_short_text():
    model = _get_model()
    result = predict_text(model, "ok")
    assert "predicted_label" in result

#Funciona con varios textos en secuencia (predict_dataframe)
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