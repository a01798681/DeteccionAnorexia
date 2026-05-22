import pandas as pd
from src.train import prepare_dataframe, build_random_forest_pipeline


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


def _trained_model():
    df = _mini_df()
    prepared = prepare_dataframe(df)
    X = prepared[["clean_text"]]
    y = prepared["label"]
    model = build_random_forest_pipeline()
    model.fit(X, y)
    return model, X


def test_random_forest_pipeline_trains():
    model, X = _trained_model()
    assert hasattr(model, "predict")


def test_random_forest_predicts_valid_classes():
    model, X = _trained_model()
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})


def test_random_forest_predict_proba_range():
    model, X = _trained_model()
    probs = model.predict_proba(X)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_random_forest_output_length_matches_input():
    model, X = _trained_model()
    preds = model.predict(X)
    assert len(preds) == len(X)


def test_random_forest_handles_new_text():
    model, _ = _trained_model()
    sample = pd.DataFrame({
        "clean_text": ["quiero dejar de comer y hacer ayuno porque me siento gorda"]
    })
    preds = model.predict(sample)
    assert preds[0] in (0, 1)