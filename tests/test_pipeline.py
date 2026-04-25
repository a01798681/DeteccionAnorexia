import pandas as pd
from src.train import prepare_dataframe, build_logreg_pipeline


def test_logreg_pipeline_trains_and_predicts():
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "tweet_id": [101, 102, 103, 104],
        "tweet_text": [
            "quiero ser flaca y dejar de comer",
            "me siento gorda y no quiero comer",
            "salí con mis amigos a comer",
            "me siento bien y tranquilo"
        ],
        "class": ["anorexia", "anorexia", "control", "control"]
    })

    prepared = prepare_dataframe(df)

    X = prepared["clean_text"]
    y = prepared["label"]

    model = build_logreg_pipeline()
    model.fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape[0] == len(X)
    assert probs.shape[1] == 2