import pandas as pd

from src.predict_beto import predict_text_beto, predict_dataframe_beto


class FakeEmbedder:
    def encode(self, texts, batch_size=1):
        import numpy as np
        return np.ones((len(texts), 8), dtype=float)


class FakeClassifier:
    def predict_proba(self, X):
        import numpy as np
        probs = []
        for i in range(len(X)):
            if i % 2 == 0:
                probs.append([0.20, 0.80])
            else:
                probs.append([0.80, 0.20])
        return np.array(probs)


def test_predict_text_beto_returns_structure():
    result = predict_text_beto(
        classifier=FakeClassifier(),
        embedder=FakeEmbedder(),
        text="quiero dejar de comer porque me siento gorda",
        anorexia_threshold=0.70,
        control_threshold=0.35,
        min_words=3
    )

    expected_keys = {
        "input_text", "cleaned_text", "predicted_label",
        "predicted_numeric_label", "probability_anorexia",
        "confidence", "message", "observations",
        "word_count", "vocab_coverage", "model_backend"
    }

    assert expected_keys.issubset(result.keys())


def test_predict_text_beto_probability_range():
    result = predict_text_beto(
        classifier=FakeClassifier(),
        embedder=FakeEmbedder(),
        text="quiero vomitar y ayunar",
        anorexia_threshold=0.70,
        control_threshold=0.35,
        min_words=3
    )

    assert 0.0 <= result["probability_anorexia"] <= 1.0


def test_predict_text_beto_valid_label():
    result = predict_text_beto(
        classifier=FakeClassifier(),
        embedder=FakeEmbedder(),
        text="quiero ser flaca y dejar de comer",
        anorexia_threshold=0.70,
        control_threshold=0.35,
        min_words=3
    )

    assert result["predicted_label"] in ("anorexia", "control", "incierto")


def test_predict_dataframe_beto_multiple_rows():
    df = pd.DataFrame({
        "tweet_text": [
            "quiero dejar de comer",
            "hoy comí con mi familia",
            "thinspo bodycheck",
        ]
    })

    result_df = predict_dataframe_beto(
        classifier=FakeClassifier(),
        embedder=FakeEmbedder(),
        df=df,
        text_column="tweet_text",
        anorexia_threshold=0.70,
        control_threshold=0.35,
        min_words=3
    )

    assert len(result_df) == 3
    assert "predicted_label" in result_df.columns
    assert "model_backend" in result_df.columns