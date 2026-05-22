from src.model_runtime import predict_single_with_runtime


class FakeClassicModel:
    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.2, 0.8]])

    def predict(self, X):
        import numpy as np
        return np.array([1])


class FakeEmbedder:
    def encode(self, texts, batch_size=1):
        import numpy as np
        return np.ones((len(texts), 8), dtype=float)


class FakeBetoClassifier:
    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.7, 0.3]])


def test_predict_single_with_runtime_classic():
    runtime_bundle = {
        "config": {
            "key": "hybrid_logreg",
            "label": "Logistic Regression híbrida",
            "type": "classic"
        },
        "model": FakeClassicModel(),
        "embedder": None
    }

    result = predict_single_with_runtime(runtime_bundle, "quiero dejar de comer")
    assert result["model_label"] == "Logistic Regression híbrida"
    assert result["predicted_label"] in ("anorexia", "control", "incierto")


def test_predict_single_with_runtime_beto():
    runtime_bundle = {
        "config": {
            "key": "beto_logreg",
            "label": "BETO + Logistic Regression",
            "type": "beto"
        },
        "model": FakeBetoClassifier(),
        "embedder": FakeEmbedder()
    }

    result = predict_single_with_runtime(runtime_bundle, "hoy comí con mi familia")
    assert result["model_label"] == "BETO + Logistic Regression"
    assert result["predicted_label"] in ("anorexia", "control", "incierto")