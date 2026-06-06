# Author: Andrés Cabrera Alvarado - A01798681
# Fecha de creación: 05/06/2026
# Archivo: tests/test_model_runtime.py
# Descripción general: Pruebas unitarias para el entorno de ejecución (runtime) que maneja los modelos clásicos y de BETO estándar. 
# Asegura el enrutamiento y normalización correctos de las predicciones a través del bundle.

from src.model_runtime import predict_single_with_runtime


# Simula un modelo de scikit-learn proporcionando métodos dummy para predict_proba y predict.
class FakeClassicModel:
    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.2, 0.8]])

    def predict(self, X):
        import numpy as np
        return np.array([1])


# Simula el comportamiento del BETOEmbedder devolviendo tensores falsos.
class FakeEmbedder:
    def encode(self, texts, batch_size=1):
        import numpy as np
        return np.ones((len(texts), 8), dtype=float)


# Simula el clasificador logístico que se ejecuta sobre los embeddings de BETO.
class FakeBetoClassifier:
    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.7, 0.3]])


# Verifica que el runtime procese correctamente un texto utilizando la configuración de un modelo clásico.
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


# Verifica que el runtime procese correctamente un texto utilizando la configuración estándar de BETO + regresión logística.
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