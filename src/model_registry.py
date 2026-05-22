from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"


MODEL_REGISTRY = [
    {
        "key": "hybrid_logreg",
        "label": "Logistic Regression híbrida",
        "short_label": "Híbrido",
        "type": "classic",
        "family": "Clásico",
        "path": RESULTS_DIR / "logistic_regression_hybrid.joblib",
        "description": "Modelo clásico principal con TF-IDF + atributos manuales."
    },
    {
        "key": "random_forest_svd",
        "label": "RandomForest + SVD",
        "short_label": "RF+SVD",
        "type": "classic",
        "family": "Exploratorio",
        "path": RESULTS_DIR / "random_forest_svd.joblib",
        "description": "Modelo exploratorio con reducción de dimensionalidad y bosque aleatorio."
    },
    {
        "key": "beto_logreg",
        "label": "BETO + Logistic Regression",
        "short_label": "BETO",
        "type": "beto",
        "family": "Transformer",
        "path": RESULTS_DIR / "beto_logreg.joblib",
        "description": "Embeddings de BETO con clasificador lineal entrenado encima."
    },
]


def get_available_models():
    available = []
    for config in MODEL_REGISTRY:
        item = config.copy()
        item["exists"] = item["path"].exists()
        available.append(item)
    return available


def get_model_config(model_key: str):
    for config in MODEL_REGISTRY:
        if config["key"] == model_key:
            return config
    raise ValueError(f"Modelo no registrado: {model_key}")