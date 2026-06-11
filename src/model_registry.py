# Author: Andrés Cabrera Alvarado - A01798681
# Author: Andrea Elizabeth Roman Varela - A01749760
# Author: Pablo Alonso Galván - A01748288
# Fecha de creación: 05/06/2026
# Archivo: src/model_registry.py
# Descripción general: Registro centralizado de modelos disponibles en la aplicación.
# Define configuraciones (nombres, rutas, descripciones) para cada tipo de modelo entrenado 
# (Clásico, BETO, Ensembles) y provee funciones para consultarlos.

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"


# Lista de diccionarios con la configuración e información meta de cada modelo soportado.
MODEL_REGISTRY = [
    {
        "key": "beto_llm_ensemble",
        "label": "BETO + LLM ensemble",
        "short_label": "BETO+LLM Ens",
        "type": "beto_llm_ensemble",
        "family": "Avanzado",
        "path": RESULTS_DIR / "beto_logreg.joblib",
        "description": "Método híbrido recomendado: combina el score de BETO y del LLM.",
        "recommended": True,
    },
    {
        "key": "beto_llm_cascade",
        "label": "BETO + LLM cascade",
        "short_label": "BETO+LLM Cas",
        "type": "beto_llm_cascade",
        "family": "Avanzado",
        "path": RESULTS_DIR / "beto_logreg.joblib",
        "description": "BETO decide si está seguro; el LLM entra solo en casos ambiguos.",
        "recommended": False,
    },
    {
        "key": "beto_logreg",
        "label": "BETO + Logistic Regression",
        "short_label": "BETO",
        "type": "beto",
        "family": "Transformer",
        "path": RESULTS_DIR / "beto_logreg.joblib",
        "description": "Embeddings de BETO con clasificador lineal entrenado encima.",
        "recommended": False,
    },
    {
        "key": "hybrid_logreg",
        "label": "Logistic Regression híbrida",
        "short_label": "Híbrido",
        "type": "classic",
        "family": "Clásico",
        "path": RESULTS_DIR / "logistic_regression_hybrid.joblib",
        "description": "Modelo clásico principal con TF-IDF + atributos manuales.",
        "recommended": False,
    },
    {
        "key": "random_forest_svd",
        "label": "RandomForest + SVD",
        "short_label": "RF+SVD",
        "type": "classic",
        "family": "Exploratorio",
        "path": RESULTS_DIR / "random_forest_svd.joblib",
        "description": "Modelo exploratorio con reducción de dimensionalidad y bosque aleatorio.",
        "recommended": False,
    },
]


# Retorna la lista completa de modelos registrados, agregando un flag booleano
# "exists" que indica si el archivo físico del modelo existe en el directorio de resultados.
def get_available_models():
    available = []
    for config in MODEL_REGISTRY:
        item = config.copy()
        item["exists"] = item["path"].exists()
        available.append(item)
    return available


# Busca y retorna la configuración completa de un modelo dado su 'key'.
# Lanza un ValueError si el 'key' no se encuentra en el registro.
def get_model_config(model_key: str):
    for config in MODEL_REGISTRY:
        if config["key"] == model_key:
            return config
    raise ValueError(f"Modelo no registrado: {model_key}")


# Retorna la clave (key) del modelo por defecto a utilizar en la aplicación web.
# Prioriza el modelo marcado como "recommended" y que exista físicamente.
# Si el recomendado no existe, retorna el primero disponible.
def get_default_model_key():
    available = get_available_models()
    existing = [m for m in available if m["exists"]]

    if not existing:
        return None

    recommended = next((m for m in existing if m.get("recommended")), None)
    if recommended is not None:
        return recommended["key"]

    return existing[0]["key"]