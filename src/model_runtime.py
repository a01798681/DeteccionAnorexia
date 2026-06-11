# Author: Andrés Cabrera Alvarado - A01798681
# Author: Andrea Elizabeth Roman Varela - A01749760
# Author: Pablo Alonso Galván - A01748288
# Fecha de creación: 05/06/2026
# Archivo: src/model_runtime.py
# Descripción general: Módulo de ejecución que abstrae la carga e inferencia
# de todos los tipos de modelos (Clásicos, BETO, Ensembles y Cascada).
# Maneja la carga del modelo, inicialización de dependencias (BETOEmbedder,
# LLM cache) y expone una API unificada para predecir textos individuales
# o DataFrames completos.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd

from .beto_embeddings import BETOEmbedder
from .beto_llm_methods import (
    predict_text_beto_llm_cascade,
    predict_text_beto_llm_ensemble,
)
from .llm_classifier import classify_text
from .model_registry import get_model_config
from .predict import predict_text, predict_dataframe
from .predict_beto import predict_text_beto, predict_dataframe_beto


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
LLM_CACHE_PATH = RESULTS_DIR / "llm_cache.json"


# Carga la caché del LLM (JSON) desde el disco para no repetir consultas idénticas.
def load_llm_cache(cache_path: Path):
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


# Guarda la caché del LLM al disco de forma segura.
def save_llm_cache(cache: dict, cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# Construye un callback que clasifica un texto usando el LLM
# pero aplicando la lógica de caché para ahorrar llamadas a la API.
def build_cached_llm_callback(cache_path: Path = LLM_CACHE_PATH):
    cache = load_llm_cache(cache_path)

    def callback(text: str):
        key = str(text).strip()
        if key not in cache:
            cache[key] = classify_text(key)
            save_llm_cache(cache, cache_path)
        return cache[key]

    return callback


# Normaliza el diccionario de salida de los modelos combinados (Ensemble/Cascada)
# para que tengan una estructura consistente con la aplicación y la UI.
def _normalize_combo_result(raw: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "input_text": raw.get("input_text"),
        "cleaned_text": raw.get("cleaned_text"),
        "predicted_label": raw.get("final_label"),
        "predicted_numeric_label": raw.get("predicted_numeric_label"),
        "probability_anorexia": raw.get("final_score"),
        "confidence": raw.get("confidence", "mixta"),
        "message": raw.get("message", ""),
        "observations": raw.get("observations", ""),
        "word_count": raw.get("word_count", 0),
        "vocab_coverage": None,
        "model_backend": config["type"],
        "decision_source": raw.get("decision_source"),
        "beto_probability": raw.get("beto_probability"),
        "llm_risk_score": raw.get("llm_risk_score"),
        "llm_label": raw.get("llm_label"),
        "method": raw.get("method"),
    }


# Carga y ensambla todas las piezas requeridas para ejecutar un modelo específico.
# Retorna un diccionario con la configuración, el clasificador, el embedder y el callback del LLM.
def load_runtime_bundle(model_key: str):
    config = get_model_config(model_key)

    if not config["path"].exists():
        raise FileNotFoundError(f"No se encontró el modelo: {config['path']}")

    if config["type"] == "classic":
        model = joblib.load(config["path"])
        return {
            "config": config,
            "model": model,
            "embedder": None,
            "llm_callback": None,
        }

    if config["type"] == "beto":
        classifier = joblib.load(config["path"])
        embedder = BETOEmbedder()
        return {
            "config": config,
            "model": classifier,
            "embedder": embedder,
            "llm_callback": None,
        }

    if config["type"] in {"beto_llm_ensemble", "beto_llm_cascade"}:
        classifier = joblib.load(config["path"])
        embedder = BETOEmbedder()
        llm_callback = build_cached_llm_callback()
        return {
            "config": config,
            "model": classifier,
            "embedder": embedder,
            "llm_callback": llm_callback,
        }

    raise ValueError(f"Tipo de modelo no soportado: {config['type']}")


# Realiza la predicción de un solo texto utilizando el bundle.
# Enruta la predicción a la función correspondiente según el tipo de modelo.
def predict_single_with_runtime(
    runtime_bundle: Dict[str, Any],
    text: str,
    anorexia_threshold: float = 0.70,
    control_threshold: float = 0.35,
    min_words: int = 3
) -> Dict[str, Any]:
    config = runtime_bundle["config"]

    if config["type"] == "classic":
        result = predict_text(
            model=runtime_bundle["model"],
            text=text,
            anorexia_threshold=anorexia_threshold,
            control_threshold=control_threshold,
            min_words=min_words
        )
    elif config["type"] == "beto":
        result = predict_text_beto(
            classifier=runtime_bundle["model"],
            embedder=runtime_bundle["embedder"],
            text=text,
            anorexia_threshold=anorexia_threshold,
            control_threshold=control_threshold,
            min_words=min_words
        )
    elif config["type"] == "beto_llm_ensemble":
        raw = predict_text_beto_llm_ensemble(
            classifier=runtime_bundle["model"],
            embedder=runtime_bundle["embedder"],
            text=text,
            anorexia_threshold=anorexia_threshold,
            control_threshold=control_threshold,
            min_words=min_words,
            llm_callback=runtime_bundle["llm_callback"],
        )
        result = _normalize_combo_result(raw, config)
    elif config["type"] == "beto_llm_cascade":
        raw = predict_text_beto_llm_cascade(
            classifier=runtime_bundle["model"],
            embedder=runtime_bundle["embedder"],
            text=text,
            anorexia_threshold=anorexia_threshold,
            control_threshold=control_threshold,
            min_words=min_words,
            llm_callback=runtime_bundle["llm_callback"],
        )
        result = _normalize_combo_result(raw, config)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {config['type']}")

    result["model_key"] = config["key"]
    result["model_label"] = config["label"]
    return result


# Realiza predicciones sobre una columna de texto en un DataFrame.
# Enruta la predicción a la función de batch correspondiente según el modelo,
# o itera utilizando 'predict_single_with_runtime' si es un modelo híbrido.
def predict_dataframe_with_runtime(
    runtime_bundle: Dict[str, Any],
    df,
    text_column: str,
    anorexia_threshold: float = 0.70,
    control_threshold: float = 0.35,
    min_words: int = 3
):
    config = runtime_bundle["config"]

    if config["type"] == "classic":
        result_df = predict_dataframe(
            model=runtime_bundle["model"],
            df=df,
            text_column=text_column,
            anorexia_threshold=anorexia_threshold,
            control_threshold=control_threshold,
            min_words=min_words
        )
    elif config["type"] == "beto":
        result_df = predict_dataframe_beto(
            classifier=runtime_bundle["model"],
            embedder=runtime_bundle["embedder"],
            df=df,
            text_column=text_column,
            anorexia_threshold=anorexia_threshold,
            control_threshold=control_threshold,
            min_words=min_words
        )
    elif config["type"] in {"beto_llm_ensemble", "beto_llm_cascade"}:
        if text_column not in df.columns:
            raise ValueError(f"La columna '{text_column}' no existe en el archivo.")

        result_df = df.copy()

        predictions = result_df[text_column].apply(
            lambda x: predict_single_with_runtime(
                runtime_bundle=runtime_bundle,
                text=x,
                anorexia_threshold=anorexia_threshold,
                control_threshold=control_threshold,
                min_words=min_words
            )
        )

        result_df["cleaned_text"] = predictions.apply(lambda x: x["cleaned_text"])
        result_df["predicted_label"] = predictions.apply(lambda x: x["predicted_label"])
        result_df["probability_anorexia"] = predictions.apply(lambda x: x["probability_anorexia"])
        result_df["confidence"] = predictions.apply(lambda x: x["confidence"])
        result_df["message"] = predictions.apply(lambda x: x["message"])
        result_df["observations"] = predictions.apply(lambda x: x["observations"])
        result_df["word_count"] = predictions.apply(lambda x: x["word_count"])
        result_df["vocab_coverage"] = predictions.apply(lambda x: x["vocab_coverage"])
        result_df["decision_source"] = predictions.apply(lambda x: x.get("decision_source"))
        result_df["beto_probability"] = predictions.apply(lambda x: x.get("beto_probability"))
        result_df["llm_risk_score"] = predictions.apply(lambda x: x.get("llm_risk_score"))
        result_df["llm_label"] = predictions.apply(lambda x: x.get("llm_label"))
        result_df["method"] = predictions.apply(lambda x: x.get("method"))
    else:
        raise ValueError(f"Tipo de modelo no soportado: {config['type']}")

    result_df["model_key"] = config["key"]
    result_df["model_label"] = config["label"]
    return result_df