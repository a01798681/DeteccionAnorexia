from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import joblib

from .beto_embeddings import BETOEmbedder
from .model_registry import get_model_config
from .predict import predict_text, predict_dataframe
from .predict_beto import predict_text_beto, predict_dataframe_beto


def load_runtime_bundle(model_key: str):
    config = get_model_config(model_key)

    if not config["path"].exists():
        raise FileNotFoundError(f"No se encontró el modelo: {config['path']}")

    if config["type"] == "classic":
        model = joblib.load(config["path"])
        return {
            "config": config,
            "model": model,
            "embedder": None
        }

    if config["type"] == "beto":
        classifier = joblib.load(config["path"])
        embedder = BETOEmbedder()
        return {
            "config": config,
            "model": classifier,
            "embedder": embedder
        }

    raise ValueError(f"Tipo de modelo no soportado: {config['type']}")


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
    else:
        raise ValueError(f"Tipo de modelo no soportado: {config['type']}")

    result["model_key"] = config["key"]
    result["model_label"] = config["label"]
    return result


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
    else:
        raise ValueError(f"Tipo de modelo no soportado: {config['type']}")

    result_df["model_key"] = config["key"]
    result_df["model_label"] = config["label"]
    return result_df