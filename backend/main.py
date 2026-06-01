# backend/main.py

import sys
from io import BytesIO
from pathlib import Path
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.model_registry import get_available_models, get_default_model_key
from src.model_runtime import (
    load_runtime_bundle,
    predict_single_with_runtime,
    predict_dataframe_with_runtime,
)
from src.term_lexicon import load_custom_terms, save_custom_terms, get_term_sets


DEFAULT_ANOREXIA_THRESHOLD = 0.70
DEFAULT_CONTROL_THRESHOLD = 0.35
DEFAULT_MIN_WORDS = 3

TEXT_COLUMN_HINTS = [
    "tweet_text", "text", "texto", "contenido", "post",
    "mensaje", "comentario", "body", "caption"
]


app = FastAPI(title="Detector de desórdenes alimenticios API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictTextRequest(BaseModel):
    text: str
    model_key: str
    anorexia_threshold: float = DEFAULT_ANOREXIA_THRESHOLD
    control_threshold: float = DEFAULT_CONTROL_THRESHOLD
    min_words: int = DEFAULT_MIN_WORDS


class CompareModelsRequest(BaseModel):
    text: str
    anorexia_threshold: float = DEFAULT_ANOREXIA_THRESHOLD
    control_threshold: float = DEFAULT_CONTROL_THRESHOLD
    min_words: int = DEFAULT_MIN_WORDS

class CustomTermsRequest(BaseModel):
    risk_terms_extra: list[str] = []
    positive_safe_terms_extra: list[str] = []
    negation_safe_terms_extra: list[str] = []


@lru_cache
def load_cached_runtime_bundle(model_key: str):
    return load_runtime_bundle(model_key)


def dataframe_to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def _normalize_col_name(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def suggest_text_column(columns) -> str | None:
    normalized_map = {col: _normalize_col_name(col) for col in columns}

    for hint in TEXT_COLUMN_HINTS:
        hint_norm = _normalize_col_name(hint)
        for original, normalized in normalized_map.items():
            if normalized == hint_norm:
                return original

    for hint in TEXT_COLUMN_HINTS:
        hint_norm = _normalize_col_name(hint)
        for original, normalized in normalized_map.items():
            if hint_norm in normalized:
                return original

    return None


def sanitize_uploaded_dataframe(df: pd.DataFrame, text_column: str):
    if text_column not in df.columns:
        raise ValueError(f"La columna '{text_column}' no existe en el archivo.")

    cleaned_df = df.copy()
    series = cleaned_df[text_column]

    valid_mask = series.notna() & series.astype(str).str.strip().ne("")
    valid_mask &= ~series.astype(str).str.strip().str.lower().isin(
        ["nan", "none", "null"]
    )

    dropped_rows = int((~valid_mask).sum())
    cleaned_df = cleaned_df.loc[valid_mask].copy()
    cleaned_df[text_column] = cleaned_df[text_column].astype(str)

    return cleaned_df, dropped_rows


@app.get("/")
def home():
    return {"message": "API funcionando correctamente"}


@app.get("/models")
def get_models():
    models = [m for m in get_available_models() if m["exists"]]

    return {
        "default_model_key": get_default_model_key(),
        "models": models,
    }

@app.get("/custom-terms")
def get_custom_terms():
    custom = load_custom_terms()
    merged = get_term_sets()

    return {
        "custom_terms": custom,
        "active_counts": {
            "risk_terms": len(merged["risk_terms"]),
            "positive_safe_terms": len(merged["positive_safe_terms"]),
            "negation_safe_terms": len(merged["negation_safe_terms"]),
        },
    }


@app.post("/custom-terms")
def update_custom_terms(request: CustomTermsRequest):
    save_custom_terms(
        risk_terms_extra=request.risk_terms_extra,
        positive_safe_terms_extra=request.positive_safe_terms_extra,
        negation_safe_terms_extra=request.negation_safe_terms_extra,
    )

    custom = load_custom_terms()
    merged = get_term_sets()

    return {
        "message": "Términos guardados correctamente.",
        "custom_terms": custom,
        "active_counts": {
            "risk_terms": len(merged["risk_terms"]),
            "positive_safe_terms": len(merged["positive_safe_terms"]),
            "negation_safe_terms": len(merged["negation_safe_terms"]),
        },
    }

@app.post("/predict-text")
def predict_text(request: PredictTextRequest):
    runtime = load_cached_runtime_bundle(request.model_key)

    return predict_single_with_runtime(
        runtime_bundle=runtime,
        text=request.text,
        anorexia_threshold=request.anorexia_threshold,
        control_threshold=request.control_threshold,
        min_words=request.min_words,
    )

@app.post("/compare-models")
def compare_models(request: CompareModelsRequest):
    available_models = get_available_models()
    available_now = [m for m in available_models if m["exists"]]

    results = []

    for model_config in available_now:
        runtime = load_cached_runtime_bundle(model_config["key"])

        result = predict_single_with_runtime(
            runtime_bundle=runtime,
            text=request.text,
            anorexia_threshold=request.anorexia_threshold,
            control_threshold=request.control_threshold,
            min_words=request.min_words,
        )

        result["short_label"] = model_config["short_label"]
        result["family"] = model_config["family"]
        results.append(result)

    return results


@app.post("/predict-file")
async def predict_file(
    file: UploadFile = File(...),
    model_key: str = Form(...),
    text_column: str | None = Form(None),
    anorexia_threshold: float = Form(DEFAULT_ANOREXIA_THRESHOLD),
    control_threshold: float = Form(DEFAULT_CONTROL_THRESHOLD),
    min_words: int = Form(DEFAULT_MIN_WORDS),
):
    file_name = file.filename.lower()
    raw_bytes = await file.read()

    if file_name.endswith(".csv"):
        df = pd.read_csv(BytesIO(raw_bytes))
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(BytesIO(raw_bytes))
    else:
        return {"error": "Formato no soportado. Sube un archivo .csv o .xlsx"}

    if len(df.columns) == 0:
        return {"error": "El archivo no contiene columnas."}

    if text_column is None or text_column == "":
        text_column = suggest_text_column(df.columns)

    if text_column is None:
        return {
            "error": "No se pudo detectar automáticamente la columna de texto.",
            "columns": list(df.columns),
        }

    df_ready, dropped_rows = sanitize_uploaded_dataframe(df, text_column)

    if len(df_ready) == 0:
        return {"error": "No hay filas válidas para clasificar."}

    runtime = load_cached_runtime_bundle(model_key)

    result_df = predict_dataframe_with_runtime(
        runtime_bundle=runtime,
        df=df_ready,
        text_column=text_column,
        anorexia_threshold=anorexia_threshold,
        control_threshold=control_threshold,
        min_words=min_words,
    )

    summary = result_df["predicted_label"].value_counts(dropna=False).reset_index()
    summary.columns = ["clase_predicha", "cantidad"]
    summary["porcentaje"] = (
        summary["cantidad"] / summary["cantidad"].sum() * 100
    ).round(2)

    import numpy as np
    result_df = result_df.replace({np.nan: None})

    csv_data = result_df.to_csv(index=False).encode("utf-8-sig").decode("utf-8-sig")

    return {
        "text_column": text_column,
        "total_rows": len(df),
        "valid_rows": len(df_ready),
        "dropped_rows": dropped_rows,
        "summary": summary.to_dict(orient="records"),
        "results": result_df.to_dict(orient="records"),
        "csv": csv_data,
        "filename": f"resultados_{model_key}.csv",
    }