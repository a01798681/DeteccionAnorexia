# Author: Andrés Cabrera Alvarado - A01798681
# Author: Andrea Elizabeth Roman Varela - A01749760
# Author: Pablo Alonso Galván - A01748288
# Fecha de creación: 10/05/2026
# Archivo: backend/main.py
# Descripción general: Servidor backend construido con FastAPI que expone una API REST para la detección de trastornos alimenticios (anorexia) en textos. Permite:
#   - Clasificar textos individuales con un modelo seleccionado.
#   - Comparar predicciones de todos los modelos disponibles.
#   - Inspeccionar y clasificar archivos CSV/Excel de forma masiva.
#   - Consultar métricas de evaluación de cada modelo.
#   - Gestionar un léxico personalizado de términos de riesgo y seguridad.
import sys
from io import BytesIO
from pathlib import Path
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import json
from pathlib import Path

# Configuración de la ruta raíz del proyecto
# Se agrega al sys.path para que Python pueda encontrar los módulos en src/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.model_registry import get_available_models, get_default_model_key
    # get_available_models:   Devuelve la lista de modelos registrados con su metadata.
    # get_default_model_key:  Retorna la clave del modelo por defecto a utilizar.

from src.model_runtime import (
    load_runtime_bundle,                # Carga en memoria el pipeline completo de un modelo (vectorizador + clasificador).
    predict_single_with_runtime,        # Clasifica un texto individual usando un runtime previamente cargado.
    predict_dataframe_with_runtime,     # Clasifica todas las filas de un DataFrame usando un runtime.
)
from src.term_lexicon import load_custom_terms, save_custom_terms, get_term_sets
    # load_custom_terms:  Lee los términos personalizados guardados por el usuario.
    # save_custom_terms:  Persiste los términos personalizados del usuario en disco.
    # get_term_sets:      Combina términos base con los personalizados y los devuelve.

# Umbral de probabilidad para etiquetar un texto como "anorexia".
# Si la probabilidad predicha >= 0.70, se clasifica como positivo.
DEFAULT_ANOREXIA_THRESHOLD = 0.70

# Umbral de probabilidad para etiquetar un texto como "control".
# Si la probabilidad predicha <= 0.35, se clasifica como negativo.
DEFAULT_CONTROL_THRESHOLD = 0.35

# Número mínimo de palabras que debe tener un texto para ser clasificado.
# Textos con menos de 3 palabras se marcan como "no clasificable".
DEFAULT_MIN_WORDS = 3

# Lista de nombres de columna que se intentarán detectar automáticamente
# como la columna que contiene el texto a clasificar al subir un archivo. Se recorren en orden de prioridad.
TEXT_COLUMN_HINTS = [
    "tweet_text", "text", "texto", "contenido", "post",
    "mensaje", "comentario", "body", "caption"
]

# Directorio donde se almacenan los archivos JSON con métricas de evaluación de modelos
# (accuracy, precision, recall, F1, ROC-AUC).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Mapeo de clave de modelo → lista de posibles nombres de archivo con sus métricas.
# Se prueban en orden; se usa el primero que exista en disco.
MODEL_METRIC_CANDIDATES = {
    "beto_llm_ensemble": ["beto_llm_ensemble_metrics.json"],
    "beto_llm_cascade": ["beto_llm_cascade_metrics.json"],
    "beto_logreg": ["beto_metrics.json", "beto_logreg_metrics.json"],
    "hybrid_logreg": ["logistic_regression_hybrid_metrics.json", "hybrid_logreg_metrics.json"],
    "random_forest_svd": ["random_forest_svd_metrics.json"],
}

# Instancia principal de la aplicación FastAPI.
app = FastAPI(title="Detector de desórdenes alimenticios API")

# Configuración del middleware CORS para permitir que el frontend (Vite en localhost:5173) realice peticiones al backend sin ser bloqueado por el navegador.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictTextRequest(BaseModel):
    """Esquema de entrada para el endpoint /predict-text.
    Campos:
        text:               Texto libre a clasificar.
        model_key:          Clave identificadora del modelo a utilizar.
        anorexia_threshold: Umbral superior para clase "anorexia" (default 0.70).
        control_threshold:  Umbral inferior para clase "control"  (default 0.35).
        min_words:          Mínimo de palabras requerido para clasificar (default 3).
    """
    text: str
    model_key: str
    anorexia_threshold: float = DEFAULT_ANOREXIA_THRESHOLD
    control_threshold: float = DEFAULT_CONTROL_THRESHOLD
    min_words: int = DEFAULT_MIN_WORDS


class CompareModelsRequest(BaseModel):
    """Esquema de entrada para el endpoint /compare-models.
    Se ejecutan todos los modelos disponibles para comparar sus predicciones.
    """
    text: str
    anorexia_threshold: float = DEFAULT_ANOREXIA_THRESHOLD
    control_threshold: float = DEFAULT_CONTROL_THRESHOLD
    min_words: int = DEFAULT_MIN_WORDS

class CustomTermsRequest(BaseModel):
    """Esquema de entrada para el endpoint POST /custom-terms.
    Permite al usuario agregar términos personalizados a tres categorías
    del léxico utilizado por las reglas heurísticas del sistema:
        risk_terms_extra:           Términos adicionales de riesgo.
        positive_safe_terms_extra:  Términos positivos/seguros adicionales.
        negation_safe_terms_extra:  Términos de negación segura adicionales.
    """
    risk_terms_extra: list[str] = []
    positive_safe_terms_extra: list[str] = []
    negation_safe_terms_extra: list[str] = []


@lru_cache
def load_cached_runtime_bundle(model_key: str):
    """Carga el runtime bundle de un modelo y lo mantiene en caché (LRU).
    Evita recargar desde disco los archivos del modelo cada vez que se
    recibe una petición. La primera llamada carga el modelo; las siguientes
    devuelven el objeto en memoria.

    Args:
        model_key: Clave única del modelo (e.g. "beto_logreg").

    Returns:
        Un diccionario con el pipeline completo (vectorizador + clasificador).
    """
    return load_runtime_bundle(model_key)


def dataframe_to_csv_download(df: pd.DataFrame) -> bytes:
    """Convierte un DataFrame a bytes CSV codificados en UTF-8 con BOM.
    La codificación utf-8-sig asegura que Excel en Windows interprete
    correctamente los caracteres especiales (acentos, ñ, etc.).

    Args:
        df: DataFrame de pandas a convertir.

    Returns:
        Bytes del contenido CSV listos para ser enviados como descarga.
    """
    return df.to_csv(index=False).encode("utf-8-sig")


def _normalize_col_name(value: str) -> str:
    """Normaliza el nombre de una columna para comparación flexible.
    Elimina espacios, convierte a minúsculas y descarta caracteres
    no alfanuméricos. Esto permite hacer coincidir nombres como
    "Tweet Text", "tweet_text" y "TweetText".

    Args:
        value: Nombre original de la columna.

    Returns:
        Cadena normalizada (solo caracteres alfanuméricos en minúsculas).
    """
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def suggest_text_column(columns) -> str | None:
    """Sugiere automáticamente cuál columna contiene el texto a clasificar.
    Recorre la lista TEXT_COLUMN_HINTS y busca coincidencias en dos pasadas:
      1. Coincidencia exacta (normalizada) entre el nombre de columna y el hint.
      2. Coincidencia parcial: si el hint aparece como subcadena del nombre.

    Args:
        columns: Iterable con los nombres de las columnas del DataFrame.

    Returns:
        El nombre original de la columna sugerida, o None si no se encontró.
    """
    # Crear mapa {nombre_original: nombre_normalizado} para cada columna
    normalized_map = {col: _normalize_col_name(col) for col in columns}

    # Pasada 1: coincidencia exacta normalizada
    for hint in TEXT_COLUMN_HINTS:
        hint_norm = _normalize_col_name(hint)
        for original, normalized in normalized_map.items():
            if normalized == hint_norm:
                return original

    # Pasada 2: coincidencia parcial
    for hint in TEXT_COLUMN_HINTS:
        hint_norm = _normalize_col_name(hint)
        for original, normalized in normalized_map.items():
            if hint_norm in normalized:
                return original

    return None


def sanitize_uploaded_dataframe(df: pd.DataFrame, text_column: str):
    """Limpia un DataFrame eliminando filas con texto vacío o inválido.
    Filtra filas donde la columna de texto:
      - Sea NaN, None o vacía.
      - Contenga literales como "nan", "none" o "null".

    Args:
        df:          DataFrame original subido por el usuario.
        text_column: Nombre de la columna que contiene el texto.

    Returns:
        Tupla (df_limpio, filas_eliminadas):
            - df_limpio:         DataFrame filtrado con solo filas válidas.
            - filas_eliminadas:  Número entero de filas descartadas.

    Raises:
        ValueError: Si la columna indicada no existe en el DataFrame.
    """
    if text_column not in df.columns:
        raise ValueError(f"La columna '{text_column}' no existe en el archivo.")

    cleaned_df = df.copy()
    series = cleaned_df[text_column]

    # Máscara booleana: True si la fila tiene texto válido
    valid_mask = series.notna() & series.astype(str).str.strip().ne("")
    valid_mask &= ~series.astype(str).str.strip().str.lower().isin(
        ["nan", "none", "null"]
    )

    dropped_rows = int((~valid_mask).sum())
    cleaned_df = cleaned_df.loc[valid_mask].copy()
    cleaned_df[text_column] = cleaned_df[text_column].astype(str)

    return cleaned_df, dropped_rows

def inspect_uploaded_file(raw_bytes: bytes, file_name: str, sheet_name: str | None = None):
    """
    Inspecciona un archivo CSV o Excel y devuelve:
    - tipo de archivo
    - hojas disponibles (si aplica)
    - columnas
    - columna sugerida de texto
    - preview
    """
    if file_name.endswith(".csv"):
        df = pd.read_csv(BytesIO(raw_bytes))
        return {
            "kind": "csv",
            "sheet_names": [],
            "selected_sheet": None,
            "df": df,
        }

    # Procesamiento de archivos Excel
    if file_name.endswith(".xlsx"):
        excel_file = pd.ExcelFile(BytesIO(raw_bytes))
        sheet_names = excel_file.sheet_names

        if not sheet_names:
            raise ValueError("El archivo Excel no contiene hojas.")

        # Si la hoja solicitada no existe, se usa la primera hoja disponible
        selected_sheet = sheet_name if sheet_name in sheet_names else sheet_names[0]
        df = pd.read_excel(BytesIO(raw_bytes), sheet_name=selected_sheet)

        return {
            "kind": "xlsx",
            "sheet_names": sheet_names,
            "selected_sheet": selected_sheet,
            "df": df,
        }

    raise ValueError("Formato no soportado. Sube un archivo .csv o .xlsx")

def _safe_metric(value):
    """Convierte un valor de métrica a float redondeado de forma segura.
    Maneja valores None, no numéricos o con tipos inesperados sin lanzar
    excepciones, devolviendo None en esos casos.

    Args:
        value: Valor de la métrica (puede ser int, float, str o None).

    Returns:
        Float redondeado a 4 decimales, o None si el valor no es convertible.
    """
    try:
        if value is None:
            return None
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def load_metrics_for_model(model_key: str):
    """Carga las métricas de evaluación de un modelo desde su archivo JSON.
    Busca el archivo de métricas usando la lista de nombres candidatos
    definida en MODEL_METRIC_CANDIDATES. Retorna el primer archivo encontrado.

    Args:
        model_key: Clave del modelo (e.g. "beto_logreg").

    Returns:
        Diccionario con las métricas (accuracy, precision, recall, f1, roc_auc),
        o None si no se encontró ningún archivo de métricas.
    """
    candidate_files = MODEL_METRIC_CANDIDATES.get(model_key, [])

    for filename in candidate_files:
        path = RESULTS_DIR / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            return {
                "accuracy": _safe_metric(raw.get("accuracy")),
                "precision": _safe_metric(raw.get("precision")),
                "recall": _safe_metric(raw.get("recall")),
                "f1": _safe_metric(raw.get("f1")),
                "roc_auc": _safe_metric(raw.get("roc_auc")),
            }

    return None


def load_all_model_metrics():
    """Carga las métricas de evaluación de todos los modelos disponibles.
    Recorre todos los modelos que existen en disco y recopila sus métricas
    en un diccionario indexado por clave de modelo.

    Returns:
        Diccionario {model_key: métricas} donde métricas es un dict con
        accuracy, precision, recall, f1 y roc_auc (o None si no hay archivo).
    """
    models = [m for m in get_available_models() if m["exists"]]
    output = {}

    for model in models:
        output[model["key"]] = load_metrics_for_model(model["key"])

    return output

# Endpoint de salud (health check).
# Devuelve un mensaje simple para verificar que la API está activa.
@app.get("/")
def home():
    return {"message": "API funcionando correctamente"}


# Devuelve la lista de modelos disponibles (solo los que tienen archivos en disco)
# junto con la clave del modelo por defecto.
@app.get("/models")
def get_models():
    models = [m for m in get_available_models() if m["exists"]]

    return {
        "default_model_key": get_default_model_key(),
        "models": models,
    }

# Devuelve un diccionario con las métricas de evaluación (accuracy, precision, recall, F1, ROC-AUC) para cada modelo disponible.
@app.get("/model-metrics")
def get_model_metrics():
    return load_all_model_metrics()

# Devuelve los términos personalizados del usuario junto con los conteos
# totales activos (base + personalizados) de cada categoría.
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


# Actualiza los términos personalizados del usuario en disco.
# Recibe listas de términos extra para cada categoría y devuelve
# el estado actualizado con los conteos combinados.
@app.post("/custom-terms")
def update_custom_terms(request: CustomTermsRequest):
    save_custom_terms(
        risk_terms_extra=request.risk_terms_extra,
        positive_safe_terms_extra=request.positive_safe_terms_extra,
        negation_safe_terms_extra=request.negation_safe_terms_extra,
    )

    # Recargar para devolver el estado actualizado
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

# Clasifica un texto individual con el modelo especificado.
# Devuelve la etiqueta predicha, la probabilidad y metadatos adicionales.
@app.post("/predict-text")
def predict_text(request: PredictTextRequest):
    # Cargar el modelo desde caché
    runtime = load_cached_runtime_bundle(request.model_key)

    # Ejecutar la predicción y devolver el resultado directamente
    return predict_single_with_runtime(
        runtime_bundle=runtime,
        text=request.text,
        anorexia_threshold=request.anorexia_threshold,
        control_threshold=request.control_threshold,
        min_words=request.min_words,
    )

# Clasifica el mismo texto con TODOS los modelos disponibles.
# Útil para comparar cómo distintos modelos clasifican un mismo texto.
# Devuelve una lista de resultados, uno por cada modelo.
@app.post("/compare-models")
def compare_models(request: CompareModelsRequest):
    available_models = get_available_models()
    available_now = [m for m in available_models if m["exists"]]

    results = []

    for model_config in available_now:
        # Cargar el runtime de cada modelo
        runtime = load_cached_runtime_bundle(model_config["key"])

        # Clasificar el texto con este modelo
        result = predict_single_with_runtime(
            runtime_bundle=runtime,
            text=request.text,
            anorexia_threshold=request.anorexia_threshold,
            control_threshold=request.control_threshold,
            min_words=request.min_words,
        )

        # Agregar metadata del modelo al resultado para identificarlo
        result["short_label"] = model_config["short_label"]
        result["family"] = model_config["family"]
        results.append(result)

    return results

# Inspecciona un archivo CSV o Excel subido sin ejecutar predicciones.
# Devuelve las columnas, una sugerencia de columna de texto, un preview
# de las primeras 10 filas, y la lista de hojas.
# El frontend usa esta info para que el usuario confirme la columna antes
# de enviar el archivo a /predict-file.
@app.post("/inspect-file")
async def inspect_file(
    file: UploadFile = File(...),
    sheet_name: str | None = Form(None),
):
    file_name = file.filename.lower()
    raw_bytes = await file.read()

    try:
        # Inspeccionar la estructura del archivo (tipo, hojas, columnas)
        file_info = inspect_uploaded_file(raw_bytes, file_name, sheet_name)
        df = file_info["df"]

        if len(df.columns) == 0:
            return {"error": "El archivo no contiene columnas."}

        # Intentar detectar automáticamente la columna de texto
        suggested_text_column = suggest_text_column(df.columns)

        # Generar preview de las primeras 10 filas (reemplazar NaN por None para JSON)
        preview_df = df.head(10).copy()
        import numpy as np
        preview_df = preview_df.replace({np.nan: None})

        return {
            "kind": file_info["kind"],
            "sheet_names": file_info["sheet_names"],
            "selected_sheet": file_info["selected_sheet"],
            "columns": list(df.columns),
            "suggested_text_column": suggested_text_column,
            "preview": preview_df.to_dict(orient="records"),
            "total_rows": len(df),
        }

    except Exception as e:
        return {"error": str(e)}

# Endpoint principal para clasificación masiva de archivos.
# Recibe un archivo CSV/Excel, lo limpia, clasifica cada fila con el modelo
# indicado y devuelve:
#   - Resumen estadístico (cantidad y porcentaje por clase).
#   - Resultados fila por fila.
#   - CSV descargable con los resultados.
@app.post("/predict-file")
async def predict_file(
    file: UploadFile = File(...),                  
    model_key: str = Form(...),                  
    text_column: str | None = Form(None),                       
    sheet_name: str | None = Form(None),                    
    anorexia_threshold: float = Form(DEFAULT_ANOREXIA_THRESHOLD),  
    control_threshold: float = Form(DEFAULT_CONTROL_THRESHOLD),   
    min_words: int = Form(DEFAULT_MIN_WORDS),                    
):
    file_name = file.filename.lower()
    raw_bytes = await file.read()

    # Paso 1: Leer y parsear el archivo subido
    try:
        file_info = inspect_uploaded_file(raw_bytes, file_name, sheet_name)
        df = file_info["df"]
        selected_sheet = file_info["selected_sheet"]
    except Exception as e:
        return {"error": str(e)}

    # Paso 2: Validar que el archivo tiene columnas
    if len(df.columns) == 0:
        return {"error": "El archivo no contiene columnas."}

    # Paso 3: Determinar la columna de texto (auto-detectar si no fue proporcionada)
    if text_column is None or text_column == "":
        text_column = suggest_text_column(df.columns)

    if text_column is None:
        return {
            "error": "No se pudo detectar automáticamente la columna de texto.",
            "columns": list(df.columns),
        }

    # Paso 4: Limpiar el DataFrame (eliminar filas sin texto válido)
    df_ready, dropped_rows = sanitize_uploaded_dataframe(df, text_column)

    if len(df_ready) == 0:
        return {"error": "No hay filas válidas para clasificar."}

    # Paso 5: Cargar el modelo y ejecutar las predicciones sobre el DataFrame limpio
    runtime = load_cached_runtime_bundle(model_key)

    result_df = predict_dataframe_with_runtime(
        runtime_bundle=runtime,
        df=df_ready,
        text_column=text_column,
        anorexia_threshold=anorexia_threshold,
        control_threshold=control_threshold,
        min_words=min_words,
    )

    # Paso 6: Generar resumen estadístico (conteo y porcentaje por clase predicha)
    summary = result_df["predicted_label"].value_counts(dropna=False).reset_index()
    summary.columns = ["clase_predicha", "cantidad"]
    summary["porcentaje"] = (
        summary["cantidad"] / summary["cantidad"].sum() * 100
    ).round(2)

    # Paso 7: Preparar los datos de respuesta (reemplazar NaN por None para JSON válido)
    import numpy as np
    result_df = result_df.replace({np.nan: None})

    # Generar CSV descargable con los resultados codificado en UTF-8 con BOM
    csv_data = result_df.to_csv(index=False).encode("utf-8-sig").decode("utf-8-sig")

    # Paso 8: Devolver respuesta completa al frontend
    return {
        "text_column": text_column,              
        "sheet_name": selected_sheet,                   
        "total_rows": len(df),                            
        "valid_rows": len(df_ready),                        
        "summary": summary.to_dict(orient="records"),           
        "results": result_df.to_dict(orient="records"),
        "csv": csv_data,
        "filename": f"resultados_{model_key}.csv",
    }