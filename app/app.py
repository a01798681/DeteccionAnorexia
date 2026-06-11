# Author: Andrés Cabrera Alvarado - A01798681
# Author: Andrea Elizabeth Roman Varela - A01749760
# Author: Pablo Alonso Galván - A01748288
# Fecha de creación: 10/05/2026
# Archivo: app/app.py
# Descripción general: Interfaz gráfica construida con Streamlit que permite al usuario
#   interactuar con los modelos de detección de trastornos alimenticios. Ofrece:
#   - Clasificación individual de textos con el modelo seleccionado.
#   - Clasificación masiva de archivos CSV/Excel con resumen estadístico y descarga de resultados.
#   - Comparación lado a lado de todos los modelos disponibles sobre un mismo texto.
#   - Panel lateral para configurar umbrales, seleccionar modelo y gestionar un léxico personalizado.
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

# Configuración de la ruta raíz del proyecto
# Se agrega al sys.path para que Python pueda encontrar los módulos en src/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.model_registry import get_available_models, get_default_model_key
    # get_available_models:   Devuelve la lista de modelos registrados con su metadata.
    # get_default_model_key:  Retorna la clave del modelo por defecto a utilizar.

from src.model_runtime import (
    load_runtime_bundle,
    predict_single_with_runtime,
    predict_dataframe_with_runtime,
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
# como la columna que contiene el texto a clasificar al subir un archivo.
# Se recorren en orden de prioridad.
TEXT_COLUMN_HINTS = [
    "tweet_text", "text", "texto", "contenido", "post",
    "mensaje", "comentario", "body", "caption"
]


def inject_custom_css():
    """Inyecta estilos CSS personalizados en la aplicación Streamlit.
    Define clases para el layout principal, hero box, chips de modelos,
    tarjetas de predicción con colores semánticos (rojo=anorexia,
    verde=control, amarillo=incierto), y estilos tipográficos auxiliares.
    Se ejecuta una sola vez al inicio de la app.
    """
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1300px;
        }

        .hero-box {
            padding: 1.2rem 1.4rem;
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(21,128,61,0.16), rgba(8,47,73,0.16));
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .hero-subtitle {
            font-size: 0.95rem;
            opacity: 0.90;
        }

        .soft-chip {
            display: inline-block;
            padding: 0.30rem 0.60rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
            font-size: 0.82rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }

        .model-card {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1rem;
            background: rgba(255,255,255,0.02);
            min-height: 330px;
        }

        .model-card-title {
            font-size: 1.35rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.25rem;
        }

        .model-card-subtitle {
            font-size: 0.82rem;
            opacity: 0.75;
            margin-bottom: 0.75rem;
        }

        .prediction-pill {
            display: inline-block;
            padding: 0.45rem 0.75rem;
            border-radius: 12px;
            font-weight: 700;
            margin-bottom: 0.90rem;
        }

        .pred-anorexia {
            background: rgba(127,29,29,0.45);
            color: #fecaca;
            border: 1px solid rgba(248,113,113,0.25);
        }

        .pred-control {
            background: rgba(20,83,45,0.50);
            color: #bbf7d0;
            border: 1px solid rgba(74,222,128,0.25);
        }

        .pred-incierto {
            background: rgba(133,77,14,0.40);
            color: #fde68a;
            border: 1px solid rgba(250,204,21,0.25);
        }

        .mini-note {
            font-size: 0.88rem;
            opacity: 0.92;
            margin-top: 0.45rem;
        }

        .section-caption {
            font-size: 0.92rem;
            opacity: 0.82;
            margin-bottom: 0.5rem;
        }

        .sidebar-separator {
            margin: 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def inspect_uploaded_file(uploaded_file) -> dict:
    """Inspecciona un archivo subido por el usuario (CSV o Excel).
    Lee los bytes crudos del archivo y determina su tipo. Para CSV,
    parsea el DataFrame inmediatamente. Para Excel, extrae la lista
    de hojas disponibles sin parsear ninguna (se hace después con
    load_dataframe_from_upload).

    Args:
        uploaded_file: Objeto UploadedFile de Streamlit con el archivo subido.

    Returns:
        Diccionario con:
            - kind:        Tipo de archivo ("csv" o "xlsx").
            - raw_bytes:   Bytes crudos del archivo para re-lectura posterior.
            - sheet_names: Lista de nombres de hojas (solo para Excel, None para CSV).
            - df:          DataFrame parseado (solo para CSV, None para Excel).

    Raises:
        ValueError: Si el formato del archivo no es .csv ni .xlsx.
    """
    file_name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.getvalue()

    if file_name.endswith(".csv"):
        df = pd.read_csv(BytesIO(raw_bytes))
        return {
            "kind": "csv",
            "raw_bytes": raw_bytes,
            "sheet_names": None,
            "df": df
        }

    if file_name.endswith(".xlsx"):
        excel_file = pd.ExcelFile(BytesIO(raw_bytes))
        return {
            "kind": "xlsx",
            "raw_bytes": raw_bytes,
            "sheet_names": excel_file.sheet_names,
            "df": None
        }

    raise ValueError("Formato no soportado. Sube un archivo .csv o .xlsx")


def load_dataframe_from_upload(file_info: dict, sheet_name=None) -> pd.DataFrame:
    """Carga un DataFrame a partir del resultado de inspect_uploaded_file.
    Para CSV devuelve una copia del DataFrame ya parseado. Para Excel,
    lee la hoja especificada (o la primera si no se indica).

    Args:
        file_info:   Diccionario devuelto por inspect_uploaded_file.
        sheet_name:  Nombre de la hoja a leer. Si es None,
                     pandas usa la primera hoja por defecto.

    Returns:
        DataFrame de pandas con los datos del archivo/hoja seleccionada.
    """
    if file_info["kind"] == "csv":
        return file_info["df"].copy()

    return pd.read_excel(BytesIO(file_info["raw_bytes"]), sheet_name=sheet_name)


def sanitize_uploaded_dataframe(df: pd.DataFrame, text_column: str) -> tuple[pd.DataFrame, int]:
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
    valid_mask &= ~series.astype(str).str.strip().str.lower().isin(["nan", "none", "null"])

    dropped_rows = int((~valid_mask).sum())

    cleaned_df = cleaned_df.loc[valid_mask].copy()
    cleaned_df[text_column] = cleaned_df[text_column].astype(str)

    return cleaned_df, dropped_rows


@st.cache_resource
def load_cached_runtime_bundle(model_key: str):
    """Carga el runtime bundle de un modelo y lo mantiene en caché de Streamlit.
    Usa @st.cache_resource para evitar recargar desde disco los archivos del
    modelo en cada re-ejecución del script. La primera llamada carga el modelo;
    las siguientes devuelven el objeto en memoria.

    Args:
        model_key: Clave única del modelo (e.g. "beto_logreg").

    Returns:
        Un diccionario con el pipeline completo (vectorizador + clasificador).
    """
    return load_runtime_bundle(model_key)


def prediction_css_class(label: str) -> str:
    """Devuelve la clase CSS correspondiente a una etiqueta de predicción.
    Mapea la etiqueta textual a una clase CSS que define el color semántico
    del pill de predicción en la interfaz.

    Args:
        label: Etiqueta predicha ("anorexia", "control" o cualquier otra).

    Returns:
        Nombre de la clase CSS: "pred-anorexia", "pred-control" o "pred-incierto".
    """
    if label == "anorexia":
        return "pred-anorexia"
    if label == "control":
        return "pred-control"
    return "pred-incierto"


def render_result_card(result: dict, compact: bool = False):
    """Renderiza una tarjeta visual con los resultados de una predicción.
    Muestra el pill de predicción con color semántico, métricas clave
    (confianza, probabilidad de anorexia, conteo de palabras, cobertura
    de vocabulario), el mensaje y las observaciones del modelo.

    Args:
        result:  Diccionario con los resultados de predict_single_with_runtime.
        compact: Si es True, muestra el texto procesado dentro de un expander
                 colapsable. Si es False, lo muestra directamente como sección.
    """
    # Pill de predicción con color semántico según la etiqueta
    css_class = prediction_css_class(result["predicted_label"])

    st.markdown(
        f"<div class='prediction-pill {css_class}'>Predicción: {result['predicted_label']}</div>",
        unsafe_allow_html=True
    )

    # Métricas principales: confianza y probabilidad de anorexia
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confianza", result["confidence"])
    with col2:
        prob = result["probability_anorexia"]
        st.metric("Prob. anorexia", "N/A" if prob is None else f"{prob:.4f}")

    # Métricas adicionales: conteo de palabras y cobertura de vocabulario
    info_cols = st.columns(2)
    with info_cols[0]:
        st.metric("Palabras", result["word_count"])
    with info_cols[1]:
        coverage = result["vocab_coverage"]
        st.metric("Cobertura", "N/A" if coverage is None else f"{coverage:.2f}")

    # Mensaje interpretativo y observaciones generadas por el modelo
    st.markdown(f"**Mensaje:** {result['message']}")
    st.markdown(f"**Observaciones:** {result['observations']}")

    # Texto procesado (limpiado): compacto en expander o expandido
    if compact:
        with st.expander("Texto procesado"):
            st.code(result["cleaned_text"])
    else:
        st.subheader("Texto procesado")
        st.code(result["cleaned_text"])


def render_summary_table(compare_results: list[dict]):
    """Renderiza una tabla resumen y un gráfico de barras para la comparación entre modelos.
    Construye un DataFrame con las columnas clave de cada resultado (modelo,
    predicción, probabilidad, confianza, palabras, observaciones) y lo muestra
    como tabla interactiva. Además genera un gráfico de barras con la probabilidad
    de anorexia por modelo.

    Args:
        compare_results: Lista de diccionarios con los resultados de cada modelo.
    """
    # Construir DataFrame resumen con las columnas relevantes
    summary_df = pd.DataFrame([
        {
            "modelo": r["model_label"],
            "predicción": r["predicted_label"],
            "probabilidad_anorexia": r["probability_anorexia"],
            "confianza": r["confidence"],
            "palabras": r["word_count"],
            "observaciones": r["observations"]
        }
        for r in compare_results
    ])

    # Formatear la columna de probabilidad a 4 decimales para visualización
    display_df = summary_df.copy()
    if "probabilidad_anorexia" in display_df.columns:
        display_df["probabilidad_anorexia"] = display_df["probabilidad_anorexia"].apply(
            lambda x: None if x is None else round(float(x), 4)
        )

    # Mostrar tabla interactiva con los resultados
    st.dataframe(display_df, use_container_width=True)

    # Gráfico de barras: probabilidad de anorexia por modelo
    chart_df = summary_df.copy()
    chart_df["probabilidad_anorexia"] = chart_df["probabilidad_anorexia"].fillna(0.0)
    st.bar_chart(chart_df.set_index("modelo")["probabilidad_anorexia"])

# Configuración de la página: título, ícono y layout ancho.
# Debe ser la primera llamada a Streamlit en el script.
st.set_page_config(
    page_title="Detector de desórdenes alimenticios",
    page_icon="🧠",
    layout="wide"
)

# Inyectar los estilos CSS personalizados definidos en inject_custom_css()
inject_custom_css()

# Título principal y descripción de la aplicación
st.title("Detector de desórdenes alimenticios")
st.write(
    "Esta herramienta permite clasificar textos individuales o archivos completos "
    "para estimar si se detectan señales asociadas a anorexia."
)

# Streamlit re-ejecuta el script completo en cada interacción del usuario.
# Se usa session_state para preservar valores entre re-ejecuciones.
# manual_mode: controla si el usuario ajusta umbrales manualmente o usa los defaults
if "manual_mode" not in st.session_state:
    st.session_state.manual_mode = False

# anorexia_threshold: umbral superior para clase "anorexia" (configurable en modo manual)
if "anorexia_threshold" not in st.session_state:
    st.session_state.anorexia_threshold = DEFAULT_ANOREXIA_THRESHOLD

# control_threshold: umbral inferior para clase "control" (configurable en modo manual)
if "control_threshold" not in st.session_state:
    st.session_state.control_threshold = DEFAULT_CONTROL_THRESHOLD

# min_words: mínimo de palabras requerido (configurable en modo manual)
if "min_words" not in st.session_state:
    st.session_state.min_words = DEFAULT_MIN_WORDS

# Obtener la lista de modelos registrados y filtrar solo los que existen en disco.
# Si no hay ninguno, mostrar error y detener la aplicación.
available_models = get_available_models()
available_now = [m for m in available_models if m["exists"]]

if not available_now:
    st.error(
        "No se encontró ningún modelo disponible en la carpeta `results/`.\n\n"
        "Entrena primero alguno de estos modelos:\n"
        "- python main.py\n"
        "- python -m src.beto_train"
    )
    st.stop()

with st.sidebar:
    st.header("Configuración")

    # Selector de modelo activo
    # Construir las opciones del selectbox a partir de los modelos disponibles.
    # Se pre-selecciona el modelo por defecto definido en model_registry.
    model_labels = [m["label"] for m in available_now]
    default_key = get_default_model_key()
    default_index = 0

    if default_key is not None:
        for i, m in enumerate(available_now):
            if m["key"] == default_key:
                default_index = i
                break

    selected_label = st.selectbox(
        "Modelo activo",
        options=model_labels,
        index=default_index
    )
    selected_model_config = next(m for m in available_now if m["label"] == selected_label)

    # Mostrar metadata del modelo seleccionado (tipo, familia, descripción)
    st.caption(f"**Tipo:** {selected_model_config['type']}")
    st.caption(f"**Familia:** {selected_model_config['family']}")
    st.caption(selected_model_config["description"])

    # Cargar los términos personalizados actuales del usuario
    current_terms = load_custom_terms()

    # Toggle de modo manual
    # Si está activado, el usuario puede ajustar umbrales con sliders.
    # Si está desactivado, se usan los valores por defecto.
    manual_mode = st.toggle(
        "Usar configuración manual",
        value=st.session_state.manual_mode
    )
    st.session_state.manual_mode = manual_mode

    if not st.session_state.manual_mode:
        # Modo automático: usar valores por defecto y mostrar info
        anorexia_threshold = DEFAULT_ANOREXIA_THRESHOLD
        control_threshold = DEFAULT_CONTROL_THRESHOLD
        min_words = DEFAULT_MIN_WORDS

        st.info(
            f"Modo automático activo\n\n"
            f"Umbral anorexia = {anorexia_threshold:.2f}\n\n"
            f"Umbral control = {control_threshold:.2f}\n\n"
            f"Mínimo de palabras = {min_words}"
        )
    else:
        # Modo manual: sliders para ajustar cada parámetro individualmente
        st.session_state.anorexia_threshold = st.slider(
            "Umbral anorexia",
            min_value=0.50,
            max_value=0.95,
            value=float(st.session_state.anorexia_threshold),
            step=0.05
        )

        st.session_state.control_threshold = st.slider(
            "Umbral control",
            min_value=0.05,
            max_value=0.50,
            value=float(st.session_state.control_threshold),
            step=0.05
        )

        st.session_state.min_words = st.slider(
            "Mínimo de palabras",
            min_value=1,
            max_value=10,
            value=int(st.session_state.min_words),
            step=1
        )

        anorexia_threshold = st.session_state.anorexia_threshold
        control_threshold = st.session_state.control_threshold
        min_words = st.session_state.min_words

    # Sección de léxico personalizado
    # Permite al usuario agregar términos de riesgo, seguros y de negación
    # que complementan los términos base del sistema.
    st.divider()
    st.subheader("Jerga / palabras nuevas")

    st.caption(
        "Aquí puedes agregar términos nuevos que aparezcan en redes sociales. "
        "Se guardan en un archivo JSON y se aplican a las reglas y features manuales."
    )

    # Área de texto para términos de riesgo
    risk_terms_text = st.text_area(
        "Términos de riesgo extra (uno por línea)",
        value="\n".join(current_terms["risk_terms_extra"]),
        height=140
    )

    # Área de texto para términos seguros/positivos
    positive_safe_text = st.text_area(
        "Términos seguros extra (uno por línea)",
        value="\n".join(current_terms["positive_safe_terms_extra"]),
        height=100
    )

    # Área de texto para términos de negación segura
    negation_safe_text = st.text_area(
        "Términos de negación segura extra (uno por línea)",
        value="\n".join(current_terms["negation_safe_terms_extra"]),
        height=100
    )

    # Botones para guardar y recargar los términos personalizados
    col_save, col_reload = st.columns(2)
    with col_save:
        if st.button("Guardar términos"):
            save_custom_terms(
                risk_terms_extra=risk_terms_text.splitlines(),
                positive_safe_terms_extra=positive_safe_text.splitlines(),
                negation_safe_terms_extra=negation_safe_text.splitlines()
            )
            st.success("Términos guardados correctamente.")
            st.rerun()

    with col_reload:
        if st.button("Recargar términos"):
            st.rerun()

    # Mostrar conteo de términos activos (base + personalizados) por categoría
    merged_terms = get_term_sets()
    st.caption(
        f"Términos activos: riesgo={len(merged_terms['risk_terms'])}, "
        f"seguros={len(merged_terms['positive_safe_terms'])}, "
        f"negación segura={len(merged_terms['negation_safe_terms'])}"
    )

# Cargar el runtime del modelo seleccionado (desde caché si ya fue cargado)
active_runtime = load_cached_runtime_bundle(selected_model_config["key"])

# Hero box: muestra el modelo cargado y una breve descripción de las funcionalidades
st.markdown(
    f"""
    <div class="hero-box">
        <div class="hero-title">Modelo cargado: {selected_model_config['label']}</div>
        <div class="hero-subtitle">
            Puedes clasificar textos individuales, archivos y comparar el mismo texto entre todos los modelos disponibles.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Chips: etiquetas visuales con el nombre corto y familia de cada modelo disponible
chips_html = "".join(
    [f"<span class='soft-chip'>{m['short_label']} · {m['family']}</span>" for m in available_now]
)
st.markdown(chips_html, unsafe_allow_html=True)

# Se definen tres pestañas:
#   1. Clasificación individual: evaluar un texto con el modelo activo.
#   2. Clasificación por archivo: subir CSV/Excel y clasificar masivamente.
#   3. Comparación entre modelos: evaluar un texto con todos los modelos.
tab1, tab2, tab3 = st.tabs([
    "Clasificación individual",
    "Clasificación por archivo",
    "Comparación entre modelos"
])

# CLASIFICACIÓN INDIVIDUAL
# Permite al usuario escribir un texto libre y clasificarlo con el modelo activo.
# Muestra el resultado completo con métricas, mensaje y observaciones.
with tab1:
    st.subheader("Clasificación individual")
    st.markdown("<div class='section-caption'>Evalúa un texto con el modelo actualmente seleccionado.</div>", unsafe_allow_html=True)

    user_text = st.text_area(
        "Escribe el texto a evaluar",
        height=180,
        placeholder="Ejemplo: hoy no quiero comer nada porque me siento gorda..."
    )

    if st.button("Clasificar texto"):
        if not user_text.strip():
            st.warning("Por favor escribe un texto antes de clasificar.")
        else:
            # Ejecutar la predicción con el runtime activo y los umbrales configurados
            result = predict_single_with_runtime(
                runtime_bundle=active_runtime,
                text=user_text,
                anorexia_threshold=anorexia_threshold,
                control_threshold=control_threshold,
                min_words=min_words
            )

            st.subheader("Resultado")
            st.write(f"**Modelo usado:** {result['model_label']}")
            render_result_card(result, compact=False)

# CLASIFICACIÓN POR ARCHIVO
# Permite subir un archivo CSV o Excel, seleccionar la columna de texto,
# limpiar filas inválidas y clasificar todas las filas con el modelo activo.
# Muestra resumen estadístico, resultados detallados y botón de descarga CSV.
with tab2:
    st.subheader("Clasificación por archivo")

    # Advertencia especial para modelos que usan LLM (más lentos y con costo)
    if selected_model_config["type"] in {"beto_llm_ensemble", "beto_llm_cascade"}:
        st.warning(
            "Este modo usa LLM además de BETO. Puede tardar más y consumir créditos de Hugging Face al clasificar archivos grandes."
        )
    st.markdown("<div class='section-caption'>Sube un CSV o Excel y clasifícalo usando el modelo activo.</div>", unsafe_allow_html=True)

    # Widget de carga de archivo (acepta .csv y .xlsx)
    uploaded_file = st.file_uploader(
        "Sube un archivo CSV o Excel",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            # Paso 1: Inspeccionar la estructura del archivo subido
            file_info = inspect_uploaded_file(uploaded_file)

            # Paso 2: Si es Excel, permitir al usuario seleccionar la hoja
            sheet_selected = None
            if file_info["kind"] == "xlsx":
                st.write("### Opciones del Excel")
                sheet_names = file_info["sheet_names"]

                if not sheet_names:
                    st.error("El archivo Excel no contiene hojas.")
                    st.stop()

                sheet_selected = st.selectbox(
                    "Selecciona la hoja a procesar",
                    options=sheet_names
                )

            # Paso 3: Cargar el DataFrame de la hoja/archivo seleccionado
            df_uploaded = load_dataframe_from_upload(file_info, sheet_name=sheet_selected)

            if len(df_uploaded.columns) == 0:
                st.error("El archivo no contiene columnas.")
                st.stop()

            # Paso 4: Mostrar vista previa de las primeras 10 filas
            st.write("### Vista previa del archivo")
            st.dataframe(df_uploaded.head(10), use_container_width=True)

            # Paso 5: Sugerir automáticamente la columna de texto y permitir cambiarla
            suggested_text_column = suggest_text_column(df_uploaded.columns)
            default_index = 0
            if suggested_text_column is not None:
                default_index = list(df_uploaded.columns).index(suggested_text_column)

            text_column = st.selectbox(
                "Selecciona la columna que contiene el texto",
                options=list(df_uploaded.columns),
                index=default_index
            )

            # Paso 6: Limpiar el DataFrame (eliminar filas sin texto válido)
            df_ready, dropped_rows = sanitize_uploaded_dataframe(df_uploaded, text_column=text_column)

            # Mostrar métricas de filas totales, válidas y descartadas
            total_rows = len(df_uploaded)
            valid_rows = len(df_ready)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Filas totales", total_rows)
            with col_b:
                st.metric("Filas válidas", valid_rows)
            with col_c:
                st.metric("Filas descartadas", dropped_rows)

            if suggested_text_column is not None:
                st.caption(f"Columna sugerida automáticamente: **{suggested_text_column}**")

            if valid_rows == 0:
                st.warning("No hay filas válidas para clasificar después de limpiar la columna seleccionada.")

            # Paso 7: Ejecutar clasificación masiva al presionar el botón
            if st.button("Clasificar archivo"):
                if valid_rows == 0:
                    st.error("No se puede clasificar porque no hay texto válido en la columna seleccionada.")
                else:
                    # Clasificar todas las filas del DataFrame limpio
                    result_df = predict_dataframe_with_runtime(
                        runtime_bundle=active_runtime,
                        df=df_ready,
                        text_column=text_column,
                        anorexia_threshold=anorexia_threshold,
                        control_threshold=control_threshold,
                        min_words=min_words
                    )

                    # Resumen estadístico: conteo y porcentaje por clase predicha
                    st.write("### Resumen general")
                    summary = result_df["predicted_label"].value_counts(dropna=False).reset_index()
                    summary.columns = ["clase_predicha", "cantidad"]
                    summary["porcentaje"] = (
                        summary["cantidad"] / summary["cantidad"].sum() * 100
                    ).round(2)

                    st.dataframe(summary, use_container_width=True)
                    st.bar_chart(summary.set_index("clase_predicha")["cantidad"])

                    # Tabla detallada con todos los resultados fila por fila
                    st.write("### Resultados detallados")
                    st.dataframe(result_df, use_container_width=True)

                    # Botón de descarga: CSV con BOM para compatibilidad con Excel
                    csv_data = dataframe_to_csv_download(result_df)

                    st.download_button(
                        label="Descargar resultados en CSV",
                        data=csv_data,
                        file_name=f"resultados_{selected_model_config['key']}.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")

# COMPARACIÓN ENTRE MODELOS
# Evalúa el mismo texto con todos los modelos disponibles y muestra los
# resultados en tarjetas lado a lado, seguidos de una tabla resumen comparativa
# y un gráfico de barras con la probabilidad de anorexia por modelo.
with tab3:
    st.subheader("Comparación entre modelos")
    st.markdown(
        "<div class='section-caption'>Evalúa el mismo texto con todos los modelos disponibles para comparar probabilidad, confianza y etiqueta final.</div>",
        unsafe_allow_html=True
    )

    compare_text = st.text_area(
        "Texto para comparar entre modelos",
        height=150,
        placeholder="Ejemplo: llevo dos días ayunando porque me siento enorme..."
    )

    if st.button("Comparar modelos"):
        if not compare_text.strip():
            st.warning("Escribe un texto para comparar.")
        else:
            compare_results = []

            # Iterar sobre todos los modelos disponibles y clasificar el texto con cada uno
            for model_config in available_now:
                runtime = load_cached_runtime_bundle(model_config["key"])
                result = predict_single_with_runtime(
                    runtime_bundle=runtime,
                    text=compare_text,
                    anorexia_threshold=anorexia_threshold,
                    control_threshold=control_threshold,
                    min_words=min_words
                )
                # Agregar metadata del modelo al resultado para identificarlo
                result["short_label"] = model_config["short_label"]
                result["family"] = model_config["family"]
                compare_results.append(result)

            # Renderizar tarjetas lado a lado
            cols = st.columns(len(compare_results))

            for col, result in zip(cols, compare_results):
                with col:
                    st.markdown("<div class='model-card'>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='model-card-title'>{result['short_label']}</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div class='model-card-subtitle'>{result['model_label']} · {result['family']}</div>",
                        unsafe_allow_html=True
                    )
                    render_result_card(result, compact=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            # Tabla resumen comparativa y gráfico de barras
            st.write("### Resumen comparativo")
            render_summary_table(compare_results)