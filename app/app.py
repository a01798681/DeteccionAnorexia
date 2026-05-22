import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.predict import load_model, predict_text, predict_dataframe
from src.term_lexicon import load_custom_terms, save_custom_terms, get_term_sets


MODEL_PATH = ROOT_DIR / "results" / "logistic_regression_hybrid_v1.joblib"

DEFAULT_ANOREXIA_THRESHOLD = 0.70
DEFAULT_CONTROL_THRESHOLD = 0.35
DEFAULT_MIN_WORDS = 3

TEXT_COLUMN_HINTS = [
    "tweet_text", "text", "texto", "contenido", "post",
    "mensaje", "comentario", "body", "caption"
]


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


def inspect_uploaded_file(uploaded_file) -> dict:
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
    if file_info["kind"] == "csv":
        return file_info["df"].copy()

    return pd.read_excel(BytesIO(file_info["raw_bytes"]), sheet_name=sheet_name)


def sanitize_uploaded_dataframe(df: pd.DataFrame, text_column: str) -> tuple[pd.DataFrame, int]:
    if text_column not in df.columns:
        raise ValueError(f"La columna '{text_column}' no existe en el archivo.")

    cleaned_df = df.copy()
    series = cleaned_df[text_column]

    valid_mask = series.notna() & series.astype(str).str.strip().ne("")
    valid_mask &= ~series.astype(str).str.strip().str.lower().isin(["nan", "none", "null"])

    dropped_rows = int((~valid_mask).sum())

    cleaned_df = cleaned_df.loc[valid_mask].copy()
    cleaned_df[text_column] = cleaned_df[text_column].astype(str)

    return cleaned_df, dropped_rows


@st.cache_resource
def load_cached_model(model_path: str):
    return load_model(model_path)


st.set_page_config(
    page_title="Detector de desórdenes alimenticios",
    page_icon="🧠",
    layout="wide"
)

st.title("Detector de desórdenes alimenticios")
st.write(
    "Esta herramienta permite clasificar un solo texto o un archivo completo "
    "para estimar si se detectan señales asociadas a anorexia o si no se detectan señales claras."
)

if not MODEL_PATH.exists():
    st.error(
        f"No se encontró el modelo en: {MODEL_PATH}\n\n"
        "Primero ejecuta `python main.py` y asegúrate de tener guardado "
        "`logistic_regression_hybrid_v1.joblib` en la carpeta `results/`."
    )
    st.stop()

model = load_cached_model(str(MODEL_PATH))

if "manual_mode" not in st.session_state:
    st.session_state.manual_mode = False

if "anorexia_threshold" not in st.session_state:
    st.session_state.anorexia_threshold = DEFAULT_ANOREXIA_THRESHOLD

if "control_threshold" not in st.session_state:
    st.session_state.control_threshold = DEFAULT_CONTROL_THRESHOLD

if "min_words" not in st.session_state:
    st.session_state.min_words = DEFAULT_MIN_WORDS


with st.sidebar:
    st.header("Configuración")

    current_terms = load_custom_terms()

    manual_mode = st.toggle(
        "Usar configuración manual",
        value=st.session_state.manual_mode
    )
    st.session_state.manual_mode = manual_mode

    if not st.session_state.manual_mode:
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

    st.divider()
    st.subheader("Jerga / palabras nuevas")

    st.caption(
        "Aquí puedes agregar términos nuevos que aparezcan en redes sociales. "
        "Se guardan en un archivo JSON y se aplican a las reglas y features manuales."
    )

    risk_terms_text = st.text_area(
        "Términos de riesgo extra (uno por línea)",
        value="\n".join(current_terms["risk_terms_extra"]),
        height=180
    )

    positive_safe_text = st.text_area(
        "Términos seguros extra (uno por línea)",
        value="\n".join(current_terms["positive_safe_terms_extra"]),
        height=120
    )

    negation_safe_text = st.text_area(
        "Términos de negación segura extra (uno por línea)",
        value="\n".join(current_terms["negation_safe_terms_extra"]),
        height=120
    )

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

    merged_terms = get_term_sets()
    st.caption(
        f"Términos activos: riesgo={len(merged_terms['risk_terms'])}, "
        f"seguros={len(merged_terms['positive_safe_terms'])}, "
        f"negación segura={len(merged_terms['negation_safe_terms'])}"
    )

tab1, tab2 = st.tabs(["Clasificación individual", "Clasificación por archivo"])

with tab1:
    st.subheader("Clasificación individual")

    user_text = st.text_area(
        "Escribe el texto a evaluar",
        height=180,
        placeholder="Ejemplo: hoy no quiero comer nada porque me siento gorda..."
    )

    if st.button("Clasificar texto"):
        if not user_text.strip():
            st.warning("Por favor escribe un texto antes de clasificar.")
        else:
            result = predict_text(
                model=model,
                text=user_text,
                anorexia_threshold=anorexia_threshold,
                control_threshold=control_threshold,
                min_words=min_words
            )

            st.subheader("Resultado")

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Confianza", result["confidence"])
            with metric_col2:
                st.metric("Cantidad de palabras", result["word_count"])
            with metric_col3:
                st.metric("Cobertura del vocabulario", f"{result['vocab_coverage']:.2f}")

            if result["probability_anorexia"] is not None:
                st.write(f"**Probabilidad de anorexia:** {result['probability_anorexia']:.4f}")

            st.write(f"**Mensaje:** {result['message']}")
            st.write(f"**Observaciones:** {result['observations']}")

            st.subheader("Texto procesado")
            st.code(result["cleaned_text"])

            if result["predicted_label"] == "anorexia":
                st.error("Se detectan señales asociadas a anorexia en el texto.")
            elif result["predicted_label"] == "control":
                st.success("No se detectan señales claras asociadas a anorexia en el texto.")
            elif result["predicted_label"] == "incierto":
                st.warning("El resultado es incierto. Se recomienda revisión manual.")

with tab2:
    st.subheader("Clasificación por archivo")
    st.write("La clasificación del archivo usa la misma configuración actualmente activa.")

    uploaded_file = st.file_uploader(
        "Sube un archivo CSV o Excel",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            file_info = inspect_uploaded_file(uploaded_file)

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

            df_uploaded = load_dataframe_from_upload(file_info, sheet_name=sheet_selected)

            if len(df_uploaded.columns) == 0:
                st.error("El archivo no contiene columnas.")
                st.stop()

            st.write("### Vista previa del archivo")
            st.dataframe(df_uploaded.head(10), use_container_width=True)

            suggested_text_column = suggest_text_column(df_uploaded.columns)
            default_index = 0
            if suggested_text_column is not None:
                default_index = list(df_uploaded.columns).index(suggested_text_column)

            text_column = st.selectbox(
                "Selecciona la columna que contiene el texto",
                options=list(df_uploaded.columns),
                index=default_index
            )

            df_ready, dropped_rows = sanitize_uploaded_dataframe(df_uploaded, text_column=text_column)

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

            if st.button("Clasificar archivo"):
                if valid_rows == 0:
                    st.error("No se puede clasificar porque no hay texto válido en la columna seleccionada.")
                else:
                    result_df = predict_dataframe(
                        model=model,
                        df=df_ready,
                        text_column=text_column,
                        anorexia_threshold=anorexia_threshold,
                        control_threshold=control_threshold,
                        min_words=min_words
                    )

                    st.write("### Resumen general")
                    summary = result_df["predicted_label"].value_counts(dropna=False).reset_index()
                    summary.columns = ["clase_predicha", "cantidad"]
                    summary["porcentaje"] = (
                        summary["cantidad"] / summary["cantidad"].sum() * 100
                    ).round(2)

                    st.dataframe(summary, use_container_width=True)
                    st.bar_chart(summary.set_index("clase_predicha")["cantidad"])

                    st.write("### Resultados detallados")
                    st.dataframe(result_df, use_container_width=True)

                    csv_data = dataframe_to_csv_download(result_df)

                    st.download_button(
                        label="Descargar resultados en CSV",
                        data=csv_data,
                        file_name="resultados_clasificacion.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")