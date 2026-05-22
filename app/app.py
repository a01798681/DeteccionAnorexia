import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.model_registry import get_available_models
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


def inject_custom_css():
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
def load_cached_runtime_bundle(model_key: str):
    return load_runtime_bundle(model_key)


def prediction_css_class(label: str) -> str:
    if label == "anorexia":
        return "pred-anorexia"
    if label == "control":
        return "pred-control"
    return "pred-incierto"


def render_result_card(result: dict, compact: bool = False):
    css_class = prediction_css_class(result["predicted_label"])

    st.markdown(
        f"<div class='prediction-pill {css_class}'>Predicción: {result['predicted_label']}</div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confianza", result["confidence"])
    with col2:
        prob = result["probability_anorexia"]
        st.metric("Prob. anorexia", "N/A" if prob is None else f"{prob:.4f}")

    info_cols = st.columns(2)
    with info_cols[0]:
        st.metric("Palabras", result["word_count"])
    with info_cols[1]:
        coverage = result["vocab_coverage"]
        st.metric("Cobertura", "N/A" if coverage is None else f"{coverage:.2f}")

    st.markdown(f"**Mensaje:** {result['message']}")
    st.markdown(f"**Observaciones:** {result['observations']}")

    if compact:
        with st.expander("Texto procesado"):
            st.code(result["cleaned_text"])
    else:
        st.subheader("Texto procesado")
        st.code(result["cleaned_text"])


def render_summary_table(compare_results: list[dict]):
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

    display_df = summary_df.copy()
    if "probabilidad_anorexia" in display_df.columns:
        display_df["probabilidad_anorexia"] = display_df["probabilidad_anorexia"].apply(
            lambda x: None if x is None else round(float(x), 4)
        )

    st.dataframe(display_df, use_container_width=True)

    chart_df = summary_df.copy()
    chart_df["probabilidad_anorexia"] = chart_df["probabilidad_anorexia"].fillna(0.0)
    st.bar_chart(chart_df.set_index("modelo")["probabilidad_anorexia"])


st.set_page_config(
    page_title="Detector de desórdenes alimenticios",
    page_icon="🧠",
    layout="wide"
)

inject_custom_css()

st.title("Detector de desórdenes alimenticios")
st.write(
    "Esta herramienta permite clasificar textos individuales o archivos completos "
    "para estimar si se detectan señales asociadas a anorexia."
)

if "manual_mode" not in st.session_state:
    st.session_state.manual_mode = False

if "anorexia_threshold" not in st.session_state:
    st.session_state.anorexia_threshold = DEFAULT_ANOREXIA_THRESHOLD

if "control_threshold" not in st.session_state:
    st.session_state.control_threshold = DEFAULT_CONTROL_THRESHOLD

if "min_words" not in st.session_state:
    st.session_state.min_words = DEFAULT_MIN_WORDS

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

    model_labels = [m["label"] for m in available_now]
    selected_label = st.selectbox("Modelo activo", options=model_labels)
    selected_model_config = next(m for m in available_now if m["label"] == selected_label)

    st.caption(f"**Tipo:** {selected_model_config['type']}")
    st.caption(f"**Familia:** {selected_model_config['family']}")
    st.caption(selected_model_config["description"])

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
        height=140
    )

    positive_safe_text = st.text_area(
        "Términos seguros extra (uno por línea)",
        value="\n".join(current_terms["positive_safe_terms_extra"]),
        height=100
    )

    negation_safe_text = st.text_area(
        "Términos de negación segura extra (uno por línea)",
        value="\n".join(current_terms["negation_safe_terms_extra"]),
        height=100
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

active_runtime = load_cached_runtime_bundle(selected_model_config["key"])

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

chips_html = "".join(
    [f"<span class='soft-chip'>{m['short_label']} · {m['family']}</span>" for m in available_now]
)
st.markdown(chips_html, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "Clasificación individual",
    "Clasificación por archivo",
    "Comparación entre modelos"
])

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

with tab2:
    st.subheader("Clasificación por archivo")
    st.markdown("<div class='section-caption'>Sube un CSV o Excel y clasifícalo usando el modelo activo.</div>", unsafe_allow_html=True)

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
                    result_df = predict_dataframe_with_runtime(
                        runtime_bundle=active_runtime,
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
                        file_name=f"resultados_{selected_model_config['key']}.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")

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

            for model_config in available_now:
                runtime = load_cached_runtime_bundle(model_config["key"])
                result = predict_single_with_runtime(
                    runtime_bundle=runtime,
                    text=compare_text,
                    anorexia_threshold=anorexia_threshold,
                    control_threshold=control_threshold,
                    min_words=min_words
                )
                result["short_label"] = model_config["short_label"]
                result["family"] = model_config["family"]
                compare_results.append(result)

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

            st.write("### Resumen comparativo")
            render_summary_table(compare_results)