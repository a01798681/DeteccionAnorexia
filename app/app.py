import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.predict import load_model, predict_text, predict_dataframe


MODEL_PATH = ROOT_DIR / "results" / "logistic_regression_hybrid_v1.joblib"

DEFAULT_ANOREXIA_THRESHOLD = 0.70
DEFAULT_CONTROL_THRESHOLD = 0.35
DEFAULT_MIN_WORDS = 3


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Formato no soportado. Sube un archivo .csv o .xlsx")


def dataframe_to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


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

model = load_model(str(MODEL_PATH))

if "manual_mode" not in st.session_state:
    st.session_state.manual_mode = False

if "anorexia_threshold" not in st.session_state:
    st.session_state.anorexia_threshold = DEFAULT_ANOREXIA_THRESHOLD

if "control_threshold" not in st.session_state:
    st.session_state.control_threshold = DEFAULT_CONTROL_THRESHOLD

if "min_words" not in st.session_state:
    st.session_state.min_words = DEFAULT_MIN_WORDS


st.write("### Configuración del sistema")

manual_mode = st.toggle(
    "Usar configuración manual",
    value=st.session_state.manual_mode,
    key="manual_mode_toggle"
)

st.session_state.manual_mode = manual_mode

if not st.session_state.manual_mode:
    anorexia_threshold = DEFAULT_ANOREXIA_THRESHOLD
    control_threshold = DEFAULT_CONTROL_THRESHOLD
    min_words = DEFAULT_MIN_WORDS

    st.info(
        f"Modo automático activo. "
        f"Umbral anorexia = {anorexia_threshold:.2f}, "
        f"umbral control = {control_threshold:.2f}, "
        f"mínimo de palabras = {min_words}"
    )
else:
    col_cfg_1, col_cfg_2, col_cfg_3 = st.columns(3)

    with col_cfg_1:
        st.session_state.anorexia_threshold = st.slider(
            "Umbral anorexia",
            min_value=0.50,
            max_value=0.95,
            value=float(st.session_state.anorexia_threshold),
            step=0.05
        )

    with col_cfg_2:
        st.session_state.control_threshold = st.slider(
            "Umbral control",
            min_value=0.05,
            max_value=0.50,
            value=float(st.session_state.control_threshold),
            step=0.05
        )

    with col_cfg_3:
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
            df_uploaded = load_uploaded_file(uploaded_file)

            st.write("### Vista previa del archivo")
            st.dataframe(df_uploaded.head(10), use_container_width=True)

            if len(df_uploaded.columns) == 0:
                st.error("El archivo no contiene columnas.")
                st.stop()

            text_column = st.selectbox(
                "Selecciona la columna que contiene el texto",
                options=list(df_uploaded.columns)
            )

            if st.button("Clasificar archivo"):
                result_df = predict_dataframe(
                    model=model,
                    df=df_uploaded,
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