import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.predict import load_model, predict_text, predict_dataframe


MODEL_PATH = ROOT_DIR / "results" / "logistic_regression_hybrid_v1.joblib"

def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Carga un archivo CSV o Excel subido en Streamlit.
    """
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
    "para estimar si se parece más a la clase `control` o `anorexia`."
)

if not MODEL_PATH.exists():
    st.error(
        f"No se encontró el modelo en: {MODEL_PATH}\n\n"
        "Primero ejecuta `python main.py` para entrenar y guardar el modelo."
    )
    st.stop()

model = load_model(str(MODEL_PATH))

tab1, tab2 = st.tabs(["Clasificación individual", "Clasificación por archivo"])


with tab1:
    st.subheader("Clasificación individual")

    user_text = st.text_area(
        "Escribe el texto a evaluar",
        height=180,
        placeholder="Ejemplo: hoy no quiero comer nada porque me siento gorda..."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        anorexia_threshold = st.slider(
            "Umbral anorexia",
            min_value=0.50,
            max_value=0.95,
            value=0.70,
            step=0.05
        )
    with col2:
        control_threshold = st.slider(
            "Umbral control",
            min_value=0.05,
            max_value=0.50,
            value=0.30,
            step=0.05
        )
    with col3:
        min_words = st.slider(
            "Mínimo de palabras",
            min_value=1,
            max_value=10,
            value=4,
            step=1
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
            st.write(f"**Clase predicha:** {result['predicted_label']}")
            st.write(f"**Nivel de confianza:** {result['confidence']}")

            if result["probability_anorexia"] is not None:
                st.write(f"**Probabilidad de anorexia:** {result['probability_anorexia']:.4f}")

            st.write(f"**Mensaje:** {result['message']}")
            st.write(f"**Observaciones:** {result['observations']}")

            st.subheader("Texto procesado")
            st.code(result["cleaned_text"])

            if result["predicted_label"] == "anorexia":
                st.error("El modelo considera que este texto se parece más a la clase anorexia.")
            elif result["predicted_label"] == "control":
                st.success("El modelo considera que este texto se parece más a la clase control.")
            elif result["predicted_label"] == "incierto":
                st.warning("El modelo no tiene suficiente certeza. Se recomienda revisión manual.")
            else:
                st.info("El texto es demasiado corto o insuficiente para clasificarlo con confianza.")


with tab2:
    st.subheader("Clasificación por archivo")

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

            col1, col2, col3 = st.columns(3)
            with col1:
                batch_anorexia_threshold = st.slider(
                    "Umbral anorexia (archivo)",
                    min_value=0.50,
                    max_value=0.95,
                    value=0.70,
                    step=0.05,
                    key="batch_anorexia_threshold"
                )
            with col2:
                batch_control_threshold = st.slider(
                    "Umbral control (archivo)",
                    min_value=0.05,
                    max_value=0.50,
                    value=0.30,
                    step=0.05,
                    key="batch_control_threshold"
                )
            with col3:
                batch_min_words = st.slider(
                    "Mínimo de palabras (archivo)",
                    min_value=1,
                    max_value=10,
                    value=4,
                    step=1,
                    key="batch_min_words"
                )

            if st.button("Clasificar archivo"):
                result_df = predict_dataframe(
                    model=model,
                    df=df_uploaded,
                    text_column=text_column,
                    anorexia_threshold=batch_anorexia_threshold,
                    control_threshold=batch_control_threshold,
                    min_words=batch_min_words
                )

                st.write("### Resultados")
                st.dataframe(result_df, use_container_width=True)

                st.write("### Resumen")
                summary = result_df["predicted_label"].value_counts(dropna=False).reset_index()
                summary.columns = ["clase_predicha", "cantidad"]
                st.dataframe(summary, use_container_width=True)

                csv_data = dataframe_to_csv_download(result_df)

                st.download_button(
                    label="Descargar resultados en CSV",
                    data=csv_data,
                    file_name="resultados_clasificacion.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")