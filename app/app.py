import os
import sys
from pathlib import Path

import streamlit as st

# Permite importar desde src/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.predict import load_model, predict_text


MODEL_PATH = ROOT_DIR / "results" / "logistic_regression.joblib"


st.set_page_config(
    page_title="Detector de desórdenes alimenticios",
    page_icon="🧠",
    layout="centered"
)

st.title("Detector de desórdenes alimenticios")
st.write("Ingresa un texto y el modelo estimará si se parece más a la clase `control` o `anorexia`.")

if not MODEL_PATH.exists():
    st.error(
        f"No se encontró el modelo en: {MODEL_PATH}\n\n"
        "Primero ejecuta `python main.py` para entrenar y guardar el modelo."
    )
    st.stop()

model = load_model(str(MODEL_PATH))

user_text = st.text_area(
    "Escribe el texto a evaluar",
    height=180,
    placeholder="Ejemplo: hoy no quiero comer nada porque me siento gorda..."
)

if st.button("Clasificar texto"):
    if not user_text.strip():
        st.warning("Por favor escribe un texto antes de clasificar.")
    else:
        result = predict_text(model, user_text)

        st.subheader("Resultado")
        st.write(f"**Clase predicha:** {result['predicted_label']}")

        if result["probability_anorexia"] is not None:
            st.write(f"**Probabilidad de anorexia:** {result['probability_anorexia']:.4f}")

        st.subheader("Texto procesado")
        st.code(result["cleaned_text"])

        if result["predicted_label"] == "anorexia":
            st.error("El modelo considera que este texto se parece más a la clase anorexia.")
        else:
            st.success("El modelo considera que este texto se parece más a la clase control.")