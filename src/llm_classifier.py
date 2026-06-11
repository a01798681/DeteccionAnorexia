# Author: Andrés Cabrera Alvarado - A01798681
# Author: Andrea Elizabeth Roman Varela - A01749760
# Author: Pablo Alonso Galván - A01748288
# Fecha de creación: 05/06/2026
# Archivo: src/llm_classifier.py
# Descripción general: Módulo para la clasificación de textos utilizando modelos
#   de lenguaje grandes (LLMs) a través de la API de Hugging Face. Incluye lógica
#   para recuperar ejemplos dinámicos (RAG simple), construir prompts, invocar
#   el modelo y procesar/parsear la respuesta JSON. También incluye un modo "mock".

import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from .llm_prompts import build_dynamic_prompt
from .llm_retrieval import retrieve_examples

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.3-70B-Instruct")
HF_USE_MOCK = os.getenv("HF_USE_MOCK", "true").lower() == "true"


# Construye el prompt dinámico recuperando ejemplos relevantes para el texto dado.
def build_prompt(text: str) -> str:
    examples = retrieve_examples(text, k_per_class=3, max_chars=180)
    return build_dynamic_prompt(text, examples)


# Analiza y extrae el JSON de la respuesta cruda del LLM.
# Maneja posibles bloques de código Markdown y retorna un diccionario con
# etiqueta, puntaje de riesgo y razón. En caso de error, retorna valores por defecto.
def parse_llm_response(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw_text)
        return {
            "label": data.get("label", "control"),
            "risk_score": float(data.get("risk_score", 0.5)),
            "reason": data.get("reason", "Sin explicación.")
        }
    except Exception:
        return {
            "label": "control",
            "risk_score": 0.5,
            "reason": f"Respuesta no parseable: {raw_text}"
        }


# Simula la clasificación del LLM buscando términos de riesgo estáticos.
# Útil para pruebas y desarrollo sin gastar cuota de la API.
def classify_text_mock(text: str) -> Dict[str, Any]:
    text_lower = text.lower()

    risk_terms = [
        "flaca", "gorda", "ayuno", "vomitar", "vomito",
        "dejar de comer", "no quiero comer", "thinspo",
        "thinspiration", "proana"
    ]

    if any(term in text_lower for term in risk_terms):
        return {
            "label": "anorexia",
            "risk_score": 0.92,
            "reason": "Clasificación mock: se detectaron términos de riesgo.",
            "model_id": "mock",
            "raw_response": "mock_response"
        }

    return {
        "label": "control",
        "risk_score": 0.08,
        "reason": "Clasificación mock: no se detectaron términos de riesgo.",
        "model_id": "mock",
        "raw_response": "mock_response"
    }


# Clasifica un texto invocando a la API de inferencia de Hugging Face.
# Construye el prompt, llama al modelo especificado en las variables de entorno
# y parsea la respuesta.
def classify_text_with_hf(text: str) -> Dict[str, Any]:
    if not HF_API_TOKEN:
        raise ValueError("No se encontró HF_API_TOKEN en variables de entorno.")

    prompt = build_prompt(text)

    client = InferenceClient(
        provider="auto",
        api_key=HF_API_TOKEN,
    )

    completion = client.chat.completions.create(
        model=HF_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=150,
        temperature=0.2,
    )

    generated_text = completion.choices[0].message.content

    parsed = parse_llm_response(generated_text)
    parsed["model_id"] = HF_MODEL_ID
    parsed["raw_response"] = generated_text

    return parsed


# Función principal de enrutamiento: decide si usar el clasificador real (API HF)
# o el simulado (mock) basado en las variables de entorno (HF_USE_MOCK).
def classify_text(text: str) -> Dict[str, Any]:
    if HF_USE_MOCK:
        return classify_text_mock(text)

    return classify_text_with_hf(text)