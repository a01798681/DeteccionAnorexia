import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from .llm_prompts import FEW_SHOT_PROMPT

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.3-70B-Instruct")
HF_PROVIDER = os.getenv("HF_PROVIDER", "groq")
HF_USE_MOCK = os.getenv("HF_USE_MOCK", "true").lower() == "true"


def build_prompt(text: str) -> str:
    return FEW_SHOT_PROMPT.format(text=text)


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


def classify_text_with_hf(text: str) -> Dict[str, Any]:
    if not HF_API_TOKEN:
        raise ValueError("No se encontró HF_API_TOKEN en variables de entorno.")

    prompt = build_prompt(text)

    client = InferenceClient(
        provider=HF_PROVIDER,
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


def classify_text(text: str) -> Dict[str, Any]:
    if HF_USE_MOCK:
        return classify_text_mock(text)

    return classify_text_with_hf(text)