# Author: Andrés Cabrera Alvarado - A01798681
# Fecha de creación: 05/06/2026
# Archivo: src/term_lexicon.py
# Descripción general: Define listas de términos base (riesgo, seguros, negaciones)
# y maneja la carga/guardado de términos personalizados desde un archivo JSON.
# Provee funciones para normalizar y combinar ambos conjuntos de términos.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parent.parent
CUSTOM_TERMS_PATH = ROOT_DIR / "data" / "custom_terms.json"


BASE_RISK_TERMS = [
    "vomit", "vomitar", "vomitando", "purga", "purging",
    "adelgazar", "bajar de peso", "peso", "gorda", "flaca",
    "abdomen", "cuerpo", "grasa", "ayuno", "ayunas",
    "thinspo", "thinspiration", "proana", "#thinspo", "#thinspiration",
    "#proana", "#ana", "#mia", "dejar de comer", "no quiero comer",
    "quiero ser flaca", "me siento gorda", "anorexia", "bulimia",
    "bodycheck", "edtwt", "skinnytok"
]

BASE_POSITIVE_SAFE_TERMS = [
    "me siento bien",
    "estoy bien",
    "sin problema",
    "tranquilo",
    "tranquila",
    "disfruté",
    "disfrute",
    "con mis amigos",
    "con mi familia",
    "me gusta comer",
    "comí con mis amigos",
    "comi con mis amigos",
    "me siento bien conmigo",
    "me siento bien conmigo mismo",
    "me siento bien conmigo misma",
    "me siento bien con mi cuerpo",
    "no tengo problema",
    "no tengo problema con mi cuerpo",
    "disfruté mucho la comida",
    "disfrute mucho la comida"
]

BASE_NEGATION_SAFE_TERMS = [
    "no tengo problema",
    "no tengo problema con mi cuerpo",
    "no quiero dejar de comer",
    "sí como",
    "si como",
    "como normal",
    "comí normal",
    "comi normal"
]


# Limpia, pasa a minúsculas y elimina duplicados de una lista de términos.
def _normalize_terms(terms: List[str]) -> List[str]:
    seen = set()
    normalized = []

    for term in terms or []:
        if term is None:
            continue

        term = str(term).strip().lower()
        if not term:
            continue

        if term not in seen:
            seen.add(term)
            normalized.append(term)

    return normalized


# Verifica que exista el archivo JSON de términos personalizados. Si no existe,
# lo crea con una estructura vacía por defecto.
def ensure_custom_terms_file(path: Path = CUSTOM_TERMS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        default_payload = {
            "risk_terms_extra": [],
            "positive_safe_terms_extra": [],
            "negation_safe_terms_extra": []
        }
        path.write_text(
            json.dumps(default_payload, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )


# Carga y normaliza los términos adicionales definidos por el usuario desde
# el archivo JSON de configuración.
def load_custom_terms(path: Path = CUSTOM_TERMS_PATH) -> Dict[str, List[str]]:
    ensure_custom_terms_file(path)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}

    return {
        "risk_terms_extra": _normalize_terms(payload.get("risk_terms_extra", [])),
        "positive_safe_terms_extra": _normalize_terms(payload.get("positive_safe_terms_extra", [])),
        "negation_safe_terms_extra": _normalize_terms(payload.get("negation_safe_terms_extra", [])),
    }


# Guarda listas de términos adicionales definidos por el usuario en el archivo JSON.
def save_custom_terms(
    risk_terms_extra: List[str],
    positive_safe_terms_extra: List[str] | None = None,
    negation_safe_terms_extra: List[str] | None = None,
    path: Path = CUSTOM_TERMS_PATH
) -> None:
    ensure_custom_terms_file(path)

    payload = {
        "risk_terms_extra": _normalize_terms(risk_terms_extra or []),
        "positive_safe_terms_extra": _normalize_terms(positive_safe_terms_extra or []),
        "negation_safe_terms_extra": _normalize_terms(negation_safe_terms_extra or []),
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# Retorna un diccionario con las listas finales de términos combinando
# las listas base hardcodeadas con los términos personalizados del usuario.
def get_term_sets(path: Path = CUSTOM_TERMS_PATH) -> Dict[str, List[str]]:
    custom = load_custom_terms(path)

    risk_terms = _normalize_terms(BASE_RISK_TERMS + custom["risk_terms_extra"])
    positive_safe_terms = _normalize_terms(BASE_POSITIVE_SAFE_TERMS + custom["positive_safe_terms_extra"])
    negation_safe_terms = _normalize_terms(BASE_NEGATION_SAFE_TERMS + custom["negation_safe_terms_extra"])

    return {
        "risk_terms": risk_terms,
        "positive_safe_terms": positive_safe_terms,
        "negation_safe_terms": negation_safe_terms,
        "custom": custom
    }