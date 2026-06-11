# Author: Andrés Cabrera Alvarado - A01798681
# Author: Andrea Elizabeth Roman Varela - A01749760
# Author: Pablo Alonso Galván - A01748288
# Fecha de creación: 10/05/2026
# Archivo: src/preprocessing.py
# Descripción general: Pipeline de preprocesamiento de texto base. Incluye:
#   - Reparación de codificación corrupta (mojibake latin1 → UTF-8).
#   - Eliminación de URLs y menciones (@usuario).
#   - Normalización de espacios en blanco.
#   - Limpieza general conservando hashtags y caracteres en español.

import re


def fix_mojibake(text: str) -> str:
    """
    Intenta reparar texto con problemas de codificación como:
    'mÃ¡s' -> 'más'
    """
    if not isinstance(text, str):
        return ""

    try:
        repaired = text.encode("latin1").decode("utf-8")
        return repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_mentions(text: str) -> str:
    return re.sub(r"@\w+", " ", text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Limpieza base:
    - corrige mojibake
    - pasa a minúsculas
    - elimina urls y menciones
    - conserva hashtags
    - elimina signos raros, pero mantiene letras, números, espacios y #
    """
    if not isinstance(text, str):
        return ""

    text = fix_mojibake(text)
    text = text.lower()
    text = remove_urls(text)
    text = remove_mentions(text)

    text = re.sub(r"[^\w\s#áéíóúüñ]", " ", text, flags=re.UNICODE)
    text = normalize_whitespace(text)

    return text