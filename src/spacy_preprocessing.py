from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import pandas as pd
import spacy

from .preprocessing import clean_text


SPACY_MODEL_NAME = "es_core_news_sm"


@lru_cache(maxsize=1)
def get_spacy_nlp(model_name: str = SPACY_MODEL_NAME):
    try:
        return spacy.load(model_name, disable=["ner"])
    except OSError as e:
        raise RuntimeError(
            f"No se pudo cargar el modelo spaCy '{model_name}'. "
            f"Instálalo con: python -m spacy download {model_name}"
        ) from e


def normalize_spacy_text(text: str) -> str:
    text = clean_text(text if isinstance(text, str) else "")
    if not text:
        return ""

    nlp = get_spacy_nlp()
    doc = nlp(text)

    tokens: List[str] = []

    for token in doc:
        raw = token.text.strip().lower()

        if not raw or token.is_space:
            continue

        if raw.startswith("#"):
            tokens.append(raw)
            continue

        if token.is_punct:
            continue

        lemma = token.lemma_.strip().lower()
        tokens.append(lemma if lemma else raw)

    return " ".join(tokens).strip()


def normalize_texts_spacy(texts: Iterable[str], batch_size: int = 64) -> pd.Series:
    texts = [clean_text(t if isinstance(t, str) else "") for t in texts]
    if not texts:
        return pd.Series(dtype=str)

    nlp = get_spacy_nlp()
    normalized: List[str] = []

    for doc in nlp.pipe(texts, batch_size=batch_size):
        tokens: List[str] = []

        for token in doc:
            raw = token.text.strip().lower()

            if not raw or token.is_space:
                continue

            if raw.startswith("#"):
                tokens.append(raw)
                continue

            if token.is_punct:
                continue

            lemma = token.lemma_.strip().lower()
            tokens.append(lemma if lemma else raw)

        normalized.append(" ".join(tokens).strip())

    return pd.Series(normalized)