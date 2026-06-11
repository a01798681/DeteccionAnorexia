# Author: Andrés Cabrera Alvarado - A01798681
# Author: Andrea Elizabeth Roman Varela - A01749760
# Author: Pablo Alonso Galván - A01748288
# Fecha de creación: 05/06/2026
# Archivo: tests/test_spacy_preprocessing.py
# Descripción general: Pruebas unitarias para el procesamiento avanzado de texto con spaCy (lematización, filtrado de signos de puntuación). 
# Simula el comportamiento del pipeline de spaCy (FakeNLP) para evitar descargas pesadas durante los tests.

import pandas as pd

import src.spacy_preprocessing as sp


# Simula un token de spaCy con los atributos básicos necesarios (text, lemma_, is_space, is_punct).
class FakeToken:
    def __init__(self, text, lemma=None, is_space=False, is_punct=False):
        self.text = text
        self.lemma_ = lemma if lemma is not None else text
        self.is_space = is_space
        self.is_punct = is_punct


# Simula un objeto Doc de spaCy (esencialmente una lista de FakeTokens).
class FakeDoc(list):
    pass


# Simula el modelo de lenguaje de spaCy con llamadas directas (__call__)
# y procesamiento por lotes (pipe), retornando tokens falsos predecibles.
class FakeNLP:
    def __call__(self, text):
        tokens = [
            FakeToken("quiero", "querer"),
            FakeToken("comiendo", "comer"),
            FakeToken("#thinspo", "#thinspo"),
            FakeToken(".", ".", is_punct=True),
        ]
        return FakeDoc(tokens)

    def pipe(self, texts, batch_size=64):
        for _ in texts:
            yield FakeDoc([
                FakeToken("vomitando", "vomitar"),
                FakeToken("#ana", "#ana"),
                FakeToken(".", ".", is_punct=True),
            ])


# Verifica que la normalización de un texto aplique lematización ("comiendo" -> "comer"),
# mantenga intactos los hashtags (#thinspo) y elimine signos de puntuación.
def test_normalize_spacy_text_uses_lemmas_and_keeps_hashtag(monkeypatch):
    monkeypatch.setattr(sp, "get_spacy_nlp", lambda *args, **kwargs: FakeNLP())
    result = sp.normalize_spacy_text("Quiero comiendo #thinspo.")
    assert "querer" in result
    assert "comer" in result
    assert "#thinspo" in result
    assert "." not in result


# Comprueba que el procesamiento en lote (pipe) devuelva una Serie de Pandas
# con la misma cantidad de elementos que la lista de entrada.
def test_normalize_texts_spacy_returns_expected_length(monkeypatch):
    monkeypatch.setattr(sp, "get_spacy_nlp", lambda *args, **kwargs: FakeNLP())
    result = sp.normalize_texts_spacy(["uno", "dos", "tres"])
    assert isinstance(result, pd.Series)
    assert len(result) == 3


# Verifica que la función maneje adecuadamente listas vacías sin errores, retornando una Serie vacía.
def test_normalize_texts_spacy_empty_input(monkeypatch):
    monkeypatch.setattr(sp, "get_spacy_nlp", lambda *args, **kwargs: FakeNLP())
    result = sp.normalize_texts_spacy([])
    assert isinstance(result, pd.Series)
    assert len(result) == 0