# Author: Andrés Cabrera Alvarado - A01798681
# Author: Andrea Elizabeth Roman Varela - A01749760
# Author: Pablo Alonso Galván - A01748288
# Fecha de creación: 05/06/2026
# Archivo: tests/test_beto_embeddings.py
# Descripción general: Pruebas unitarias para la clase BETOEmbedder.
# Utiliza mocks (FakeTokenizer, FakeModel) para simular el comportamiento del modelo de lenguaje 
# sin necesidad de descargar pesos reales durante el testing.

import numpy as np
import torch

from src.beto_embeddings import BETOEmbedder


# Simula el comportamiento del tokenizador de Hugging Face devolviendo
# tensores predecibles de unos para input_ids y attention_mask.
class FakeTokenizer:
    def __call__(self, batch, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        batch_size = len(batch)
        seq_len = 4
        return {
            "input_ids": torch.ones((batch_size, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        }


# Simula el objeto de salida de un modelo BERT, conteniendo únicamente
# el atributo last_hidden_state requerido por el embedder.
class FakeModelOutput:
    def __init__(self, batch_size, seq_len, hidden_size):
        self.last_hidden_state = torch.ones((batch_size, seq_len, hidden_size), dtype=torch.float32)


# Simula un modelo de lenguaje de Hugging Face. Implementa la interfaz mínima
# requerida (to, eval, __call__) y devuelve un FakeModelOutput simulado.
class FakeModel:
    def __init__(self, hidden_size=8):
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.hidden_size = hidden_size

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **encoded):
        batch_size, seq_len = encoded["input_ids"].shape
        return FakeModelOutput(batch_size, seq_len, self.hidden_size)


# Verifica que la función _mean_pool reduzca correctamente las dimensiones
# espaciales (seq_len) basándose en la máscara de atención.
def test_mean_pool_shape():
    last_hidden = torch.ones((2, 4, 8))
    attention_mask = torch.ones((2, 4), dtype=torch.long)
    pooled = BETOEmbedder._mean_pool(last_hidden, attention_mask)
    assert pooled.shape == (2, 8)


# Comprueba que el método encode procese los textos correctamente utilizando
# los modelos falsos y devuelva un array de NumPy con la forma esperada.
def test_encode_returns_expected_shape():
    embedder = BETOEmbedder.__new__(BETOEmbedder)
    embedder.model_name = "fake"
    embedder.max_length = 128
    embedder.device = torch.device("cpu")
    embedder.tokenizer = FakeTokenizer()
    embedder.model = FakeModel(hidden_size=8)

    vectors = embedder.encode(["hola mundo", "quiero ser flaca"], batch_size=2)
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, 8)


# Asegura que el método encode maneje correctamente una lista vacía de textos
# sin arrojar errores y devolviendo un array con la forma correcta.
def test_encode_empty_input():
    embedder = BETOEmbedder.__new__(BETOEmbedder)
    embedder.model_name = "fake"
    embedder.max_length = 128
    embedder.device = torch.device("cpu")
    embedder.tokenizer = FakeTokenizer()
    embedder.model = FakeModel(hidden_size=8)

    vectors = embedder.encode([], batch_size=2)
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (0, 8)