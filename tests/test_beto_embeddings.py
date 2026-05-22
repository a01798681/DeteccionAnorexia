import numpy as np
import torch

from src.beto_embeddings import BETOEmbedder


class FakeTokenizer:
    def __call__(self, batch, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        batch_size = len(batch)
        seq_len = 4
        return {
            "input_ids": torch.ones((batch_size, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        }


class FakeModelOutput:
    def __init__(self, batch_size, seq_len, hidden_size):
        self.last_hidden_state = torch.ones((batch_size, seq_len, hidden_size), dtype=torch.float32)


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


def test_mean_pool_shape():
    last_hidden = torch.ones((2, 4, 8))
    attention_mask = torch.ones((2, 4), dtype=torch.long)
    pooled = BETOEmbedder._mean_pool(last_hidden, attention_mask)
    assert pooled.shape == (2, 8)


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