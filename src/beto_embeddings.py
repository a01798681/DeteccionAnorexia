from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


BETO_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"


class BETOEmbedder:
    def __init__(self, model_name: str = BETO_MODEL_NAME, max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_embeddings = last_hidden_state * mask
        summed = masked_embeddings.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: Iterable[str], batch_size: int = 16) -> np.ndarray:
        texts = [t if isinstance(t, str) else "" for t in texts]
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])

            all_embeddings.append(pooled.cpu().numpy())

        if not all_embeddings:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)

        return np.vstack(all_embeddings)