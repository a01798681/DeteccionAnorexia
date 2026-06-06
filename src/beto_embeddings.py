# Author: Andrés Cabrera Alvarado - A01798681
# Fecha de creación: 10/05/2026
# Archivo: src/beto_embeddings.py
# Descripción general: Clase que encapsula el modelo BETO (BERT en español) para
#   generar embeddings de texto. Carga el modelo preentrenado, tokeniza textos
#   en lotes y aplica mean pooling sobre la última capa oculta para obtener
#   vectores densos de dimensión fija. Estos embeddings alimentan al clasificador
#   Logistic Regression en la fase BETO del pipeline.

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Nombre del modelo preentrenado BETO (BERT español) alojado en Hugging Face.
BETO_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"


class BETOEmbedder:
    """Genera embeddings de texto en español usando el modelo BETO."""

    def __init__(self, model_name: str = BETO_MODEL_NAME, max_length: int = 128):
        """Carga el tokenizer y el modelo, y los mueve a GPU si está disponible."""
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Aplica mean pooling sobre los tokens válidos (no padding) de la última capa oculta."""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_embeddings = last_hidden_state * mask
        summed = masked_embeddings.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: Iterable[str], batch_size: int = 16) -> np.ndarray:
        """Convierte una lista de textos en una matriz de embeddings (NumPy)."""
        texts = [t if isinstance(t, str) else "" for t in texts]
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            # Tokeniza el lote con padding y truncamiento.
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Inferencia sin gradientes para mayor eficiencia.
            with torch.no_grad():
                outputs = self.model(**encoded)
                pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])

            all_embeddings.append(pooled.cpu().numpy())

        if not all_embeddings:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)

        return np.vstack(all_embeddings)