# Author: Andrés Cabrera Alvarado - A01798681
# Author: Andrea Elizabeth Roman Varela - A01749760
# Author: Pablo Alonso Galván - A01748288
# Fecha de creación: 05/06/2026
# Archivo: src/features.py
# Descripción general: Módulo para la extracción de características del texto.
#   Define el vectorizador TF-IDF y un extractor de características manuales
#   (ManualFeatureExtractor) basado en términos de riesgo, negaciones y contexto.

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from .term_lexicon import get_term_sets


# Construye y retorna un vectorizador TF-IDF con parámetros predefinidos.
def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )


# Cuenta la frecuencia total de aparición de un conjunto de términos en un texto.
def count_any(text: str, terms) -> int:
    text = text.lower()
    return sum(1 for term in terms if term in text)


# Transformador personalizado para scikit-learn que extrae características
# manuales a partir de un arreglo de textos.
class ManualFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    # Aplica la extracción de características (conteos, banderas de contexto, 
    # longitud de texto, hashtags, etc.) y retorna un DataFrame con las mismas.
    def transform(self, X):
        term_sets = get_term_sets()
        risk_terms = term_sets["risk_terms"]
        positive_safe_terms = term_sets["positive_safe_terms"]
        negation_safe_terms = term_sets["negation_safe_terms"]

        features = []

        for text in X:
            text = text if isinstance(text, str) else ""
            text_lower = text.lower()
            words = text_lower.split()

            has_thinspo = int("#thinspo" in text_lower or "thinspo" in text_lower)
            has_thinspiration = int("#thinspiration" in text_lower or "thinspiration" in text_lower)
            has_proana = int("#proana" in text_lower or "proana" in text_lower or "#ana" in text_lower)
            has_vomit_term = int(any(term in text_lower for term in ["vomit", "vomitar", "vomitando", "purga", "purging"]))
            has_weight_term = int(any(term in text_lower for term in ["peso", "gorda", "flaca", "adelgazar", "bajar de peso", "grasa"]))
            has_fasting_term = int(any(term in text_lower for term in ["ayuno", "ayunas", "sin comer", "no comer", "dejar de comer"]))
            has_body_term = int(any(term in text_lower for term in ["abdomen", "cuerpo", "bodycheck"]))
            has_hashtag = int("#" in text_lower)

            risk_term_count = count_any(text_lower, risk_terms)
            positive_safe_count = count_any(text_lower, positive_safe_terms)
            negation_safe_count = count_any(text_lower, negation_safe_terms)

            has_positive_safe = int(positive_safe_count > 0)
            has_negation_safe = int(negation_safe_count > 0)

            positive_without_risk = int(positive_safe_count > 0 and risk_term_count == 0)
            safe_context = int((positive_safe_count + negation_safe_count) > 0 and risk_term_count == 0)
            mixed_context = int((positive_safe_count + negation_safe_count) > 0 and risk_term_count > 0)

            features.append({
                "text_length_chars": len(text_lower),
                "text_length_words": len(words),
                "has_thinspo": has_thinspo,
                "has_thinspiration": has_thinspiration,
                "has_proana": has_proana,
                "has_vomit_term": has_vomit_term,
                "has_weight_term": has_weight_term,
                "has_fasting_term": has_fasting_term,
                "has_body_term": has_body_term,
                "has_hashtag": has_hashtag,
                "risk_term_count": risk_term_count,
                "positive_safe_count": positive_safe_count,
                "negation_safe_count": negation_safe_count,
                "has_positive_safe": has_positive_safe,
                "has_negation_safe": has_negation_safe,
                "positive_without_risk": positive_without_risk,
                "safe_context": safe_context,
                "mixed_context": mixed_context
            })

        return pd.DataFrame(features)