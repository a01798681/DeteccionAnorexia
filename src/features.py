import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


RISK_TERMS = [
    "vomit", "vomitar", "vomitando", "purga", "purging",
    "adelgazar", "bajar de peso", "peso", "gorda", "flaca",
    "abdomen", "cuerpo", "grasa", "ayuno", "ayunas",
    "thinspo", "thinspiration", "proana", "#thinspo", "#thinspiration", "#proana", "#ana", "#mia"
]


def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )


class ManualFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
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
            has_fasting_term = int(any(term in text_lower for term in ["ayuno", "ayunas", "sin comer", "no comer"]))
            has_body_term = int(any(term in text_lower for term in ["abdomen", "cuerpo"]))
            has_hashtag = int("#" in text_lower)

            risk_term_count = sum(1 for term in RISK_TERMS if term in text_lower)

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
                "risk_term_count": risk_term_count
            })

        return pd.DataFrame(features)