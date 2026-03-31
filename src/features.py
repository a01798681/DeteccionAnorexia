import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


RISK_TERMS = [
    "vomit", "vomitar", "vomitando", "purga", "purging",
    "adelgazar", "bajar de peso", "peso", "gorda", "flaca",
    "abdomen", "cuerpo", "grasa", "ayuno", "ayunas",
    "thinspo", "thinspiration", "proana", "#thinspo", "#thinspiration",
    "#proana", "#ana", "#mia", "dejar de comer", "no quiero comer",
    "quiero ser flaca", "me siento gorda"
]

POSITIVE_SAFE_TERMS = [
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

NEGATION_SAFE_TERMS = [
    "no tengo problema",
    "no tengo problema con mi cuerpo",
    "no quiero dejar de comer",
    "sí como",
    "si como",
    "como normal",
    "comí normal",
    "comi normal"
]


def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )


def contains_any(text: str, terms) -> int:
    text = text.lower()
    return int(any(term in text for term in terms))


def count_any(text: str, terms) -> int:
    text = text.lower()
    return sum(1 for term in terms if term in text)


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
            has_fasting_term = int(any(term in text_lower for term in ["ayuno", "ayunas", "sin comer", "no comer", "dejar de comer"]))
            has_body_term = int(any(term in text_lower for term in ["abdomen", "cuerpo"]))
            has_hashtag = int("#" in text_lower)

            risk_term_count = count_any(text_lower, RISK_TERMS)
            positive_safe_count = count_any(text_lower, POSITIVE_SAFE_TERMS)
            negation_safe_count = count_any(text_lower, NEGATION_SAFE_TERMS)

            has_positive_safe = int(positive_safe_count > 0)
            has_negation_safe = int(negation_safe_count > 0)

            # combinaciones útiles
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