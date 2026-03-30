import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


RISK_TERMS = [
    "vomit", "vomitar", "vomitando", "purga", "purging",
    "adelgazar", "bajar de peso", "peso", "gorda", "flaca",
    "abdomen", "cuerpo", "grasa", "ayuno", "ayunas",
    "thinspo", "thinspiration", "proana", "anorexia", "bulimia",
    "#thinspo", "#thinspiration", "#proana", "#ana", "#mia"
]

FIRST_PERSON_TERMS = [
    "yo", "me", "mi", "mí", "estoy", "quiero", "siento", "tengo", "soy", "voy"
]

INFORMATIONAL_TERMS = [
    "trastorno", "trastornos", "enfermedad", "salud", "prevencion",
    "prevención", "concientizacion", "concientización", "fundacion",
    "fundación", "informacion", "información", "bulimia", "anorexia"
]

DIET_FITNESS_TERMS = [
    "dieta", "dietafit", "fit", "perderpeso", "perder peso",
    "abdomen plano", "ejercicio", "fitness", "saludable"
]

NEGATION_OR_RESTRICTION_TERMS = [
    "no comer", "sin comer", "ayuno", "ayunas", "dejar de comer",
    "no quiero comer", "quiero ser flaca", "me siento gorda"
]


def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )


def extract_hashtags(text: str):
    return re.findall(r"#\w+", text.lower())


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
            hashtags = extract_hashtags(text_lower)

            has_thinspo = int(any("thinspo" in h for h in hashtags) or "thinspo" in text_lower)
            has_thinspiration = int(any("thinspiration" in h for h in hashtags) or "thinspiration" in text_lower)
            has_proana = int(any(("proana" in h or h == "#ana") for h in hashtags) or "proana" in text_lower)
            has_anorexia_tag = int(any("anorexia" in h for h in hashtags) or "anorexia" in text_lower)
            has_bulimia_tag = int(any("bulimia" in h for h in hashtags) or "bulimia" in text_lower)

            has_vomit_term = contains_any(text_lower, ["vomit", "vomitar", "vomitando", "purga", "purging"])
            has_weight_term = contains_any(text_lower, ["peso", "gorda", "flaca", "adelgazar", "bajar de peso", "grasa"])
            has_fasting_term = contains_any(text_lower, ["ayuno", "ayunas", "sin comer", "no comer", "dejar de comer"])
            has_body_term = contains_any(text_lower, ["abdomen", "cuerpo"])
            has_hashtag = int("#" in text_lower)

            risk_term_count = count_any(text_lower, RISK_TERMS)

            first_person_count = count_any(text_lower, FIRST_PERSON_TERMS)
            has_first_person = int(first_person_count > 0)

            informational_term_count = count_any(text_lower, INFORMATIONAL_TERMS)
            has_informational_terms = int(informational_term_count > 0)

            diet_fitness_term_count = count_any(text_lower, DIET_FITNESS_TERMS)
            has_diet_fitness_terms = int(diet_fitness_term_count > 0)

            restriction_term_count = count_any(text_lower, NEGATION_OR_RESTRICTION_TERMS)
            has_restriction_terms = int(restriction_term_count > 0)

            # combinaciones útiles
            first_person_and_risk = int(has_first_person and risk_term_count > 0)
            informational_without_first_person = int(has_informational_terms and not has_first_person)
            hashtag_risk_count = sum(
                1 for h in hashtags
                if any(term in h for term in ["thinspo", "thinspiration", "proana", "ana", "mia", "anorexia", "bulimia", "perderpeso"])
            )

            features.append({
                "text_length_chars": len(text_lower),
                "text_length_words": len(words),
                "has_thinspo": has_thinspo,
                "has_thinspiration": has_thinspiration,
                "has_proana": has_proana,
                "has_anorexia_tag": has_anorexia_tag,
                "has_bulimia_tag": has_bulimia_tag,
                "has_vomit_term": has_vomit_term,
                "has_weight_term": has_weight_term,
                "has_fasting_term": has_fasting_term,
                "has_body_term": has_body_term,
                "has_hashtag": has_hashtag,
                "risk_term_count": risk_term_count,
                "first_person_count": first_person_count,
                "has_first_person": has_first_person,
                "informational_term_count": informational_term_count,
                "has_informational_terms": has_informational_terms,
                "diet_fitness_term_count": diet_fitness_term_count,
                "has_diet_fitness_terms": has_diet_fitness_terms,
                "restriction_term_count": restriction_term_count,
                "has_restriction_terms": has_restriction_terms,
                "first_person_and_risk": first_person_and_risk,
                "informational_without_first_person": informational_without_first_person,
                "hashtag_risk_count": hashtag_risk_count
            })

        return pd.DataFrame(features)