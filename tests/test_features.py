import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.features import (
    ManualFeatureExtractor,
    build_tfidf_vectorizer,
    RISK_TERMS,
)

#Helper
def _extract(texts):
    extractor = ManualFeatureExtractor()
    return extractor.transform(pd.Series(texts))

#Genera matriz TF-IDF sin error
def test_tfidf_vectorizer_fits_and_transforms():
    corpus = [
        "quiero ser flaca y dejar de comer",
        "me siento bien con mi cuerpo",
        "thinspo proana ayuno extremo",
        "salí con mis amigos hoy",
    ]
    tfidf = build_tfidf_vectorizer()
    X = tfidf.fit_transform(corpus)
    assert X.shape[0] == 4
    assert X.shape[1] > 0

#Genera el número esperado de columnas manuales
def test_manual_extractor_expected_number_of_columns():
    features = _extract(["hola mundo"])
    assert len(features.columns) == 18  # según features.py

#Detecta términos de restricción alimentaria
def test_detects_fasting_terms():
    features = _extract(["llevo tres días en ayunas sin comer"])
    assert features.iloc[0]["has_fasting_term"] == 1

#Detecta términos de ayuno
def test_detects_ayuno():
    features = _extract(["hice ayuno todo el día"])
    assert features.iloc[0]["has_fasting_term"] == 1

#Detecta términos de vómito/purga
def test_detects_vomit_purge_terms():
    features = _extract(["después de comer vomitar todo"])
    assert features.iloc[0]["has_vomit_term"] == 1

#Detecta términos de peso/cuerpo
def test_detects_weight_body_terms():
    features = _extract(["quiero bajar de peso y ser más flaca"])
    row = features.iloc[0]
    assert row["has_weight_term"] == 1

#Detecta hashtags de riesgo
def test_detects_risk_hashtags():
    features = _extract(["#thinspo #proana #ana"])
    row = features.iloc[0]
    assert row["has_thinspo"] == 1
    assert row["has_proana"] == 1
    assert row["has_hashtag"] == 1

#Texto neutro no activa atributos de riesgo
def test_neutral_text_no_risk_flags():
    features = _extract(["hoy fui al cine con amigos, estuvo genial"])
    row = features.iloc[0]
    assert row["has_thinspo"] == 0
    assert row["has_vomit_term"] == 0
    assert row["has_fasting_term"] == 0
    assert row["risk_term_count"] == 0

#Combinación TF-IDF + manuales conserva forma esperada
def test_combined_tfidf_manual_shape():
    corpus = [
        "quiero ser flaca dejar de comer #thinspo",
        "me siento bien con mi cuerpo estoy tranquila",
        "ayuno purga vomitar peso grasa",
        "salí a cenar con mi familia fue genial",
    ]
    df = pd.DataFrame({"clean_text": corpus})

    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", build_tfidf_vectorizer(), "clean_text"),
            ("manual", ManualFeatureExtractor(), "clean_text"),
        ]
    )
    X = preprocessor.fit_transform(df)
    assert X.shape[0] == 4
    assert X.shape[1] > 18  # TF-IDF + 18 manuales

#Salida consistente entre varias ejecuciones con mismos datos
def test_manual_extractor_deterministic():
    texts = pd.Series(["quiero ser flaca #thinspo ayuno"])
    extractor = ManualFeatureExtractor()
    result1 = extractor.transform(texts)
    result2 = extractor.transform(texts)
    pd.testing.assert_frame_equal(result1, result2)