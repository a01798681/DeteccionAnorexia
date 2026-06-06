# Author: Andrés Cabrera Alvarado - A01798681
# Fecha de creación: 05/06/2026
# Archivo: tests/test_features.py
# Descripción general: Pruebas unitarias para la extracción de características (features.py).
# Verifica que el extractor manual cuente correctamente los términos de riesgo (ayuno, purga, hashtags) 
# y que la vectorización TF-IDF funcione como se espera.

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.features import (
    ManualFeatureExtractor,
    build_tfidf_vectorizer,
)

# Helper que inicializa el extractor manual y procesa una lista de textos.
def _extract(texts):
    extractor = ManualFeatureExtractor()
    return extractor.transform(pd.Series(texts))

# Verifica que el vectorizador TF-IDF se ajuste correctamente a un corpus
# de prueba y genere una matriz dispersa válida sin errores.
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

# Comprueba que el extractor manual genere exactamente el número de columnas de características definidas.
def test_manual_extractor_expected_number_of_columns():
    features = _extract(["hola mundo"])
    assert len(features.columns) == 18  # según features.py

# Asegura que el extractor marque correctamente la bandera de "ayuno"
# (has_fasting_term) cuando el texto habla de restricción alimentaria o ayunas.
def test_detects_fasting_terms():
    features = _extract(["llevo tres días en ayunas sin comer"])
    assert features.iloc[0]["has_fasting_term"] == 1

# Asegura que el extractor detecte explícitamente la palabra "ayuno".
def test_detects_ayuno():
    features = _extract(["hice ayuno todo el día"])
    assert features.iloc[0]["has_fasting_term"] == 1

# Asegura que el extractor marque la bandera de vómito/purga (has_vomit_term)
# cuando aparecen los términos correspondientes.
def test_detects_vomit_purge_terms():
    features = _extract(["después de comer vomitar todo"])
    assert features.iloc[0]["has_vomit_term"] == 1

# Verifica que se detecten los términos relacionados con el peso o el cuerpo (has_weight_term).
def test_detects_weight_body_terms():
    features = _extract(["quiero bajar de peso y ser más flaca"])
    row = features.iloc[0]
    assert row["has_weight_term"] == 1

# Comprueba que el extractor detecte y marque hashtags específicos de riesgo
# (#thinspo, #proana) y active la bandera general de presencia de hashtags.
def test_detects_risk_hashtags():
    features = _extract(["#thinspo #proana #ana"])
    row = features.iloc[0]
    assert row["has_thinspo"] == 1
    assert row["has_proana"] == 1
    assert row["has_hashtag"] == 1

# Verifica que un texto completamente seguro/neutral no active ninguna
# bandera de riesgo ni cuente términos sospechosos.
def test_neutral_text_no_risk_flags():
    features = _extract(["hoy fui al cine con amigos, estuvo genial"])
    row = features.iloc[0]
    assert row["has_thinspo"] == 0
    assert row["has_vomit_term"] == 0
    assert row["has_fasting_term"] == 0
    assert row["risk_term_count"] == 0

# Asegura que el ColumnTransformer combine correctamente las columnas generadas
# por TF-IDF y el extractor manual, manteniendo una forma bidimensional válida.
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

# Comprueba que la salida del extractor manual sea consistente y determinista
# al procesar el mismo texto en ejecuciones consecutivas.
def test_manual_extractor_deterministic():
    texts = pd.Series(["quiero ser flaca #thinspo ayuno"])
    extractor = ManualFeatureExtractor()
    result1 = extractor.transform(texts)
    result2 = extractor.transform(texts)
    pd.testing.assert_frame_equal(result1, result2)