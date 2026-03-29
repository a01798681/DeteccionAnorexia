from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Vectorizador base para el baseline.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )