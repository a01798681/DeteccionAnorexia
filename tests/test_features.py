import pandas as pd
from src.features import ManualFeatureExtractor


def test_manual_feature_extractor_returns_dataframe():
    extractor = ManualFeatureExtractor()
    X = pd.Series(["quiero ser flaca y dejar de comer"])
    features = extractor.transform(X)

    assert isinstance(features, pd.DataFrame)
    assert len(features) == 1


def test_manual_feature_extractor_expected_columns():
    extractor = ManualFeatureExtractor()
    X = pd.Series(["quiero ser flaca y dejar de comer #thinspo"])
    features = extractor.transform(X)

    expected_columns = {
        "text_length_chars",
        "text_length_words",
        "has_thinspo",
        "has_thinspiration",
        "has_proana",
        "has_vomit_term",
        "has_weight_term",
        "has_fasting_term",
        "has_body_term",
        "has_hashtag",
        "risk_term_count"
    }

    assert expected_columns.issubset(set(features.columns))


def test_manual_feature_extractor_detects_risk_terms():
    extractor = ManualFeatureExtractor()
    X = pd.Series(["quiero ser flaca y dejar de comer #thinspo"])
    features = extractor.transform(X)

    row = features.iloc[0]
    assert row["has_thinspo"] == 1
    assert row["has_weight_term"] == 1
    assert row["risk_term_count"] >= 1