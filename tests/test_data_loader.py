import pandas as pd
import pytest
from src.data_loader import validate_dataset


def test_validate_dataset_ok():
    df = pd.DataFrame({
        "user_id": [1],
        "tweet_id": [10],
        "tweet_text": ["texto de prueba"],
        "class": ["control"]
    })
    validate_dataset(df)


def test_validate_dataset_missing_columns():
    df = pd.DataFrame({
        "tweet_text": ["texto de prueba"],
        "class": ["control"]
    })

    with pytest.raises(ValueError):
        validate_dataset(df)