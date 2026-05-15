import pytest
import pandas as pd
from src.data_loader import load_dataset, validate_dataset


#Helpers
def _valid_df(**overrides):
    data = {
        "user_id":    [1],
        "tweet_id":   [10],
        "tweet_text": ["texto de prueba"],
        "class":      ["control"],
    }
    data.update(overrides)
    return pd.DataFrame(data)

#Carga correcta de dataset válido
def test_load_dataset_valid(tmp_path):
    path = tmp_path / "data.xlsx"
    _valid_df().to_excel(path, index=False)
    df = load_dataset(str(path))
    assert not df.empty
    assert "tweet_text" in df.columns
    assert "class" in df.columns

#Error si falta columna tweet_text
def test_validate_missing_tweet_text():
    df = pd.DataFrame({
        "user_id":  [1],
        "tweet_id": [10],
        "class":    ["control"],
    })
    with pytest.raises(ValueError, match="columnas"):
        validate_dataset(df)

#Error si falta columna class
def test_validate_missing_class_column():
    df = pd.DataFrame({
        "user_id":    [1],
        "tweet_id":   [10],
        "tweet_text": ["hola"],
    })
    with pytest.raises(ValueError, match="columnas"):
        validate_dataset(df)

#Error si el dataset está vacío
def test_validate_empty_dataset():
    df = pd.DataFrame(columns=["user_id", "tweet_id", "tweet_text", "class"])
    with pytest.raises(ValueError, match="vacío"):
        validate_dataset(df)

#Error si hay etiquetas inválidas
def test_validate_invalid_labels():
    df = _valid_df(**{"class": ["desconocido"]})
    with pytest.raises(ValueError, match="clases"):
        validate_dataset(df)

#Validación de tipos de datos
def test_validate_text_column_type():
    df = _valid_df()
    assert pd.api.types.is_string_dtype(df["tweet_text"])

#Manejo de valores nulos en tweet_text
def test_validate_all_null_tweet_text():
    df = _valid_df(**{"tweet_text": [None]})
    with pytest.raises(ValueError):
        validate_dataset(df)

#Manejo de archivo inexistente
def test_load_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("ruta/que/no/existe.xlsx")