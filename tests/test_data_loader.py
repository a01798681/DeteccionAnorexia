# Author: Andrés Cabrera Alvarado - A01798681
# Author: Andrea Elizabeth Roman Varela - A01749760
# Author: Pablo Alonso Galván - A01748288
# Fecha de creación: 05/06/2026
# Archivo: tests/test_data_loader.py
# Descripción general: Pruebas unitarias para el módulo data_loader.py.
# Verifica la carga de archivos Excel, validación de columnas obligatorias, manejo de datos vacíos, 
# etiquetas inválidas y errores de ruta.

import pytest
import pandas as pd
from src.data_loader import load_dataset, validate_dataset


# Helper que genera un DataFrame válido con datos de prueba básicos.
# Permite sobrescribir valores específicos usando **overrides.
def _valid_df(**overrides):
    data = {
        "user_id":    [1],
        "tweet_id":   [10],
        "tweet_text": ["texto de prueba"],
        "class":      ["control"],
    }
    data.update(overrides)
    return pd.DataFrame(data)

# Verifica la carga correcta de un dataset válido desde un archivo temporal Excel.
def test_load_dataset_valid(tmp_path):
    path = tmp_path / "data.xlsx"
    _valid_df().to_excel(path, index=False)
    df = load_dataset(str(path))
    assert not df.empty
    assert "tweet_text" in df.columns
    assert "class" in df.columns

# Comprueba que lance un error (ValueError) si falta la columna 'tweet_text'.
def test_validate_missing_tweet_text():
    df = pd.DataFrame({
        "user_id":  [1],
        "tweet_id": [10],
        "class":    ["control"],
    })
    with pytest.raises(ValueError, match="columnas"):
        validate_dataset(df)

# Comprueba que lance un error (ValueError) si falta la columna 'class'.
def test_validate_missing_class_column():
    df = pd.DataFrame({
        "user_id":    [1],
        "tweet_id":   [10],
        "tweet_text": ["hola"],
    })
    with pytest.raises(ValueError, match="columnas"):
        validate_dataset(df)

# Verifica que se lance un error si el dataset está completamente vacío.
def test_validate_empty_dataset():
    df = pd.DataFrame(columns=["user_id", "tweet_id", "tweet_text", "class"])
    with pytest.raises(ValueError, match="vacío"):
        validate_dataset(df)

# Asegura que las etiquetas (class) que no sean 'control' o 'anorexia' lancen un error.
def test_validate_invalid_labels():
    df = _valid_df(**{"class": ["desconocido"]})
    with pytest.raises(ValueError, match="clases"):
        validate_dataset(df)

# Valida que la columna de texto mantenga el tipo correcto (string).
def test_validate_text_column_type():
    df = _valid_df()
    assert pd.api.types.is_string_dtype(df["tweet_text"])

# Asegura el manejo correcto (lanzando error) de valores nulos en el texto.
def test_validate_all_null_tweet_text():
    df = _valid_df(**{"tweet_text": [None]})
    with pytest.raises(ValueError):
        validate_dataset(df)

# Verifica el manejo de errores (FileNotFoundError) al pasar una ruta de archivo inexistente.
def test_load_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("ruta/que/no/existe.xlsx")