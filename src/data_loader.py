from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {"user_id", "tweet_id", "tweet_text", "class"}


def load_dataset(path: str) -> pd.DataFrame:
    """
    Carga el dataset desde Excel.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_excel(file_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Verifica estructura mínima del dataset.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    if df.empty:
        raise ValueError("El dataset está vacío.")

    if df["tweet_text"].isna().all():
        raise ValueError("La columna tweet_text está completamente vacía.")

    valid_classes = {"anorexia", "control"}
    found_classes = set(df["class"].dropna().unique())

    if not found_classes.issubset(valid_classes):
        raise ValueError(
            f"Se encontraron clases no esperadas: {found_classes - valid_classes}"
        )