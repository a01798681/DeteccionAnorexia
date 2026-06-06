# Author: Andrés Cabrera Alvarado - A01798681
# Fecha de creación: 05/06/2026
# Archivo: tests/test_model_registry.py
# Descripción general: Pruebas unitarias para verificar el funcionamiento del registro de modelos (model_registry.py). 
# Asegura que los modelos estén disponibles y sus configuraciones sean correctas.

from src.model_registry import get_available_models, get_model_config, get_default_model_key


# Verifica que get_available_models retorne una lista no vacía.
def test_get_available_models_returns_list():
    models = get_available_models()
    assert isinstance(models, list)
    assert len(models) >= 1


# Comprueba que los nuevos modelos híbridos (ensemble y cascade) estén registrados.
def test_registry_contains_new_combo_models():
    models = get_available_models()
    keys = {m["key"] for m in models}
    assert "beto_llm_ensemble" in keys
    assert "beto_llm_cascade" in keys


# Verifica que se pueda obtener la configuración correcta de un modelo usando su clave.
def test_get_model_config_by_key():
    config = get_model_config("beto_llm_ensemble")
    assert config["label"] == "BETO + LLM ensemble"


# Comprueba que la clave del modelo por defecto exista y sea válida (cadena o None).
def test_default_model_key_exists():
    default_key = get_default_model_key()
    assert default_key is None or isinstance(default_key, str)