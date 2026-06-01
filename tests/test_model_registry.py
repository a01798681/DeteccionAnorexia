from src.model_registry import get_available_models, get_model_config, get_default_model_key


def test_get_available_models_returns_list():
    models = get_available_models()
    assert isinstance(models, list)
    assert len(models) >= 1


def test_registry_contains_new_combo_models():
    models = get_available_models()
    keys = {m["key"] for m in models}
    assert "beto_llm_ensemble" in keys
    assert "beto_llm_cascade" in keys


def test_get_model_config_by_key():
    config = get_model_config("beto_llm_ensemble")
    assert config["label"] == "BETO + LLM ensemble"


def test_default_model_key_exists():
    default_key = get_default_model_key()
    assert default_key is None or isinstance(default_key, str)