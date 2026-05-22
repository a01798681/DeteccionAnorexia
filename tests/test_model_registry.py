from src.model_registry import get_available_models, get_model_config


def test_get_available_models_returns_list():
    models = get_available_models()
    assert isinstance(models, list)
    assert len(models) >= 1


def test_registered_models_have_required_keys():
    models = get_available_models()
    required_keys = {"key", "label", "short_label", "type", "family", "path", "description", "exists"}

    for model in models:
        assert required_keys.issubset(model.keys())


def test_get_model_config_by_key():
    config = get_model_config("hybrid_logreg")
    assert config["label"] == "Logistic Regression híbrida"