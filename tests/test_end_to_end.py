"""
Pruebas end-to-end ligeras: validan el flujo completo de main.py
usando datasets pequeños en memoria / tmp_path, sin tocar el disco real.
"""
import json
import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import load_dataset, validate_dataset
from src.train import run_baseline_experiment, prepare_dataframe


#Dataset mínimo suficiente para GridSearchCV con 5-fold
#Necesitamos al menos 10 muestras por clase (5 folds × 2 clases)
def _make_mini_xlsx(tmp_path, name="data.xlsx", n_per_class=10):
    anorexia_texts = [
        "quiero ser flaca y dejar de comer ayuno purga vomitar",
        "thinspo proana quiero bajar de peso grasa cuerpo flaca",
        "me siento gorda no quiero comer nada hoy #thinspo",
        "ayuno todo el día para perder peso y ser delgada",
        "me siento horrible con mi cuerpo quiero vomitar todo",
        "dejar de comer es la única solución para ser flaca",
        "#ana #mia quiero ser tan delgada como ella thinspo",
        "purga ayuno restricción no como nada desde ayer peso",
        "me siento gorda y horrible quiero dejar de comer grasa",
        "vomitando todo lo que como para bajar de peso flaca",
    ]
    control_texts = [
        "hoy salí con mis amigos a comer pizza fue genial",
        "me siento bien y tranquila disfruté mucho la comida",
        "fui al gimnasio y luego comí con mi familia contenta",
        "estoy muy feliz hoy el día estuvo increíble con amigos",
        "comí rico con mis amigos hoy todo estuvo muy bien",
        "me siento bien conmigo misma estoy tranquila y feliz",
        "disfruté mucho la cena familiar todo estuvo delicioso",
        "salí a correr y luego comí saludable me siento genial",
        "buen día con amigos comida rica y muchas risas feliz",
        "todo bien hoy descansé bien y comí rico con familia",
    ]
    rows = (
        [{"user_id": i, "tweet_id": i, "tweet_text": t, "class": "anorexia"}
         for i, t in enumerate(anorexia_texts[:n_per_class])]
        + [{"user_id": i + 100, "tweet_id": i + 100, "tweet_text": t, "class": "control"}
           for i, t in enumerate(control_texts[:n_per_class])]
    )
    df = pd.DataFrame(rows)
    path = tmp_path / name
    df.to_excel(path, index=False)
    return path, df


#main.py corre sin excepciones con dataset pequeño
def test_run_experiment_no_exception(tmp_path):
    _, train_df = _make_mini_xlsx(tmp_path, "train.xlsx")
    _, val_df   = _make_mini_xlsx(tmp_path, "val.xlsx")
    results = run_baseline_experiment(train_df, val_df, output_dir=str(tmp_path / "results"))
    assert isinstance(results, dict)


#Genera carpeta results/
def test_results_folder_created(tmp_path):
    _, train_df = _make_mini_xlsx(tmp_path, "train.xlsx")
    _, val_df   = _make_mini_xlsx(tmp_path, "val.xlsx")
    output_dir = tmp_path / "results"
    run_baseline_experiment(train_df, val_df, output_dir=str(output_dir))
    assert output_dir.exists()

#Guarda archivos de métricas
def test_metrics_files_saved(tmp_path):
    _, train_df = _make_mini_xlsx(tmp_path, "train.xlsx")
    _, val_df   = _make_mini_xlsx(tmp_path, "val.xlsx")
    output_dir = tmp_path / "results"
    run_baseline_experiment(train_df, val_df, output_dir=str(output_dir))
    json_files = list(output_dir.glob("*_metrics.json"))
    assert len(json_files) >= 1

#Guarda modelo entrenado (.joblib)
def test_model_files_saved(tmp_path):
    _, train_df = _make_mini_xlsx(tmp_path, "train.xlsx")
    _, val_df   = _make_mini_xlsx(tmp_path, "val.xlsx")
    output_dir = tmp_path / "results"
    run_baseline_experiment(train_df, val_df, output_dir=str(output_dir))
    joblib_files = list(output_dir.glob("*.joblib"))
    assert len(joblib_files) >= 1

#Produce salida con llaves esperadas
def test_results_have_expected_keys(tmp_path):
    _, train_df = _make_mini_xlsx(tmp_path, "train.xlsx")
    _, val_df   = _make_mini_xlsx(tmp_path, "val.xlsx")
    output_dir = tmp_path / "results"
    results = run_baseline_experiment(train_df, val_df, output_dir=str(output_dir))
    for model_name, metrics in results.items():
        assert "roc_auc"   in metrics, f"Falta 'roc_auc' en {model_name}"
        assert "accuracy"  in metrics, f"Falta 'accuracy' en {model_name}"
        assert "confusion_matrix" in metrics, f"Falta 'confusion_matrix' en {model_name}"