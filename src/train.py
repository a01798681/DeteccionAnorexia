from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from .preprocessing import clean_text
from .features import build_tfidf_vectorizer
from .evaluate import compute_metrics


def map_labels(series: pd.Series) -> pd.Series:
    return series.map({"control": 0, "anorexia": 1})


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["tweet_text", "class"])
    df["clean_text"] = df["tweet_text"].apply(clean_text)
    df["label"] = map_labels(df["class"])
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


def build_logreg_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])


def build_svm_pipeline() -> Pipeline:
    base_svm = LinearSVC(class_weight="balanced", random_state=42)
    calibrated_svm = CalibratedClassifierCV(base_svm, cv=5)

    return Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        ("clf", calibrated_svm)
    ])


def train_and_evaluate_model(model_name: str, pipeline: Pipeline, X_train, X_val, y_train, y_val):
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)

    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_val)[:, 1]
    else:
        raise ValueError(f"El modelo {model_name} no soporta predict_proba.")

    metrics = compute_metrics(y_val, y_pred, y_score)
    return pipeline, metrics


def run_baseline_experiment(df: pd.DataFrame, output_dir: str = "results"):
    df_prepared = prepare_dataframe(df)

    X = df_prepared["clean_text"]
    y = df_prepared["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    models = {
        "logistic_regression": build_logreg_pipeline(),
        "linear_svm": build_svm_pipeline()
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name, pipeline in models.items():
        trained_model, metrics = train_and_evaluate_model(
            model_name, pipeline, X_train, X_val, y_train, y_val
        )

        all_results[model_name] = metrics

        model_file = output_path / f"{model_name}.joblib"
        results_file = output_path / f"{model_name}_metrics.json"

        joblib.dump(trained_model, model_file)

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    summary = {
        model_name: {
            "roc_auc": metrics["roc_auc"],
            "accuracy": metrics["accuracy"]
        }
        for model_name, metrics in all_results.items()
    }

    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return all_results