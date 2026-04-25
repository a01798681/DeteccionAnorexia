from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from .preprocessing import clean_text
from .features import build_tfidf_vectorizer, ManualFeatureExtractor
from .evaluate import compute_metrics


LABEL_TO_NUM = {
    "control": 0,
    "anorexia": 1
}

NUM_TO_LABEL = {
    0: "control",
    1: "anorexia"
}


def map_labels(series: pd.Series) -> pd.Series:
    return series.map(LABEL_TO_NUM)


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
            max_iter=3000,
            random_state=42
        ))
    ])


def build_hybrid_logreg_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", build_tfidf_vectorizer(), "clean_text"),
            ("manual", Pipeline([
                ("extract", ManualFeatureExtractor()),
                ("scale", StandardScaler())
            ]), "clean_text")
        ]
    )

    return Pipeline([
        ("features", preprocessor),
        ("clf", LogisticRegression(
            max_iter=3000,
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


def optimize_logistic_regression(X_train, y_train):
    pipeline = build_logreg_pipeline()

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 5],
        "tfidf__max_df": [0.95, 1.0],
        "tfidf__sublinear_tf": [True],
        "clf__C": [1.0, 5.0, 10.0],
        "clf__class_weight": [None, "balanced"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    return grid


def optimize_hybrid_logistic_regression(train_df, y_train):
    pipeline = build_hybrid_logreg_pipeline()

    param_grid = {
        "clf__C": [0.1, 1.0, 5.0, 10.0],
        "clf__class_weight": [None, "balanced"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(train_df[["clean_text"]], y_train)
    return grid


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_validation_predictions_dataframe(df_val, y_true, y_pred, y_score):
    result = df_val.copy()
    result["true_label_num"] = y_true
    result["predicted_label_num"] = y_pred
    result["true_label"] = result["true_label_num"].map(NUM_TO_LABEL)
    result["predicted_label"] = result["predicted_label_num"].map(NUM_TO_LABEL)
    result["probability_anorexia"] = y_score
    result["is_correct"] = result["true_label_num"] == result["predicted_label_num"]

    def classify_error(row):
        if row["true_label_num"] == 1 and row["predicted_label_num"] == 1:
            return "true_positive"
        elif row["true_label_num"] == 0 and row["predicted_label_num"] == 0:
            return "true_negative"
        elif row["true_label_num"] == 0 and row["predicted_label_num"] == 1:
            return "false_positive"
        elif row["true_label_num"] == 1 and row["predicted_label_num"] == 0:
            return "false_negative"
        return "unknown"

    result["prediction_type"] = result.apply(classify_error, axis=1)
    result["text_length_chars"] = result["tweet_text"].astype(str).apply(len)
    result["text_length_words"] = result["clean_text"].astype(str).apply(lambda x: len(x.split()))
    return result


def save_validation_analysis_files(validation_df, output_path: Path, prefix: str):
    validation_df.to_csv(output_path / f"{prefix}_validation_predictions.csv", index=False, encoding="utf-8-sig")
    validation_df[validation_df["prediction_type"] == "false_positive"].to_csv(
        output_path / f"{prefix}_false_positives.csv", index=False, encoding="utf-8-sig"
    )
    validation_df[validation_df["prediction_type"] == "false_negative"].to_csv(
        output_path / f"{prefix}_false_negatives.csv", index=False, encoding="utf-8-sig"
    )
    validation_df[validation_df["is_correct"] == False].to_csv(
        output_path / f"{prefix}_incorrect_predictions.csv", index=False, encoding="utf-8-sig"
    )


def evaluate_and_save(model_name, model, X_val, y_val, df_val, output_path):
    y_pred = model.predict(X_val)
    y_score = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_pred, y_score)

    validation_df = build_validation_predictions_dataframe(
        df_val=df_val,
        y_true=y_val.values,
        y_pred=y_pred,
        y_score=y_score
    )
    save_validation_analysis_files(validation_df, output_path, model_name)

    return metrics


def run_baseline_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: str = "results"):
    train_df = prepare_dataframe(train_df)
    val_df = prepare_dataframe(val_df)

    X_train = train_df["clean_text"]
    y_train = train_df["label"]

    X_val = val_df["clean_text"]
    y_val = val_df["label"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # 1. Logistic Regression optimizada
    grid = optimize_logistic_regression(X_train, y_train)
    best_logreg_model = grid.best_estimator_

    logreg_metrics = evaluate_and_save(
        "logistic_regression_optimized",
        best_logreg_model,
        X_val,
        y_val,
        val_df,
        output_path
    )

    logreg_results = {
        **logreg_metrics,
        "best_params": grid.best_params_,
        "best_cv_score": grid.best_score_
    }

    all_results["logistic_regression_optimized"] = logreg_results

    joblib.dump(best_logreg_model, output_path / "logistic_regression_optimized.joblib")
    save_json(logreg_results, output_path / "logistic_regression_optimized_metrics.json")

    # 2. Logistic Regression híbrida
    hybrid_grid = optimize_hybrid_logistic_regression(train_df, y_train)
    best_hybrid_model = hybrid_grid.best_estimator_

    hybrid_metrics = evaluate_and_save(
        "logistic_regression_hybrid",
        best_hybrid_model,
        val_df[["clean_text"]],
        y_val,
        val_df,
        output_path
    )

    hybrid_results = {
        **hybrid_metrics,
        "best_params": hybrid_grid.best_params_,
        "best_cv_score": hybrid_grid.best_score_
    }

    all_results["logistic_regression_hybrid"] = hybrid_results

    joblib.dump(best_hybrid_model, output_path / "logistic_regression_hybrid.joblib")
    save_json(hybrid_results, output_path / "logistic_regression_hybrid_metrics.json")

    # 3. Linear SVM
    svm_pipeline = build_svm_pipeline()
    svm_pipeline.fit(X_train, y_train)

    svm_metrics = evaluate_and_save(
        "linear_svm",
        svm_pipeline,
        X_val,
        y_val,
        val_df,
        output_path
    )

    all_results["linear_svm"] = svm_metrics

    joblib.dump(svm_pipeline, output_path / "linear_svm.joblib")
    save_json(svm_metrics, output_path / "linear_svm_metrics.json")

    # 4. Resumen general
    summary = {
        model_name: {
            "roc_auc": metrics["roc_auc"],
            "accuracy": metrics["accuracy"]
        }
        for model_name, metrics in all_results.items()
    }

    save_json(summary, output_path / "summary.json")

    return all_results