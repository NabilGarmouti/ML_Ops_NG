"""Entrainement et comparaison de plusieurs modeles de classification."""

from __future__ import annotations

import argparse
import tempfile
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, MODEL_DIR, MODEL_STAGE, RANDOM_STATE
from data import load_data, split
from features import build_preprocessor, clean_data, validate_clean_data

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


def build_models(c: float = 1.0, max_iter: int = 1000) -> dict[str, Pipeline]:
    """Build the five candidate models used for the first benchmark."""
    candidates = {
        "logistic_regression": LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=12,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            eval_metric="logloss",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "lightgbm": LGBMClassifier(
            class_weight="balanced",
            n_estimators=100,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }

    return {
        name: Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("clf", estimator),
            ]
        )
        for name, estimator in candidates.items()
    }


def _predict_proba(model: Pipeline, x_test) -> pd.Series:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    scores = model.decision_function(x_test)
    return (scores - scores.min()) / (scores.max() - scores.min())


def evaluate_model(name: str, model: Pipeline, x_test, y_test) -> dict[str, float | str]:
    """Evaluate a fitted model with classification metrics."""
    proba = _predict_proba(model, x_test)
    preds = (proba >= 0.5).astype(int)

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def save_confusion_matrix(model: Pipeline, x_test, y_test, output_path: Path) -> None:
    """Save the confusion matrix for the selected model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds = model.predict(x_test)
    display = ConfusionMatrixDisplay.from_predictions(y_test, preds)
    display.figure_.tight_layout()
    display.figure_.savefig(output_path)
    plt.close(display.figure_)


METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def configure_mlflow() -> None:
    """Configure MLflow tracking from project settings."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def _estimator_params(model: Pipeline) -> dict[str, str | int | float | bool | None]:
    estimator = model.named_steps["clf"]
    params = estimator.get_params()
    return {
        key: value
        for key, value in params.items()
        if isinstance(value, str | int | float | bool) or value is None
    }


def train(
    c: float = 1.0,
    max_iter: int = 1000,
    sample_size: int = 0,
    selection_metric: str = "f1",
) -> pd.DataFrame:
    """Train five models and save the best one according to the selected metric."""
    if selection_metric not in METRICS:
        raise ValueError(f"selection_metric doit etre dans {METRICS}")

    configure_mlflow()

    raw_df = load_data()
    df = clean_data(raw_df)
    validate_clean_data(df)

    if sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    x_train, x_test, y_train, y_test = split(df)
    models = build_models(c=c, max_iter=max_iter)

    results: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}

    with mlflow.start_run(run_name="baseline_benchmark"):
        mlflow.log_params(
            {
                "sample_size": sample_size,
                "selection_metric": selection_metric,
                "model_stage": MODEL_STAGE,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "positive_rate": float(df["Response"].mean()),
            }
        )

        for name, model in models.items():
            print(f">> Entrainement: {name}")
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_param("model_name", name)
                mlflow.log_params(_estimator_params(model))

                model.fit(x_train, y_train)
                metrics = evaluate_model(name, model, x_test, y_test)
                mlflow.log_metrics({key: float(metrics[key]) for key in METRICS})
                mlflow.sklearn.log_model(model, name="model")

                results.append(metrics)
                fitted_models[name] = model
                print(
                    f"{name}: "
                    f"f1={metrics['f1']:.3f} "
                    f"roc_auc={metrics['roc_auc']:.3f} "
                    f"recall={metrics['recall']:.3f}"
                )

        results_df = pd.DataFrame(results).sort_values(selection_metric, ascending=False)
        best_name = str(results_df.iloc[0]["model"])
        best_model = fitted_models[best_name]

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = MODEL_DIR / "metrics.csv"
        confusion_matrix_path = MODEL_DIR / "confusion_matrix.png"
        model_path = MODEL_DIR / "model.joblib"

        results_df.to_csv(metrics_path, index=False)
        joblib.dump(best_model, model_path)
        save_confusion_matrix(best_model, x_test, y_test, confusion_matrix_path)

        mlflow.log_param("best_model", best_name)
        mlflow.log_metric(f"best_{selection_metric}", float(results_df.iloc[0][selection_metric]))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(confusion_matrix_path))
        with tempfile.TemporaryDirectory() as tmp_dir:
            best_model_path = Path(tmp_dir) / "best_model.joblib"
            joblib.dump(best_model, best_model_path)
            mlflow.log_artifact(str(best_model_path), artifact_path="best_model")

    print(f">> Meilleur modele ({selection_metric}): {best_name}")
    print(results_df.to_string(index=False))
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Nombre de lignes a echantillonner pour un benchmark rapide. 0 = dataset complet.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=METRICS,
        default="f1",
        help="Metrique utilisee pour choisir le modele sauvegarde.",
    )
    args = parser.parse_args()
    train(
        c=args.c,
        max_iter=args.max_iter,
        sample_size=args.sample_size,
        selection_metric=args.selection_metric,
    )


if __name__ == "__main__":
    main()
