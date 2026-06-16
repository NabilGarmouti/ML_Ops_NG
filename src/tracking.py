"""Fonctions de tracking MLflow."""

from __future__ import annotations

import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report

from config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_STAGE
from evaluation import log_shap_summary


def configure_mlflow() -> None:
    """Configure MLflow from project settings."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def log_optimized_model_run(result, x_test, y_test, cv: int, scoring: str) -> None:
    """Log one optimized model as a nested MLflow run."""
    with mlflow.start_run(run_name=result.name, nested=True):
        mlflow.set_tag("model_family", result.name)
        mlflow.set_tag("model_stage", MODEL_STAGE)
        mlflow.set_tag("search_method", "GridSearchCV")
        mlflow.log_param("cv", cv)
        mlflow.log_param("scoring", scoring)
        mlflow.log_params(result.best_params)
        mlflow.log_metrics(
            {
                f"cv_{scoring}": result.cv_score,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "roc_auc": result.roc_auc,
            }
        )
        report = classification_report(
            y_test,
            result.best_estimator.predict(x_test),
            output_dict=True,
        )
        mlflow.log_dict(report, "classification_report.json")
        log_shap_summary(result.best_estimator, x_test, result.name)
        mlflow.sklearn.log_model(result.best_estimator, name="model")


def log_optimized_parent_run_params(
    cv: int,
    scoring: str,
    sample_size: int,
    train_rows: int,
    test_rows: int,
) -> None:
    """Log global params for the optimized model comparison run."""
    mlflow.log_params(
        {
            "cv": cv,
            "scoring": scoring,
            "sample_size": sample_size,
            "model_stage": MODEL_STAGE,
            "train_rows": train_rows,
            "test_rows": test_rows,
        }
    )


def log_optimized_best_model(
    best,
    scoring: str,
    metrics_path,
    confusion_matrix_path,
) -> None:
    """Log the selected optimized model and parent run artifacts."""
    mlflow.set_tag("best_model", best.name)
    mlflow.log_param("registered_model_name", MODEL_NAME)
    mlflow.log_metric(f"best_{scoring}", float(getattr(best, scoring)))
    mlflow.log_artifact(str(metrics_path))
    mlflow.log_artifact(str(confusion_matrix_path))
    mlflow.sklearn.log_model(
        best.best_estimator,
        name="best_model",
        registered_model_name=MODEL_NAME,
    )
