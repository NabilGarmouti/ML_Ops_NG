"""Evaluation automatisee et validation du modele enregistre dans MLflow."""

from __future__ import annotations

import argparse
import logging

import mlflow
import mlflow.models
from mlflow.exceptions import MlflowException
from mlflow.models import MetricThreshold

from config import EVAL_F1_MIN, EVAL_ROC_AUC_MIN, MODEL_NAME, TARGET
from data import load_data, split
from tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def latest_model_uri() -> str:
    """Return the latest registered MLflow model URI."""
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"Aucune version enregistree pour {MODEL_NAME!r}. "
            "Lance d'abord train_models.py ou train_optuna.py."
        )

    latest = max(versions, key=lambda version: int(version.version))
    return f"models:/{MODEL_NAME}/{latest.version}"


def build_thresholds() -> dict[str, MetricThreshold]:
    """Build MLflow evaluation thresholds from project configuration."""
    return {
        "roc_auc": MetricThreshold(threshold=EVAL_ROC_AUC_MIN, greater_is_better=True),
        "f1_score": MetricThreshold(threshold=EVAL_F1_MIN, greater_is_better=True),
    }


def evaluate_model(model_uri: str | None = None, validate: bool = True):
    """Evaluate a registered model and optionally enforce quality thresholds."""
    raw_df = load_data()
    _, x_test, _, y_test = split(raw_df)
    eval_df = x_test.copy()
    eval_df[TARGET] = y_test.values

    setup_experiment()
    model_uri = model_uri or latest_model_uri()
    logger.info("Evaluation de %s", model_uri)

    with mlflow.start_run(run_name="evaluate"):
        log_dataset(eval_df, context="evaluation", name="eval")
        result = mlflow.models.evaluate(
            model_uri,
            data=eval_df,
            targets=TARGET,
            model_type="classifier",
            evaluators=["default"],
        )
        logger.info(
            "f1_score=%.3f roc_auc=%.3f",
            result.metrics.get("f1_score", 0.0),
            result.metrics.get("roc_auc", 0.0),
        )

        if validate:
            mlflow.validate_evaluation_results(
                validation_thresholds=build_thresholds(),
                candidate_result=result,
            )

        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-uri", default=None)
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    args = parser.parse_args()

    try:
        evaluate_model(model_uri=args.model_uri, validate=args.validate)
    except MlflowException as exc:
        logger.error("Validation echouee : %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
