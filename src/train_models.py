"""Optimisation de trois familles de modeles avec GridSearchCV et MLflow."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODEL_NAME,
    MODEL_STAGE,
    RANDOM_STATE,
)
from data import load_data, split
from evaluation import log_shap_summary
from features import build_preprocessor, clean_data, validate_clean_data

SUPPORTED_SCORING = ["f1", "roc_auc"]

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


@dataclass
class ModelSpec:
    """Specification d'une famille de modele a optimiser."""

    name: str
    estimator: object
    param_grid: dict[str, list]


@dataclass
class FitResult:
    """Resultat d'optimisation et d'evaluation d'un modele."""

    name: str
    best_estimator: Pipeline
    best_params: dict
    cv_score: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def configure_mlflow() -> None:
    """Configure MLflow from project settings."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def build_model_specs() -> list[ModelSpec]:
    """Build the three optimized model families required for this step."""
    return [
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [10, 20],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
        ModelSpec(
            name="xgboost",
            estimator=XGBClassifier(
                eval_metric="logloss",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.1, 0.03],
            },
        ),
        ModelSpec(
            name="lightgbm",
            estimator=LGBMClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__num_leaves": [31, 63],
                "clf__learning_rate": [0.1, 0.03],
            },
        ),
    ]


def build_pipeline(estimator: object) -> Pipeline:
    """Build a preprocessing + classifier pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", estimator),
        ]
    )


def optimize_model(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    cv: int,
    scoring: str,
) -> FitResult:
    """Optimize one model family with GridSearchCV and evaluate it on the test set."""
    search = GridSearchCV(
        estimator=build_pipeline(spec.estimator),
        param_grid=spec.param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)

    best = search.best_estimator_
    proba = best.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return FitResult(
        name=spec.name,
        best_estimator=best,
        best_params=search.best_params_,
        cv_score=float(search.best_score_),
        precision=float(precision_score(y_test, preds, zero_division=0)),
        recall=float(recall_score(y_test, preds, zero_division=0)),
        f1=float(f1_score(y_test, preds, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, proba)),
    )


def save_confusion_matrix(model: Pipeline, x_test, y_test, output_path: Path) -> None:
    """Save the confusion matrix for the selected model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds = model.predict(x_test)
    display = ConfusionMatrixDisplay.from_predictions(y_test, preds)
    display.figure_.tight_layout()
    display.figure_.savefig(output_path)
    plt.close(display.figure_)


def result_to_dict(result: FitResult) -> dict[str, float | str]:
    """Convert a fit result to a tabular record."""
    return {
        "model": result.name,
        "cv_score": result.cv_score,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "roc_auc": result.roc_auc,
    }


def log_model_run(result: FitResult, x_test, y_test, cv: int, scoring: str) -> None:
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


def train_all(cv: int = 5, scoring: str = "f1", sample_size: int = 0) -> pd.DataFrame:
    """Optimize three model families, track them in MLflow and save the best one."""
    if scoring not in SUPPORTED_SCORING:
        raise ValueError(f"scoring doit etre dans {SUPPORTED_SCORING}")

    configure_mlflow()

    raw_df = load_data()
    df = clean_data(raw_df)
    validate_clean_data(df)

    if sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    x_train, x_test, y_train, y_test = split(df)

    results: list[FitResult] = []
    with mlflow.start_run(run_name="compare-optimized-models"):
        mlflow.log_params(
            {
                "cv": cv,
                "scoring": scoring,
                "sample_size": sample_size,
                "model_stage": MODEL_STAGE,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
            }
        )

        for spec in build_model_specs():
            print(f">> Optimisation: {spec.name}")
            result = optimize_model(spec, x_train, y_train, x_test, y_test, cv=cv, scoring=scoring)
            results.append(result)
            log_model_run(result, x_test, y_test, cv=cv, scoring=scoring)
            print(
                f"{result.name}: "
                f"cv_{scoring}={result.cv_score:.3f} "
                f"f1={result.f1:.3f} "
                f"roc_auc={result.roc_auc:.3f}"
            )

        results.sort(key=lambda item: getattr(item, scoring), reverse=True)
        best = results[0]
        results_df = pd.DataFrame(result_to_dict(result) for result in results)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = MODEL_DIR / "optimized_metrics.csv"
        confusion_matrix_path = MODEL_DIR / "optimized_confusion_matrix.png"
        results_df.to_csv(metrics_path, index=False)
        joblib.dump(best.best_estimator, MODEL_DIR / "model.joblib")
        save_confusion_matrix(best.best_estimator, x_test, y_test, confusion_matrix_path)

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

    print(f">> Meilleur modele optimise ({scoring}): {best.name}")
    print(results_df.to_string(index=False))
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--scoring", choices=SUPPORTED_SCORING, default="f1")
    parser.add_argument("--sample-size", type=int, default=0)
    args = parser.parse_args()
    train_all(cv=args.cv, scoring=args.scoring, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
