"""Optimisation de trois familles de modeles avec GridSearchCV et MLflow."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from math import prod
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import (
    MODEL_DIR,
    RANDOM_STATE,
)
from data import load_data, split
from features import build_preprocessor, clean_data, validate_clean_data
from tracking import (
    configure_mlflow,
    log_optimized_best_model,
    log_optimized_model_run,
    log_optimized_parent_run_params,
)

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


def count_param_combinations(param_grid: dict[str, list]) -> int:
    """Count how many hyperparameter combinations GridSearchCV will try."""
    return prod(len(values) for values in param_grid.values())


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
    combinations = count_param_combinations(spec.param_grid)
    total_fits = combinations * cv
    print(
        f">> {spec.name}: {combinations} combinaisons x {cv} folds = "
        f"{total_fits} entrainements",
        flush=True,
    )
    search = GridSearchCV(
        estimator=build_pipeline(spec.estimator),
        param_grid=spec.param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        verbose=2,
    )
    print(f">> {spec.name}: debut GridSearchCV", flush=True)
    search.fit(x_train, y_train)
    print(
        f">> {spec.name}: fin GridSearchCV, meilleur cv_{scoring}={search.best_score_:.3f}",
        flush=True,
    )

    best = search.best_estimator_
    print(f">> {spec.name}: evaluation sur le jeu de test", flush=True)
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
        log_optimized_parent_run_params(
            cv=cv,
            scoring=scoring,
            sample_size=sample_size,
            train_rows=len(x_train),
            test_rows=len(x_test),
        )

        for spec in build_model_specs():
            print(f">> Optimisation: {spec.name}")
            result = optimize_model(spec, x_train, y_train, x_test, y_test, cv=cv, scoring=scoring)
            results.append(result)
            log_optimized_model_run(result, x_test, y_test, cv=cv, scoring=scoring)
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

        log_optimized_best_model(
            best=best,
            scoring=scoring,
            metrics_path=metrics_path,
            confusion_matrix_path=confusion_matrix_path,
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
