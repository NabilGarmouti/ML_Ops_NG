"""Optimisation de trois familles de modeles avec Optuna et MLflow."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import optuna
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
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import MODEL_DIR, RANDOM_STATE
from data import load_data, split
from features import build_preprocessor, clean_data, validate_clean_data
from tracking import (
    configure_mlflow,
    log_optuna_best_model,
    log_optuna_model_run,
    log_optuna_parent_run_params,
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


@dataclass
class FitResult:
    """Resultat d'optimisation et d'evaluation d'un modele."""

    name: str
    best_estimator: Pipeline
    best_params: dict[str, object]
    cv_score: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    study: optuna.Study


def build_model_specs() -> list[ModelSpec]:
    """Build the three optimized model families required for this step."""
    return [
        ModelSpec(name="random_forest"),
        ModelSpec(name="xgboost"),
        ModelSpec(name="lightgbm"),
    ]


def build_pipeline(estimator: object) -> Pipeline:
    """Build a preprocessing + classifier pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", estimator),
        ]
    )


def suggest_estimator(spec: ModelSpec, trial: optuna.Trial) -> object:
    """Build one estimator from the Optuna search space."""
    if spec.name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=100),
            max_depth=trial.suggest_int("max_depth", 6, 24, step=6),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 8),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    if spec.name == "xgboost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 7),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    if spec.name == "lightgbm":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300, step=100),
            num_leaves=trial.suggest_int("num_leaves", 31, 127, step=16),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 50, step=10),
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
        )

    raise ValueError(f"Famille de modele non supportee: {spec.name}")


def objective(
    spec: ModelSpec,
    trial: optuna.Trial,
    x_train,
    y_train,
    cv: int,
    scoring: str,
) -> float:
    """Evaluate one Optuna trial with cross-validation."""
    estimator = suggest_estimator(spec, trial)
    pipeline = build_pipeline(estimator)
    scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )
    return float(scores.mean())


def optimize_model(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    n_trials: int,
    cv: int,
    scoring: str,
) -> FitResult:
    """Optimize one model family with Optuna and evaluate it on the test set."""
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(spec, trial, x_train, y_train, cv=cv, scoring=scoring),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_estimator = suggest_estimator(spec, optuna.trial.FixedTrial(study.best_params))
    best_pipeline = build_pipeline(best_estimator)
    best_pipeline.fit(x_train, y_train)

    proba = best_pipeline.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return FitResult(
        name=spec.name,
        best_estimator=best_pipeline,
        best_params=study.best_params,
        cv_score=float(study.best_value),
        precision=float(precision_score(y_test, preds, zero_division=0)),
        recall=float(recall_score(y_test, preds, zero_division=0)),
        f1=float(f1_score(y_test, preds, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        study=study,
    )


def save_confusion_matrix(model: Pipeline, x_test, y_test, output_path: Path) -> None:
    """Save the confusion matrix for the selected model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds = model.predict(x_test)
    display = ConfusionMatrixDisplay.from_predictions(y_test, preds)
    display.figure_.tight_layout()
    display.figure_.savefig(output_path)
    plt.close(display.figure_)


def result_to_dict(result: FitResult, scoring: str) -> dict[str, float | str]:
    """Convert a fit result to a tabular record."""
    return {
        "model": result.name,
        f"cv_{scoring}": result.cv_score,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "roc_auc": result.roc_auc,
    }


def train_all(
    n_trials: int = 30,
    cv: int = 5,
    scoring: str = "f1",
    sample_size: int = 0,
) -> pd.DataFrame:
    """Optimize three model families with Optuna, track them in MLflow and save the best one."""
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
    with mlflow.start_run(run_name="optuna-compare-models"):
        log_optuna_parent_run_params(
            n_trials=n_trials,
            cv=cv,
            scoring=scoring,
            sample_size=sample_size,
            train_rows=len(x_train),
            test_rows=len(x_test),
        )

        for spec in build_model_specs():
            print(f">> Optuna: {spec.name}")
            result = optimize_model(
                spec,
                x_train,
                y_train,
                x_test,
                y_test,
                n_trials=n_trials,
                cv=cv,
                scoring=scoring,
            )
            results.append(result)
            log_optuna_model_run(result, x_test, y_test, n_trials=n_trials, cv=cv, scoring=scoring)
            print(
                f"{result.name}: "
                f"cv_{scoring}={result.cv_score:.3f} "
                f"f1={result.f1:.3f} "
                f"roc_auc={result.roc_auc:.3f}"
            )

        results.sort(key=lambda item: getattr(item, scoring), reverse=True)
        best = results[0]
        results_df = pd.DataFrame(result_to_dict(result, scoring) for result in results)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = MODEL_DIR / "optuna_metrics.csv"
        confusion_matrix_path = MODEL_DIR / "optuna_confusion_matrix.png"
        results_df.to_csv(metrics_path, index=False)
        joblib.dump(best.best_estimator, MODEL_DIR / "model.joblib")
        save_confusion_matrix(best.best_estimator, x_test, y_test, confusion_matrix_path)

        log_optuna_best_model(
            best=best,
            scoring=scoring,
            metrics_path=metrics_path,
            confusion_matrix_path=confusion_matrix_path,
        )

    print(f">> Meilleur modele Optuna ({scoring}): {best.name}")
    print(results_df.to_string(index=False))
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--scoring", choices=SUPPORTED_SCORING, default="f1")
    parser.add_argument("--sample-size", type=int, default=0)
    args = parser.parse_args()
    train_all(
        n_trials=args.n_trials,
        cv=args.cv,
        scoring=args.scoring,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
