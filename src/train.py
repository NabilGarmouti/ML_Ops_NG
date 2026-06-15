"""Entrainement et comparaison de plusieurs modeles de classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
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

from config import MODEL_DIR, RANDOM_STATE
from data import load_data, split
from features import build_preprocessor, clean_data, validate_clean_data


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
        "extra_trees": ExtraTreesClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            random_state=RANDOM_STATE,
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


def train(
    c: float = 1.0,
    max_iter: int = 1000,
    sample_size: int = 0,
    selection_metric: str = "f1",
) -> pd.DataFrame:
    """Train five models and save the best one according to the selected metric."""
    if selection_metric not in METRICS:
        raise ValueError(f"selection_metric doit etre dans {METRICS}")

    raw_df = load_data()
    df = clean_data(raw_df)
    validate_clean_data(df)

    if sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    x_train, x_test, y_train, y_test = split(df)
    models = build_models(c=c, max_iter=max_iter)

    results: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        print(f">> Entrainement: {name}")
        model.fit(x_train, y_train)
        metrics = evaluate_model(name, model, x_test, y_test)
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
    results_df.to_csv(MODEL_DIR / "metrics.csv", index=False)
    joblib.dump(best_model, MODEL_DIR / "model.joblib")
    save_confusion_matrix(best_model, x_test, y_test, MODEL_DIR / "confusion_matrix.png")

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
