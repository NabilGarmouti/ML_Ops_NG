"""Artefacts d'evaluation et d'explicabilite."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from sklearn.pipeline import Pipeline

from config import RANDOM_STATE

warnings.filterwarnings(
    "ignore",
    message="LightGBM binary classifier with TreeExplainer shap values output has changed",
    category=UserWarning,
)


def _as_dense_array(values):
    if sparse.issparse(values):
        return values.toarray()
    return values


def _binary_shap_values(values):
    if isinstance(values, list):
        return values[1] if len(values) > 1 else values[0]
    values_array = np.asarray(values)
    if values_array.ndim == 3:
        return values_array[:, :, 1]
    return values_array


def log_shap_summary(
    model: Pipeline,
    x_test: pd.DataFrame,
    model_name: str,
    sample_size: int = 500,
) -> None:
    """Log a SHAP summary plot for a fitted tree-based pipeline."""
    sample = x_test.sample(
        n=min(sample_size, len(x_test)),
        random_state=RANDOM_STATE,
    )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["clf"]

    transformed_sample = _as_dense_array(preprocessor.transform(sample))
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    shap_values = _binary_shap_values(explainer.shap_values(transformed_sample))

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        transformed_sample,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.title(f"SHAP summary - {model_name}")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "shap_summary.png")
    plt.close()
