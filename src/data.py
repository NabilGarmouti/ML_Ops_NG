"""Chargement, validation et decoupage des donnees."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
)


def expected_columns() -> list[str]:
    return [TARGET, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the project dataset from CSV."""
    return pd.read_csv(path)


def validate_data(df: pd.DataFrame) -> None:
    """Validate the minimal contract expected by the training pipeline."""
    missing_columns = sorted(set(expected_columns()) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans le dataset: {missing_columns}")

    target_values = set(df[TARGET].dropna().unique())
    if not target_values <= {0, 1}:
        raise ValueError(f"La cible {TARGET!r} doit etre binaire avec les valeurs 0/1.")

    if df[TARGET].isna().any():
        raise ValueError(f"La cible {TARGET!r} contient des valeurs manquantes.")

    feature_columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    empty_feature_columns = [column for column in feature_columns if df[column].isna().all()]
    if empty_feature_columns:
        raise ValueError(f"Colonnes entierement vides: {empty_feature_columns}")


def split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[Any, Any, Any, Any]:
    """Split features and target with stratification on the binary target."""
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(x, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable: {DATA_PATH}. "
            "Telecharge le dataset Kaggle puis place le CSV dans data/dataset.csv."
        )

    df = load_data()
    validate_data(df)

    print(f"Dataset valide: {DATA_PATH}")
    print(f"Lignes: {len(df)}")
    print(f"Colonnes utilisees: {expected_columns()}")
    print("Repartition de la cible:")
    print(df[TARGET].value_counts(normalize=True).rename("ratio").to_string())


if __name__ == "__main__":
    main()
