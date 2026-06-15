"""Chargement et decoupage des donnees."""

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
    """Load the raw project dataset from CSV."""
    return pd.read_csv(path)


def split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[Any, Any, Any, Any]:
    """Split features and target with stratification on the binary target."""
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(x, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable: {DATA_PATH}. "
            "Telecharge le dataset Kaggle puis place train.csv dans data/."
        )

    df = load_data()

    print(f"Dataset charge: {DATA_PATH}")
    print(f"Lignes: {len(df)}")
    print(f"Colonnes disponibles: {list(df.columns)}")
    if TARGET in df.columns:
        print("Repartition de la cible:")
        print(df[TARGET].value_counts(normalize=True).rename("ratio").to_string())


if __name__ == "__main__":
    main()
