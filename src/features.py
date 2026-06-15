"""Pre-processing des variables numeriques et categorielles."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATEGORICAL_FEATURES, DATA_PATH, NUMERIC_FEATURES, TARGET


def expected_columns() -> list[str]:
    return [TARGET, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Kaggle dataset and keep the columns used by the model."""
    columns = expected_columns()
    missing_columns = sorted(set(columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans le dataset: {missing_columns}")

    clean_df = df.loc[:, columns].copy()

    for column in NUMERIC_FEATURES:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        clean_df[column] = clean_df[column].astype("string").str.strip()

    clean_df[TARGET] = pd.to_numeric(clean_df[TARGET], errors="coerce")
    clean_df = clean_df.dropna(subset=[TARGET])
    clean_df[TARGET] = clean_df[TARGET].astype(int)

    return clean_df


def validate_clean_data(df: pd.DataFrame) -> None:
    """Validate the cleaned dataset before training."""
    missing_columns = sorted(set(expected_columns()) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans le dataset nettoye: {missing_columns}")

    target_values = set(df[TARGET].dropna().unique())
    if not target_values <= {0, 1}:
        raise ValueError(f"La cible {TARGET!r} doit etre binaire avec les valeurs 0/1.")

    if df[TARGET].isna().any():
        raise ValueError(f"La cible {TARGET!r} contient des valeurs manquantes.")

    feature_columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    empty_feature_columns = [column for column in feature_columns if df[column].isna().all()]
    if empty_feature_columns:
        raise ValueError(f"Colonnes entierement vides: {empty_feature_columns}")


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing pipeline used before model training."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable: {DATA_PATH}. "
            "Telecharge le dataset Kaggle puis place train.csv dans data/."
        )

    raw_df = pd.read_csv(DATA_PATH)
    clean_df = clean_data(raw_df)
    validate_clean_data(clean_df)

    print(f"Dataset nettoye et valide: {DATA_PATH}")
    print(f"Lignes: {len(clean_df)}")
    print(f"Colonnes utilisees: {expected_columns()}")
    print("Repartition de la cible:")
    print(clean_df[TARGET].value_counts(normalize=True).rename("ratio").to_string())


if __name__ == "__main__":
    main()
