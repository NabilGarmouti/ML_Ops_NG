"""Configuration centrale du projet de classification.

Le pipeline lit les chemins, la cible et les colonnes depuis ce fichier. Pour
changer de dataset plus tard, l'objectif est de modifier cette configuration
plutot que les scripts d'entrainement.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_PATH = ROOT / "data" / "train.csv"
MODEL_DIR = ROOT / "models"

TARGET = "Response"

NUMERIC_FEATURES: list[str] = [
    "Age",
    "Region_Code",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage",
]

CATEGORICAL_FEATURES: list[str] = [
    "Gender",
    "Driving_License",
    "Previously_Insured",
    "Vehicle_Age",
    "Vehicle_Damage",
]

RANDOM_STATE = 42

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "cars-cross-sell-baseline")
MODEL_NAME = os.getenv("MODEL_NAME", "cars-cross-sell-classifier")
MODEL_STAGES = ["integ", "prepod", "prod"]
MODEL_STAGE = os.getenv("MODEL_STAGE", "integ")

if MODEL_STAGE not in MODEL_STAGES:
    raise ValueError(f"MODEL_STAGE doit etre dans {MODEL_STAGES}, valeur recue: {MODEL_STAGE!r}")
