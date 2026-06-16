"""API FastAPI pour servir le modele de classification."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the trained model once when the API starts."""
    model_path = MODEL_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modele introuvable: {model_path}. Lance d'abord `uv run python -m train`."
        )

    ml["model"] = joblib.load(model_path)
    logger.info("Modele charge depuis %s", model_path)
    yield
    ml.clear()


app = FastAPI(title="Cars Cross-Sell API", version="0.1.0", lifespan=lifespan)


class Features(BaseModel):
    """Input features expected by the trained pipeline."""

    Gender: str = Field(..., description="Genre du client")
    Age: int = Field(..., ge=18, le=100, description="Age du client")
    Driving_License: int = Field(..., ge=0, le=1, description="Possede un permis")
    Region_Code: float = Field(..., ge=0, description="Code region")
    Previously_Insured: int = Field(..., ge=0, le=1, description="Deja assure")
    Vehicle_Age: str = Field(..., description="Age du vehicule")
    Vehicle_Damage: str = Field(..., description="Vehicule deja endommage")
    Annual_Premium: float = Field(..., ge=0, description="Prime annuelle")
    Policy_Sales_Channel: float = Field(..., ge=0, description="Canal de vente")
    Vintage: int = Field(..., ge=0, description="Anciennete client")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Gender": "Male",
                    "Age": 44,
                    "Driving_License": 1,
                    "Region_Code": 28.0,
                    "Previously_Insured": 0,
                    "Vehicle_Age": "> 2 Years",
                    "Vehicle_Damage": "Yes",
                    "Annual_Premium": 40454.0,
                    "Policy_Sales_Channel": 26.0,
                    "Vintage": 217,
                }
            ]
        }
    }


class PredictionOut(BaseModel):
    """Prediction response returned by the API."""

    prediction: int
    probability: float


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    """Return served model metadata."""
    return {
        "version": os.environ.get("MODEL_VERSION", "local"),
        "model_path": str(MODEL_DIR / "model.joblib"),
    }


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    """Predict whether a customer is interested in vehicle insurance."""
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    row = pd.DataFrame([features.model_dump()])
    probability = float(model.predict_proba(row)[0, 1])
    return PredictionOut(prediction=int(probability >= 0.5), probability=round(probability, 4))
