from __future__ import annotations

import joblib
from fastapi.testclient import TestClient
import numpy as np

import api


class DummyModel:
    def predict_proba(self, rows):
        return np.array([[0.3, 0.7] for _ in range(len(rows))])


def test_api_endpoints(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    joblib.dump(DummyModel(), model_dir / "model.joblib")
    monkeypatch.setattr(api, "MODEL_DIR", model_dir)

    with TestClient(api.app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json() == {"status": "ok"}

        model_info_response = client.get("/model-info")
        assert model_info_response.status_code == 200
        assert model_info_response.json()["model_path"].endswith("model.joblib")

        payload = {
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
        predict_response = client.post("/predict", json=payload)
        assert predict_response.status_code == 200
        assert predict_response.json() == {"prediction": 1, "probability": 0.7}
