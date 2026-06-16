"""Client de test pour l'API FastAPI du modele."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import httpx

from config import API_URL, TARGET
from data import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

N_SAMPLES = 3


def build_payloads(n: int = N_SAMPLES) -> list[dict[str, object]]:
    """Build n valid payloads from the project dataset."""
    features = load_data().drop(columns=[TARGET])
    sample = features.sample(n=min(n, len(features)), random_state=42)
    return [json.loads(row.to_json()) for _, row in sample.iterrows()]


def load_payloads(payload_path: str | None) -> list[dict[str, object]]:
    """Load payloads from disk or build them from the dataset."""
    if payload_path is None:
        return build_payloads()

    path = Path(payload_path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data
    return [data]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=API_URL,
        help="URL de base de l'API (defaut: %(default)s).",
    )
    parser.add_argument(
        "--payload",
        help="Chemin vers un fichier JSON contenant un payload ou une liste de payloads.",
    )
    args = parser.parse_args()

    payloads = load_payloads(args.payload)

    try:
        with httpx.Client(base_url=args.url, timeout=10.0) as client:
            health = client.get("/health")
            logger.info("GET /health -> %s %s", health.status_code, health.json())

            for index, payload in enumerate(payloads, start=1):
                response = client.post("/predict", json=payload)
                logger.info(
                    "POST /predict (#%d) -> %s %s",
                    index,
                    response.status_code,
                    response.json(),
                )

            info = client.get("/model-info")
            logger.info("GET /model-info -> %s %s", info.status_code, info.json())
    except httpx.HTTPError as error:
        raise SystemExit(f"Impossible de joindre l'API: {error}") from error


if __name__ == "__main__":
    main()
