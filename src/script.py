"""Client simple pour appeler l'API de prediction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_PAYLOAD = {
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


def load_payload(payload_path: str | None) -> dict[str, object]:
    """Load a JSON payload from disk or fall back to the default example."""
    if payload_path is None:
        return DEFAULT_PAYLOAD

    path = Path(payload_path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def call_predict_api(api_url: str, payload: dict[str, object]) -> dict[str, object]:
    """Send one prediction request to the FastAPI service."""
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=30) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_PREDICT_URL", "http://127.0.0.1:8000/predict"),
        help="URL complete du endpoint /predict.",
    )
    parser.add_argument(
        "--payload",
        help="Chemin vers un fichier JSON contenant un payload de prediction.",
    )
    args = parser.parse_args()

    payload = load_payload(args.payload)
    print("Payload envoye :")
    print(json.dumps(payload, indent=2))

    try:
        prediction = call_predict_api(args.api_url, payload)
    except HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Erreur HTTP {error.code}: {message}") from error
    except URLError as error:
        raise SystemExit(f"Impossible de joindre l'API: {error.reason}") from error

    print("\nReponse API :")
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
