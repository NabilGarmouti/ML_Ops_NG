from __future__ import annotations

import json

import pandas as pd

import script


def test_build_payloads(monkeypatch):
    dataset = pd.DataFrame(
        [
            {
                "Response": 1,
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
    )
    monkeypatch.setattr(script, "load_data", lambda: dataset)

    payloads = script.build_payloads(3)

    assert len(payloads) == 1
    assert "Response" not in payloads[0]


def test_load_payloads_reads_json_file(tmp_path):
    payload = {"Age": 30, "Gender": "Female"}
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    assert script.load_payloads(str(payload_path)) == [payload]
