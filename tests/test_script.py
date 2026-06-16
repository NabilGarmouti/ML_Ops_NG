from __future__ import annotations

import json

from script import DEFAULT_PAYLOAD, load_payload


def test_load_payload_uses_default_when_no_path():
    assert load_payload(None) == DEFAULT_PAYLOAD


def test_load_payload_reads_json_file(tmp_path):
    payload = {"Age": 30, "Gender": "Female"}
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_payload(str(payload_path)) == payload

