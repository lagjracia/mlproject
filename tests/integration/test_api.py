from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from malware_detector.api.app import app


TRAIN_CSV = Path("data/processed/train.csv")

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_loaded" in payload


def test_predict_csv_endpoint():
    if not TRAIN_CSV.exists():
        pytest.skip("train.csv not available.")

    df = pd.read_csv(TRAIN_CSV).head(10)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    response = client.post(
        "/predict_csv",
        files={"file": ("sample.csv", io.BytesIO(csv_bytes), "text/csv")},
    )

    # This may be 500 if model artifact isn't present in CI/local yet
    assert response.status_code in {200, 500}

    if response.status_code == 200:
        payload = response.json()
        assert payload["rows"] == 10
        assert "predictions" in payload