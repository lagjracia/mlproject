from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest


MODEL_PATH = Path("artifacts/model.joblib")
TRAIN_CSV = Path("data/processed/train.csv")
TARGET_COL = "Label"


def test_model_artifact_exists():
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not available.")
    assert MODEL_PATH.exists()


def test_model_artifact_loads_and_predicts():
    if not MODEL_PATH.exists() or not TRAIN_CSV.exists():
        pytest.skip("Local model artifact or train.csv not available.")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TRAIN_CSV).head(5)
    X = df.drop(columns=[TARGET_COL])

    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    assert len(probs) == 5
    assert len(preds) == 5