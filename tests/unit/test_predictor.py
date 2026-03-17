from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from malware_detector.inference.predictor import load_predictor


MODEL_PATH = Path("artifacts/model.joblib")
TRAIN_CSV = Path("data/processed/train.csv")
TARGET_COL = "Label"


def test_predictor_loads_successfully():
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not available.")
    predictor = load_predictor(MODEL_PATH)
    assert predictor is not None
    assert predictor.pipeline is not None


def test_predictor_outputs_expected_columns():
    if not MODEL_PATH.exists() or not TRAIN_CSV.exists():
        pytest.skip("Local model artifact or train.csv not available.")

    predictor = load_predictor(MODEL_PATH)
    df = pd.read_csv(TRAIN_CSV).head(10)
    X = df.drop(columns=[TARGET_COL])

    result = predictor.predict_dataframe(X)

    assert "malware_probability" in result.columns
    assert "predicted_label" in result.columns
    assert len(result) == 10


def test_prediction_probabilities_are_valid():
    if not MODEL_PATH.exists() or not TRAIN_CSV.exists():
        pytest.skip("Local model artifact or train.csv not available.")

    predictor = load_predictor(MODEL_PATH)
    df = pd.read_csv(TRAIN_CSV).head(25)
    X = df.drop(columns=[TARGET_COL])

    probs = predictor.predict_proba(X)

    assert len(probs) == 25
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_predicted_labels_are_binary():
    if not MODEL_PATH.exists() or not TRAIN_CSV.exists():
        pytest.skip("Local model artifact or train.csv not available.")

    predictor = load_predictor(MODEL_PATH)
    df = pd.read_csv(TRAIN_CSV).head(25)
    X = df.drop(columns=[TARGET_COL])

    preds = predictor.predict(X)

    assert len(preds) == 25
    assert set(preds).issubset({0, 1})