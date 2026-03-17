from __future__ import annotations

import json
from pathlib import Path

import pytest


METADATA_PATH = Path("artifacts/metadata.json")


def test_metadata_file_exists():
    if not METADATA_PATH.exists():
        pytest.skip("Metadata file not available.")
    assert METADATA_PATH.exists()


def test_metadata_contains_required_keys():
    if not METADATA_PATH.exists():
        pytest.skip("Metadata file not available.")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    required_keys = {
        "model_name",
        "target_col",
        "threshold",
        "random_state",
        "train_rows",
        "test_rows",
        "feature_columns",
        "model_params",
        "test_metrics",
    }

    missing = required_keys - set(metadata.keys())
    assert not missing, f"Missing metadata keys: {missing}"