from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_paths():
    return {
        "model": Path("artifacts/model.joblib"),
        "metadata": Path("artifacts/metadata.json"),
        "train_csv": Path("data/processed/train.csv"),
    }