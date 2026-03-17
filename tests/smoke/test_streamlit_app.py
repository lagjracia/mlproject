from __future__ import annotations

from pathlib import Path


def test_streamlit_app_exists():
    app_path = Path("web/streamlit_app.py")
    assert app_path.exists(), f"Missing Streamlit app: {app_path}"