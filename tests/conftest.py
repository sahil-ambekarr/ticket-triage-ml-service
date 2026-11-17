from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from scripts.train import train

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tickets.csv"


@pytest.fixture(scope="session")
def small_model(tmp_path_factory) -> dict:
    """Train once per test session on a 600-row sample; tests never depend on a committed model file."""
    out = tmp_path_factory.mktemp("model")
    metrics = train(DATA, out / "models", out / "reports", sample=600, cv_folds=3)
    return {
        "metrics": metrics,
        "model_path": out / "models" / "triage_model.joblib",
        "metrics_path": out / "reports" / "metrics.json",
        "reports_dir": out / "reports",
    }


def make_settings(small_model: dict, **overrides) -> Settings:
    base = {
        "llm_provider": "mock",
        "model_path": small_model["model_path"],
        "metrics_path": small_model["metrics_path"],
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def client(small_model):
    with TestClient(create_app(make_settings(small_model))) as c:
        yield c
