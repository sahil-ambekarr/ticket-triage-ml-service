import pytest
from fastapi.testclient import TestClient

from app.classifier import review_decision
from app.config import Settings
from app.main import create_app
from tests.conftest import make_settings


def test_both_above_threshold_is_auto():
    assert review_decision(0.9, 0.8, 0.6, 0.4) == (False, [])


@pytest.mark.parametrize(("cat", "pri", "n_reasons"), [(0.59, 0.9, 1), (0.9, 0.39, 1), (0.1, 0.1, 2)])
def test_low_confidence_flags_review(cat, pri, n_reasons):
    flagged, reasons = review_decision(cat, pri, 0.6, 0.4)
    assert flagged and len(reasons) == n_reasons


def test_threshold_is_inclusive():
    assert review_decision(0.6, 0.4, 0.6, 0.4)[0] is False


@pytest.mark.parametrize(("threshold", "expected"), [(0.0, False), (1.0, True)])
def test_api_respects_configured_threshold(small_model, threshold, expected):
    settings = make_settings(small_model, confidence_threshold=threshold, priority_confidence_threshold=0.0)
    with TestClient(create_app(settings)) as c:
        data = c.post("/triage", json={"body": "My order has not arrived yet"}).json()
    assert data["needs_human_review"] is expected
    assert (data["route_to_queue"] == "human_review") is expected


def test_invalid_threshold_env_rejected(monkeypatch):
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "1.5")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gpt")
    with pytest.raises(ValueError):
        Settings.from_env()
