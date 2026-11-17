from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import make_settings

TICKET = {
    "ticket_id": "demo-1",
    "subject": "Charged twice this month",
    "body": "I was charged $49 twice on my card for the Pro plan. This is blocking my work today.",
}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "llm_provider": "mock"}


def test_predict_shape(client):
    data = client.post("/predict", json=TICKET).json()
    assert data["ticket_id"] == "demo-1"
    assert data["category"]["label"] == "billing"
    probs = data["category"]["probabilities"]
    assert abs(sum(probs.values()) - 1) < 1e-3
    assert list(probs)[0] == data["category"]["label"]
    assert data["category"]["top_terms"] and "term" in data["category"]["top_terms"][0]
    assert isinstance(data["needs_human_review"], bool)


def test_explain_mock_is_template(client):
    exp = client.post("/explain", json=TICKET).json()["explanation"]
    assert exp["source"] == "template" and exp["provider"] == "mock"
    assert "billing" in exp["text"]


def test_triage_routing_fields(client):
    data = client.post("/triage", json=TICKET).json()
    expected = "human_review" if data["needs_human_review"] else data["category"]
    assert data["route_to_queue"] == expected
    assert data["explanation_source"] == "template"


def test_model_info(client):
    info = client.get("/model-info").json()
    assert "billing" in info["labels"]["category"]
    assert info["thresholds"] == {"category": 0.6, "priority": 0.4}
    assert "synthetic" in info["data"]
    assert "accuracy" in info["held_out_test"]["category"]


def test_demo_page(client):
    res = client.get("/")
    assert res.status_code == 200 and "Ticket Triage" in res.text


def test_rejects_blank_and_oversized_body(client):
    assert client.post("/predict", json={"body": "   "}).status_code == 422
    assert client.post("/predict", json={"body": "x" * 5001}).status_code == 422
    assert client.post("/predict", json={"subject": "no body"}).status_code == 422


def test_missing_model_returns_503(small_model, tmp_path):
    settings = make_settings(small_model, model_path=tmp_path / "nope.joblib")
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/predict", json=TICKET).status_code == 503
