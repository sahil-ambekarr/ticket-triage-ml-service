"""FastAPI service: classify a support ticket, explain the prediction, decide routing.

Run:  uvicorn app.main:app --port 4103
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from app.classifier import Prediction, TriageClassifier
from app.config import Settings
from app.explainer import Explainer
from app.logging_setup import configure_logging
from app.providers import build_provider

log = logging.getLogger("triage.api")
STATIC_DIR = Path(__file__).parent / "static"


class TicketIn(BaseModel):
    subject: str = Field("", max_length=300, description="Ticket subject line (optional)")
    body: str = Field(..., min_length=1, max_length=5000, description="Ticket body text")
    ticket_id: str | None = Field(None, max_length=64, description="Caller's own id, echoed back")

    @field_validator("body")
    @classmethod
    def body_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("body must contain non-whitespace text")
        return value


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        app.state.settings = settings
        app.state.classifier = None
        app.state.load_error = None
        try:
            app.state.classifier = TriageClassifier.load(settings.model_path)
            log.info("model loaded path=%s", settings.model_path)
        except Exception as exc:  # keep serving /health so the failure is visible
            app.state.load_error = str(exc)
            log.error("model failed to load: %s", exc)
        app.state.explainer = Explainer(build_provider(settings), max_retries=settings.llm_max_retries)
        yield

    app = FastAPI(
        title="Ticket Triage ML Service",
        version="0.1.0",
        description="scikit-learn ticket classifier with LLM explanations. Synthetic training data.",
        lifespan=lifespan,
    )

    def classifier(request: Request) -> TriageClassifier:
        clf = request.app.state.classifier
        if clf is None:
            raise HTTPException(status_code=503, detail=f"model not loaded: {request.app.state.load_error}")
        return clf

    def run_predict(request: Request, ticket: TicketIn) -> Prediction:
        return classifier(request).predict(
            ticket.subject,
            ticket.body,
            settings.confidence_threshold,
            settings.priority_confidence_threshold,
            settings.top_k_terms,
        )

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        log.info(
            "request method=%s path=%s status=%s ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        loaded = request.app.state.classifier is not None
        return {
            "status": "ok" if loaded else "degraded",
            "model_loaded": loaded,
            "llm_provider": settings.llm_provider,
        }

    @app.get("/model-info")
    def model_info(request: Request) -> dict[str, Any]:
        clf = classifier(request)
        info: dict[str, Any] = {
            "metadata": clf.metadata,
            "labels": {"category": clf.labels("category"), "priority": clf.labels("priority")},
            "thresholds": {
                "category": settings.confidence_threshold,
                "priority": settings.priority_confidence_threshold,
            },
            "llm": {
                "provider": settings.llm_provider,
                "model": request.app.state.explainer.provider.model,
            },
            "data": "synthetic, template-generated tickets; metrics overstate real-world performance",
        }
        if settings.metrics_path.exists():
            metrics = json.loads(settings.metrics_path.read_text(encoding="utf-8"))
            info["held_out_test"] = {
                target: metrics["targets"][target]["test"][metrics["served_model"]]
                for target in metrics.get("targets", {})
            }
        return info

    @app.post("/predict")
    def predict(request: Request, ticket: TicketIn) -> dict[str, Any]:
        return {"ticket_id": ticket.ticket_id, **run_predict(request, ticket).to_dict()}

    @app.post("/explain")
    def explain(request: Request, ticket: TicketIn) -> dict[str, Any]:
        prediction = run_predict(request, ticket)
        explanation = request.app.state.explainer.explain(prediction)
        return {"ticket_id": ticket.ticket_id, **prediction.to_dict(), "explanation": explanation.to_dict()}

    @app.post("/triage")
    def triage(request: Request, ticket: TicketIn) -> dict[str, Any]:
        """Single call for automations such as the n8n workflow: prediction + explanation + routing."""
        prediction = run_predict(request, ticket)
        explanation = request.app.state.explainer.explain(prediction)
        queue = "human_review" if prediction.needs_human_review else prediction.category.label
        return {
            "ticket_id": ticket.ticket_id,
            "category": prediction.category.label,
            "category_confidence": prediction.category.confidence,
            "priority": prediction.priority.label,
            "priority_confidence": prediction.priority.confidence,
            "needs_human_review": prediction.needs_human_review,
            "review_reasons": prediction.review_reasons,
            "route_to_queue": queue,
            "top_terms": {
                "category": prediction.category.top_terms,
                "priority": prediction.priority.top_terms,
            },
            "explanation": explanation.text,
            "explanation_source": explanation.source,
            "explanation_provider": explanation.provider,
        }

    return app


app = create_app()
