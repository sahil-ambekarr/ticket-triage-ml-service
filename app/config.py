"""Runtime settings, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

VALID_PROVIDERS = ("ollama", "claude", "mock")


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    anthropic_api_key: str = ""
    claude_model: str = "claude-haiku-5-5"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 1
    confidence_threshold: float = 0.6
    priority_confidence_threshold: float = 0.4
    model_path: Path = Path("models/triage_model.joblib")
    metrics_path: Path = Path("reports/metrics.json")
    top_k_terms: int = 5

    @classmethod
    def from_env(cls) -> Settings:
        provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {VALID_PROVIDERS}, got {provider!r}")
        settings = cls(
            llm_provider=provider,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url).rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            claude_model=os.getenv("CLAUDE_MODEL", cls.claude_model),
            llm_timeout_seconds=_float("LLM_TIMEOUT_SECONDS", cls.llm_timeout_seconds),
            llm_max_retries=_int("LLM_MAX_RETRIES", cls.llm_max_retries),
            confidence_threshold=_float("CONFIDENCE_THRESHOLD", cls.confidence_threshold),
            priority_confidence_threshold=_float("PRIORITY_CONFIDENCE_THRESHOLD", cls.priority_confidence_threshold),
            model_path=Path(os.getenv("MODEL_PATH", str(cls.model_path))),
            metrics_path=Path(os.getenv("METRICS_PATH", str(cls.metrics_path))),
        )
        for name in ("confidence_threshold", "priority_confidence_threshold"):
            value = getattr(settings, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        return settings
