from __future__ import annotations

from app.config import Settings
from app.providers.base import LLMProvider
from app.providers.claude import ClaudeProvider
from app.providers.mock import MockProvider
from app.providers.ollama import OllamaProvider


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings.ollama_base_url, settings.ollama_model, settings.llm_timeout_seconds)
    if settings.llm_provider == "claude":
        return ClaudeProvider(settings.anthropic_api_key, settings.claude_model, settings.llm_timeout_seconds)
    return MockProvider()
