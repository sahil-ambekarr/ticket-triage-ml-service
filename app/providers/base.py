from __future__ import annotations

from typing import Protocol


class ProviderError(RuntimeError):
    """Raised when a provider cannot return usable text (network, timeout, bad payload)."""


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, system: str, prompt: str) -> str: ...
