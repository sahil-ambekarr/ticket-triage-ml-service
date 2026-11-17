from __future__ import annotations

import httpx

from app.providers.base import ProviderError


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float, transport: httpx.BaseTransport | None = None):
        self.model = model
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def generate(self, system: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            # Short and repeatable: this is a one-paragraph explanation, not a chat.
            "options": {"temperature": 0, "num_predict": 160},
        }
        try:
            response = self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            text = response.json().get("response", "")
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"ollama request failed: {exc.__class__.__name__}: {exc}") from exc
        if not text.strip():
            raise ProviderError("ollama returned an empty response")
        return text.strip()
