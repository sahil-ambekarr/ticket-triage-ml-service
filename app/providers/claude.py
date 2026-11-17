from __future__ import annotations

from typing import Any

import anthropic

from app.providers.base import ProviderError


class ClaudeProvider:
    """Anthropic Messages API via the official SDK. Not verified against the live API in this repo."""

    name = "claude"

    def __init__(self, api_key: str, model: str, timeout: float, http_client: Any = None):
        if not api_key:
            raise ProviderError("LLM_PROVIDER=claude needs ANTHROPIC_API_KEY")
        self.model = model
        # The SDK (2.x) brings its own HTTP stack (httpx2); tests pass an httpx2 client with a mock transport.
        # max_retries=0: retries are handled once, in the explainer, for every provider.
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=0, http_client=http_client)

    def generate(self, system: str, prompt: str) -> str:
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=300,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.AnthropicError as exc:
            raise ProviderError(f"claude request failed: {exc.__class__.__name__}: {exc}") from exc
        text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
        if not text.strip():
            raise ProviderError("claude returned no text")
        return text.strip()
