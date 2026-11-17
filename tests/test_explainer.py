import json

import httpx
import httpx2
import pytest

from app.classifier import Prediction, TargetPrediction
from app.explainer import Explainer, build_prompt, template_explanation
from app.providers.base import ProviderError
from app.providers.claude import ClaudeProvider
from app.providers.ollama import OllamaProvider


def _prediction(review: bool = False) -> Prediction:
    cat = TargetPrediction(
        "login_access",
        0.82,
        {"login_access": 0.82, "account": 0.1, "bug": 0.08},
        [{"term": "password", "weight": 0.3}, {"term": "reset", "weight": 0.2}],
    )
    pri = TargetPrediction("urgent", 0.7, {"urgent": 0.7, "high": 0.3}, [{"term": "locked out", "weight": 0.4}])
    return Prediction(cat, pri, review, ["category confidence 0.50 < threshold 0.60"] if review else [])


class FlakyProvider:
    name, model = "flaky", "test"

    def __init__(self, failures: int):
        self.failures, self.calls = failures, 0

    def generate(self, system: str, prompt: str) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderError("boom")
        return 'It is login access because of "password".'


def test_prompt_contains_only_model_evidence():
    prompt = build_prompt(_prediction())
    assert "login access (82%)" in prompt
    assert '"password"' in prompt and '"locked out"' in prompt
    # The review decision is appended by code, never asked of the LLM.
    assert "review" not in prompt.lower()


def test_llm_text_gets_deterministic_review_sentence():
    exp = Explainer(FlakyProvider(failures=0), max_retries=0).explain(_prediction(review=True))
    assert exp.llm_text == 'It is login access because of "password".'
    assert exp.text.endswith("a person should check it before it is routed.")


def test_template_mentions_category_terms_and_review():
    text = template_explanation(_prediction(review=True))
    assert "login access" in text and '"password"' in text and "a person should check" in text


def test_llm_success_after_retry():
    provider = FlakyProvider(failures=1)
    exp = Explainer(provider, max_retries=1, backoff_seconds=0).explain(_prediction())
    assert exp.source == "llm" and provider.calls == 2


def test_falls_back_to_template_after_retries_exhausted():
    provider = FlakyProvider(failures=5)
    exp = Explainer(provider, max_retries=1, backoff_seconds=0).explain(_prediction())
    assert exp.source == "template_fallback" and provider.calls == 2
    assert exp.text == template_explanation(_prediction()) and exp.error == "boom"


def _ollama(handler) -> OllamaProvider:
    return OllamaProvider("http://ollama.test", "llama3.2", timeout=1, transport=httpx.MockTransport(handler))


def test_ollama_timeout_falls_back():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    exp = Explainer(_ollama(handler), max_retries=0).explain(_prediction())
    assert exp.source == "template_fallback" and "ReadTimeout" in exp.error


@pytest.mark.parametrize("response", [httpx.Response(500, text="err"), httpx.Response(200, json={"response": " "})])
def test_ollama_error_or_empty_falls_back(response):
    exp = Explainer(_ollama(lambda request: response), max_retries=0).explain(_prediction())
    assert exp.source == "template_fallback"


def test_ollama_request_payload():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"response": 'Login access, driven by "password".'})

    exp = Explainer(_ollama(handler), max_retries=0).explain(_prediction())
    assert exp.source == "llm"
    assert seen["model"] == "llama3.2" and seen["stream"] is False and seen["options"]["temperature"] == 0


def _claude(handler) -> ClaudeProvider:
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    return ClaudeProvider("test-key", "claude-haiku-5-5", timeout=1, http_client=client)


def test_claude_provider_with_stubbed_http():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": 'Login access, mainly "password".'}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 8},
            },
        )

    exp = Explainer(_claude(handler), max_retries=0).explain(_prediction())
    assert exp.source == "llm" and exp.text.startswith("Login access")
    assert seen["url"].endswith("/v1/messages") and seen["body"]["model"] == "claude-haiku-5-5"


def test_claude_api_error_falls_back():
    error = {"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}}
    exp = Explainer(_claude(lambda request: httpx2.Response(529, json=error)), max_retries=0).explain(_prediction())
    assert exp.source == "template_fallback"


def test_claude_requires_key():
    with pytest.raises(ProviderError):
        ClaudeProvider("", "claude-haiku-5-5", timeout=1)
