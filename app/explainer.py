"""Turns a prediction into a short plain-English explanation.

The prompt contains only what the classifier produced: labels, probabilities
and the top contributing terms. The ticket text itself is not sent, so the
LLM can only restate the model's evidence, not invent its own reading of the
ticket. If the LLM fails, times out or returns nothing, a template built from
the same evidence is used instead.

The review decision is never left to the LLM. A first eval run (see
reports/explain_eval_v1_prompt.json) showed llama3.2 contradicting the
needs_human_review flag in 4 of 12 explanations, so the LLM now explains only
the labels and a fixed sentence about review is appended by code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.classifier import Prediction, TargetPrediction
from app.providers.base import LLMProvider, ProviderError
from app.providers.mock import MockProvider

log = logging.getLogger("triage.explainer")

SYSTEM_PROMPT = (
    "You explain the output of a support-ticket classifier to a support agent. "
    "Use only the facts given: predicted labels, probabilities and the terms that pushed the model "
    "toward each label. Do not guess at details of the ticket that are not in the facts. "
    "Write 2 plain sentences, no lists, no markdown. Name the predicted category, quote at least "
    "one of the listed terms in double quotes, and mention the priority. "
    "Do not say whether the ticket needs human review; that is decided elsewhere."
)


@dataclass
class Explanation:
    text: str
    source: str  # "llm" | "template" | "template_fallback"
    provider: str
    model: str
    latency_ms: float
    error: str | None = None
    llm_text: str | None = None  # raw LLM output before the review sentence is appended

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


def pretty(label: str) -> str:
    return label.replace("_", " ")


def _terms(target: TargetPrediction, n: int = 3) -> list[str]:
    return [str(t["term"]) for t in target.top_terms[:n]]


def _quoted(terms: list[str]) -> str:
    return ", ".join(f'"{t}"' for t in terms)


def build_prompt(prediction: Prediction) -> str:
    def block(name: str, target: TargetPrediction) -> str:
        probs = ", ".join(f"{pretty(k)} {v:.0%}" for k, v in list(target.probabilities.items())[:3])
        terms = _quoted(_terms(target, 5)) or "(none)"
        return (
            f"{name}: {pretty(target.label)} ({target.confidence:.0%})\n"
            f"  top probabilities: {probs}\n"
            f"  terms: {terms}"
        )

    return (
        f"{block('Category', prediction.category)}\n"
        f"{block('Priority', prediction.priority)}\n\n"
        "Explain this prediction."
    )


def review_sentence(prediction: Prediction) -> str:
    if prediction.needs_human_review:
        return "Confidence is below the review threshold, so a person should check it before it is routed."
    return "Both confidences are above the review threshold, so it can be routed automatically."


def template_explanation(prediction: Prediction) -> str:
    cat, pri = prediction.category, prediction.priority
    cat_terms = _quoted(_terms(cat)) or "no single strong term"
    pri_terms = _quoted(_terms(pri)) or "no single strong term"
    text = (
        f"The model classified this ticket as {pretty(cat.label)} ({cat.confidence:.0%} confidence), "
        f"mainly because of {cat_terms}. "
        f"It set the priority to {pri.label} ({pri.confidence:.0%}), driven by {pri_terms}. "
    )
    return text + review_sentence(prediction)


class Explainer:
    def __init__(self, provider: LLMProvider, max_retries: int = 1, backoff_seconds: float = 0.5):
        self.provider = provider
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def explain(self, prediction: Prediction) -> Explanation:
        started = time.perf_counter()
        if isinstance(self.provider, MockProvider):
            return Explanation(template_explanation(prediction), "template", "mock", "template", 0.0)

        prompt = build_prompt(prediction)
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                text = self.provider.generate(SYSTEM_PROMPT, prompt)
                latency = round((time.perf_counter() - started) * 1000, 1)
                log.info("explanation generated provider=%s latency_ms=%s", self.provider.name, latency)
                full = f"{text} {review_sentence(prediction)}"
                return Explanation(full, "llm", self.provider.name, self.provider.model, latency, llm_text=text)
            except ProviderError as exc:
                last_error = str(exc)
                log.warning(
                    "explanation attempt failed provider=%s attempt=%d error=%s",
                    self.provider.name,
                    attempt + 1,
                    last_error,
                )
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))

        latency = round((time.perf_counter() - started) * 1000, 1)
        return Explanation(
            template_explanation(prediction),
            "template_fallback",
            self.provider.name,
            self.provider.model,
            latency,
            error=last_error,
        )
