"""Loads the trained bundle and turns a ticket into a prediction with evidence.

"Top terms" are the word / word-bigram features with the largest positive
contribution (tf-idf value x class coefficient) to the predicted class. Char
n-gram features also feed the model but are left out of the evidence list
because fragments like "ogi" mean nothing to a reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.pipeline import combine_text

TARGETS = ("category", "priority")


@dataclass
class TargetPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    top_terms: list[dict[str, float | str]] = field(default_factory=list)


@dataclass
class Prediction:
    category: TargetPrediction
    priority: TargetPrediction
    needs_human_review: bool
    review_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        def target(t: TargetPrediction) -> dict[str, Any]:
            return {
                "label": t.label,
                "confidence": t.confidence,
                "probabilities": t.probabilities,
                "top_terms": t.top_terms,
            }

        return {
            "category": target(self.category),
            "priority": target(self.priority),
            "needs_human_review": self.needs_human_review,
            "review_reasons": self.review_reasons,
        }


def review_decision(
    category_confidence: float,
    priority_confidence: float,
    category_threshold: float,
    priority_threshold: float,
) -> tuple[bool, list[str]]:
    """A ticket goes to a person when either prediction is below its threshold."""
    reasons: list[str] = []
    if category_confidence < category_threshold:
        reasons.append(f"category confidence {category_confidence:.2f} < threshold {category_threshold:.2f}")
    if priority_confidence < priority_threshold:
        reasons.append(f"priority confidence {priority_confidence:.2f} < threshold {priority_threshold:.2f}")
    return bool(reasons), reasons


class TriageClassifier:
    def __init__(self, bundle: dict[str, Any]):
        self.metadata: dict[str, Any] = bundle.get("metadata", {})
        self.pipelines = bundle["pipelines"]
        self._feature_names: dict[str, np.ndarray] = {}
        for target in TARGETS:
            names = self.pipelines[target].named_steps["features"].get_feature_names_out()
            self._feature_names[target] = names

    @classmethod
    def load(cls, path: Path) -> TriageClassifier:
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path} (run: python -m scripts.train)")
        return cls(joblib.load(path))

    def labels(self, target: str) -> list[str]:
        return [str(c) for c in self.pipelines[target].classes_]

    def _predict_target(self, target: str, text: str, top_k: int) -> TargetPrediction:
        pipeline = self.pipelines[target]
        features = pipeline.named_steps["features"].transform([text])
        clf = pipeline.named_steps["clf"]
        proba = clf.predict_proba(features)[0]
        classes = [str(c) for c in clf.classes_]
        best = int(np.argmax(proba))

        row = features.tocsr()
        contributions = row.data * clf.coef_[best, row.indices]
        names = self._feature_names[target][row.indices]
        terms: list[dict[str, float | str]] = []
        for i in np.argsort(-contributions):
            if contributions[i] <= 0 or len(terms) >= top_k:
                break
            name = str(names[i])
            if name.startswith("word__"):
                terms.append({"term": name.removeprefix("word__"), "weight": round(float(contributions[i]), 4)})

        return TargetPrediction(
            label=classes[best],
            confidence=round(float(proba[best]), 4),
            probabilities={
                c: round(float(p), 4) for c, p in sorted(zip(classes, proba, strict=True), key=lambda x: -x[1])
            },
            top_terms=terms,
        )

    def predict(
        self,
        subject: str,
        body: str,
        category_threshold: float,
        priority_threshold: float,
        top_k: int = 5,
    ) -> Prediction:
        text = combine_text(subject, body)
        category = self._predict_target("category", text, top_k)
        priority = self._predict_target("priority", text, top_k)
        needs_review, reasons = review_decision(
            category.confidence, priority.confidence, category_threshold, priority_threshold
        )
        return Prediction(category, priority, needs_review, reasons)
