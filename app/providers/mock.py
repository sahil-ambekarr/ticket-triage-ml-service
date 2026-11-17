from __future__ import annotations


class MockProvider:
    """Deterministic provider for tests and CI.

    The explainer recognises this provider and uses the template explanation
    built from the top terms, so mock-mode output is fully reproducible.
    """

    name = "mock"
    model = "template"

    def generate(self, system: str, prompt: str) -> str:
        return "__MOCK__"
