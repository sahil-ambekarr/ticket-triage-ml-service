"""Measure the explanation layer on held-out test tickets.

For each ticket: classify, ask the LLM for an explanation, record latency and
two simple groundedness checks:
  * names_category  - the explanation names the predicted category
  * names_top_term  - it quotes at least one of the top contributing terms
Results are written after every ticket, so a run that is stopped early still
leaves usable partial numbers in the output file.

Usage:
    python -m scripts.eval_explain --n 12                       # ollama llama3.2
    python -m scripts.eval_explain --n 12 --provider mock       # template only, instant
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from app.classifier import TriageClassifier
from app.config import Settings
from app.explainer import Explainer, pretty
from app.pipeline import RANDOM_STATE
from app.providers import build_provider


def test_tickets(data_path: Path, n: int) -> list[dict[str, str]]:
    """Same split as scripts/train.py, then round-robin across categories for variety."""
    with data_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    _, idx_test = train_test_split(
        np.arange(len(rows)), test_size=0.2, random_state=RANDOM_STATE, stratify=[r["category"] for r in rows]
    )
    by_cat: dict[str, list[dict[str, str]]] = {}
    for i in sorted(idx_test):
        by_cat.setdefault(rows[i]["category"], []).append(rows[i])
    picked: list[dict[str, str]] = []
    depth = 0
    while len(picked) < n and any(depth < len(v) for v in by_cat.values()):
        for cat in sorted(by_cat):
            if depth < len(by_cat[cat]) and len(picked) < n:
                picked.append(by_cat[cat][depth])
        depth += 1
    return picked


def grounded(text: str, category: str, terms: list[str]) -> tuple[bool, bool, list[str]]:
    lower = text.lower()
    names_category = category.lower() in lower or pretty(category).lower() in lower
    hits = [t for t in terms if t.lower() in lower]
    return names_category, bool(hits), hits


def summarise(results: list[dict]) -> dict:
    llm = [r for r in results if r["source"] == "llm"]
    lat = np.array([r["latency_ms"] for r in llm]) if llm else np.array([])
    return {
        "tickets_attempted": len(results),
        "llm_answers": len(llm),
        "fallbacks": sum(r["source"] == "template_fallback" for r in results),
        "latency_ms_mean": round(float(lat.mean()), 1) if lat.size else None,
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 1) if lat.size else None,
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1) if lat.size else None,
        "latency_ms_max": round(float(lat.max()), 1) if lat.size else None,
        "names_category_rate": round(float(np.mean([r["names_category"] for r in llm])), 3) if llm else None,
        "names_top_term_rate": round(float(np.mean([r["names_top_term"] for r in llm])), 3) if llm else None,
        "fully_grounded_rate": round(float(np.mean([r["names_category"] and r["names_top_term"] for r in llm])), 3)
        if llm
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--provider", default="ollama", choices=["ollama", "claude", "mock"])
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--data", type=Path, default=Path("data/tickets.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/explain_eval.json"))
    args = parser.parse_args()

    settings = Settings.from_env()
    settings = Settings(
        **{**settings.__dict__, "llm_provider": args.provider, "llm_timeout_seconds": args.timeout}
    )
    clf = TriageClassifier.load(settings.model_path)
    # No retries during the eval: each latency is one LLM call, and a failure counts as a failure.
    explainer = Explainer(build_provider(settings), max_retries=0)
    tickets = test_tickets(args.data, args.n)

    report: dict = {
        "run_date": date.today().isoformat(),
        "provider": args.provider,
        "model": explainer.provider.model,
        "timeout_seconds": args.timeout,
        "hardware": "CPU-only laptop (AMD integrated graphics, 16 GB RAM), Windows 11; Ollama shared with other jobs",
        "python": platform.python_version(),
        "method": "sequential calls, temperature 0, one warm-up call excluded from the stats, no retries",
        "warmup_latency_ms": None,
        "summary": {},
        "results": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        report["summary"] = summarise(report["results"])
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.provider != "mock":
        warm = clf.predict("Hello", "Test ticket for warm-up.", settings.confidence_threshold, 0.4)
        report["warmup_latency_ms"] = explainer.explain(warm).latency_ms
        print(f"warm-up: {report['warmup_latency_ms']} ms", flush=True)
        flush()

    for i, row in enumerate(tickets, 1):
        pred = clf.predict(
            row["subject"], row["body"], settings.confidence_threshold, settings.priority_confidence_threshold
        )
        started = time.perf_counter()
        exp = explainer.explain(pred)
        wall_ms = round((time.perf_counter() - started) * 1000, 1)
        terms = [str(t["term"]) for t in pred.category.top_terms + pred.priority.top_terms]
        names_cat, names_term, hits = grounded(exp.text, pred.category.label, terms)
        report["results"].append(
            {
                "ticket_id": row["ticket_id"],
                "true_category": row["category"],
                "predicted_category": pred.category.label,
                "predicted_priority": pred.priority.label,
                "needs_human_review": pred.needs_human_review,
                "source": exp.source,
                "latency_ms": wall_ms,
                "names_category": names_cat,
                "names_top_term": names_term,
                "matched_terms": hits,
                "explanation": exp.text,
                "error": exp.error,
            }
        )
        flush()
        print(
            f"[{i}/{len(tickets)}] {row['ticket_id']} {exp.source} {wall_ms:.0f} ms "
            f"category={names_cat} term={names_term}",
            flush=True,
        )

    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
