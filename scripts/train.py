"""Train and evaluate the category and priority classifiers.

Steps (identical for both targets):
  1. stratified 80/20 train/test split with a fixed seed
  2. 5-fold stratified cross-validation on the training split for three
     candidates: Logistic Regression, LinearSVC, Complement Naive Bayes
  3. refit Logistic Regression on the full training split (it is the served
     model because the API needs probabilities and per-class coefficients)
  4. held-out test metrics for all three, confusion matrix PNG for the served one

Usage:
    python -m scripts.train                       # full run, writes models/ and reports/
    python -m scripts.train --sample 600 --cv 3   # quick run (used by CI and tests)
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from datetime import date
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402

from app.pipeline import MODEL_NAMES, RANDOM_STATE, build_pipeline, combine_text  # noqa: E402

SERVED_MODEL = "logreg"
TARGETS = ("category", "priority")
THRESHOLD_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_THRESHOLDS = {"category": 0.6, "priority": 0.4}  # same defaults as app/config.py


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2.5, 1.0 * len(labels) + 2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    half = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > half else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def threshold_table(proba: np.ndarray, y_true: np.ndarray, classes: np.ndarray) -> list[dict[str, float]]:
    """For each confidence threshold: how many tickets stay automated, and how accurate those are."""
    confidence = proba.max(axis=1)
    correct = classes[proba.argmax(axis=1)] == y_true
    table = []
    for t in THRESHOLD_GRID:
        kept = confidence >= t
        table.append(
            {
                "threshold": t,
                "coverage": round(float(kept.mean()), 4),
                "accuracy_on_kept": round(float(correct[kept].mean()), 4) if kept.any() else None,
                "sent_to_review": int((~kept).sum()),
            }
        )
    return table


def routing_summary(
    pipelines: dict, x_test: list[str], rows: list[dict[str, str]], idx_test: np.ndarray
) -> dict[str, float | int]:
    """What the API's review rule (category < 0.6 or priority < 0.4) does on the held-out split."""
    cat_proba = pipelines["category"].predict_proba(x_test)
    pri_proba = pipelines["priority"].predict_proba(x_test)
    flagged = (cat_proba.max(axis=1) < DEFAULT_THRESHOLDS["category"]) | (
        pri_proba.max(axis=1) < DEFAULT_THRESHOLDS["priority"]
    )
    cat_pred = pipelines["category"].classes_[cat_proba.argmax(axis=1)]
    cat_true = np.array([rows[i]["category"] for i in idx_test])
    correct = cat_pred == cat_true
    return {
        "category_threshold": DEFAULT_THRESHOLDS["category"],
        "priority_threshold": DEFAULT_THRESHOLDS["priority"],
        "flagged_for_review": int(flagged.sum()),
        "flagged_share": round(float(flagged.mean()), 4),
        "category_accuracy_auto_routed": round(float(correct[~flagged].mean()), 4),
        "category_accuracy_flagged": round(float(correct[flagged].mean()), 4) if flagged.any() else None,
    }


def evaluate_target(
    target: str,
    x_train: list[str],
    x_test: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    clean_test_mask: np.ndarray,
    cv_folds: int,
    reports_dir: Path,
) -> tuple[dict, object]:
    result: dict = {"cv": {}, "test": {}}
    served_pipeline = None
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    for name in MODEL_NAMES:
        scores = cross_validate(build_pipeline(name), x_train, y_train, cv=cv, scoring=("accuracy", "f1_macro"))
        result["cv"][name] = {
            "accuracy_mean": round(float(scores["test_accuracy"].mean()), 4),
            "accuracy_std": round(float(scores["test_accuracy"].std()), 4),
            "macro_f1_mean": round(float(scores["test_f1_macro"].mean()), 4),
            "macro_f1_std": round(float(scores["test_f1_macro"].std()), 4),
        }

        pipeline = build_pipeline(name)
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        pred = pipeline.predict(x_test)
        result["test"][name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
            "fit_seconds": round(fit_seconds, 2),
        }
        print(
            f"[{target}] {name:14s} cv_acc={result['cv'][name]['accuracy_mean']:.4f} "
            f"cv_f1={result['cv'][name]['macro_f1_mean']:.4f} "
            f"test_acc={result['test'][name]['accuracy']:.4f} test_f1={result['test'][name]['macro_f1']:.4f}"
        )

        if name == SERVED_MODEL:
            served_pipeline = pipeline
            labels = list(pipeline.classes_)
            report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, pred, labels=labels)
            plot_confusion(cm, labels, f"{target} - {name} (held-out test)", reports_dir / f"confusion_{target}.png")
            result["served"] = {
                "model": name,
                "labels": labels,
                "per_class": {
                    label: {
                        "precision": round(report[label]["precision"], 4),
                        "recall": round(report[label]["recall"], 4),
                        "f1": round(report[label]["f1-score"], 4),
                        "support": int(report[label]["support"]),
                    }
                    for label in labels
                },
                "confusion_matrix": cm.tolist(),
                # Accuracy against the labels as generated, on the test rows whose label was NOT flipped.
                "accuracy_excluding_label_noise_rows": round(
                    float(accuracy_score(y_test[clean_test_mask], pred[clean_test_mask])), 4
                ),
                "threshold_table": threshold_table(pipeline.predict_proba(x_test), y_test, pipeline.classes_),
            }
    return result, served_pipeline


def train(
    data_path: Path,
    models_dir: Path,
    reports_dir: Path,
    sample: int | None = None,
    cv_folds: int = 5,
    test_size: float = 0.2,
) -> dict:
    rows = load_rows(data_path)
    if sample is not None and sample < len(rows):
        rng = np.random.default_rng(RANDOM_STATE)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), size=sample, replace=False))]

    texts = [combine_text(r["subject"], r["body"]) for r in rows]
    noise = np.array([r["label_noise"] == "1" for r in rows])
    # Stratify on category (the primary target); one split is shared by both targets.
    idx_train, idx_test = train_test_split(
        np.arange(len(rows)),
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=[r["category"] for r in rows],
    )
    x_train, x_test = [texts[i] for i in idx_train], [texts[i] for i in idx_test]

    metrics: dict = {
        "generated_on": date.today().isoformat(),
        "dataset": {
            "path": data_path.as_posix(),
            "rows_used": len(rows),
            "train_rows": len(idx_train),
            "test_rows": len(idx_test),
            "synthetic": True,
            "label_noise_rows_in_test": int(noise[idx_test].sum()),
        },
        "split": {"test_size": test_size, "random_state": RANDOM_STATE, "stratified_on": "category"},
        "cv_folds": cv_folds,
        "served_model": SERVED_MODEL,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "targets": {},
    }

    bundle = {"metadata": {}, "pipelines": {}}
    for target in TARGETS:
        y = np.array([r[target] for r in rows])
        result, pipeline = evaluate_target(
            target, x_train, x_test, y[idx_train], y[idx_test], ~noise[idx_test], cv_folds, reports_dir
        )
        metrics["targets"][target] = result
        bundle["pipelines"][target] = pipeline

    metrics["routing_at_default_thresholds"] = routing_summary(bundle["pipelines"], x_test, rows, idx_test)

    bundle["metadata"] = {
        "trained_on": metrics["generated_on"],
        "served_model": SERVED_MODEL,
        "rows_used": len(rows),
        "scikit_learn": sklearn.__version__,
        "test_accuracy": {t: metrics["targets"][t]["test"][SERVED_MODEL]["accuracy"] for t in TARGETS},
        "test_macro_f1": {t: metrics["targets"][t]["test"][SERVED_MODEL]["macro_f1"] for t in TARGETS},
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, models_dir / "triage_model.joblib", compress=3)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/tickets.csv"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--sample", type=int, default=None, help="train on a random subset of N rows")
    parser.add_argument("--cv", type=int, default=5, help="number of cross-validation folds")
    args = parser.parse_args()

    started = time.perf_counter()
    train(args.data, args.models_dir, args.reports_dir, sample=args.sample, cv_folds=args.cv)
    print(f"done in {time.perf_counter() - started:.1f}s -> {args.models_dir / 'triage_model.joblib'}")


if __name__ == "__main__":
    main()
