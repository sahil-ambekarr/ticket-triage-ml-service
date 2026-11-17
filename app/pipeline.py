"""Model definitions shared by the training script and the API.

Keeping the feature extraction here means the service can never drift from
what the model was trained on: both sides call `combine_text` and the same
pipeline object is what gets pickled.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
MODEL_NAMES = ("logreg", "linear_svc", "complement_nb")


def combine_text(subject: str, body: str) -> str:
    """Single text field fed to the vectoriser. Subject first, body second."""
    return f"{subject.strip()} \n {body.strip()}".strip()


def build_features() -> FeatureUnion:
    """Word 1-2 grams catch phrases; char 3-5 grams survive typos ("pasword")."""
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, lowercase=True
                ),
            ),
        ]
    )


def build_pipeline(model_name: str = "logreg") -> Pipeline:
    if model_name == "logreg":
        clf = LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    elif model_name == "linear_svc":
        clf = LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_STATE)
    elif model_name == "complement_nb":
        clf = ComplementNB(alpha=0.3)
    else:
        raise ValueError(f"unknown model: {model_name!r} (expected one of {MODEL_NAMES})")
    return Pipeline([("features", build_features()), ("clf", clf)])
