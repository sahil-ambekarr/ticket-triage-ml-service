# Model card: ticket triage classifier

## Summary

Two scikit-learn text classifiers that read a support ticket (subject + body) and predict:

- **category**: `account`, `billing`, `bug`, `feature_request`, `login_access`, `refund_return`, `shipping`
- **priority**: `low`, `medium`, `high`, `urgent`

Both are `TfidfVectorizer` (word 1-2 grams, min_df 2) + `TfidfVectorizer` (char_wb 3-5 grams, min_df 3)
joined with a `FeatureUnion`, followed by `LogisticRegression(C=5, class_weight="balanced")`.
Definitions live in `app/pipeline.py`; training in `scripts/train.py`.

A separate explanation layer (`app/explainer.py`) turns the prediction into plain English. It is not part
of the model and cannot change a label or the review decision.

## Training data: synthetic, read this first

**All data is synthetic and template-generated** by `scripts/generate_dataset.py` (seed 42, 3,000 rows,
`data/tickets.csv`). No real customer tickets were used. Tickets are built from about 190 hand-written
sentence templates (74 subjects, 68 category sentences, 50 filler and urgency cues) and slot fillers, then roughened:

| noise | rate (as configured) |
|---|---|
| typos (swap / drop / double letters) | 35% of tickets |
| ambiguous: an extra sentence from another category | 12% (303 rows) |
| vague: almost no category signal | 4% (120 rows) |
| priority cue missing | 22% of tickets |
| priority cue from a neighbouring level | 12% of tickets that have a cue |
| category label flipped at random | 4% |
| priority label flipped to a neighbouring level | 4% |

213 of 3,000 rows (7.1%) carry at least one flipped label; the `label_noise` column marks them.

Because the test tickets come from the same templates as the training tickets, **the metrics below
overstate real-world performance**. Real tickets have far more vocabulary, longer threads, multiple
languages, signatures and quoted replies. Treat these numbers as a check that the pipeline works, not as
an estimate of production accuracy.

Class balance: bug 576, billing 549, login_access 522, shipping 401, refund_return 331, account 313,
feature_request 308. Priority: medium 1142, high 819, low 670, urgent 369.

## Evaluation

Run on 2026-11-17, Python 3.14.5, scikit-learn 1.9.1, Windows 11 laptop (CPU only).
Stratified (on category) 80/20 split, `random_state=42`: 2,400 train / 600 test.
Cross-validation is 5-fold stratified on the training split only. Full output: `reports/metrics.json`.

### Model comparison

| target | model | CV accuracy | CV macro-F1 | test accuracy | test macro-F1 |
|---|---|---|---|---|---|
| category | **Logistic Regression (served)** | 0.925 ± 0.012 | 0.921 | 0.922 | 0.917 |
| category | LinearSVC | 0.925 ± 0.013 | 0.922 | 0.927 | 0.922 |
| category | Complement NB | 0.928 ± 0.014 | 0.926 | 0.925 | 0.921 |
| priority | **Logistic Regression (served)** | 0.700 ± 0.007 | 0.693 | 0.712 | 0.693 |
| priority | LinearSVC | 0.716 ± 0.008 | 0.709 | 0.732 | 0.715 |
| priority | Complement NB | 0.696 ± 0.006 | 0.691 | 0.698 | 0.684 |

The three models are within about one CV standard deviation of each other on category. LinearSVC is
about 2 points better on priority. Logistic Regression is served anyway because the API needs calibrated-ish
probabilities for the review threshold and per-class coefficients for the top-terms evidence; LinearSVC
gives neither without extra calibration.

### Per-class (served model, held-out test)

| category | precision | recall | F1 | support |
|---|---|---|---|---|
| account | 0.915 | 0.857 | 0.885 | 63 |
| billing | 0.935 | 0.918 | 0.927 | 110 |
| bug | 0.949 | 0.974 | 0.961 | 115 |
| feature_request | 0.905 | 0.919 | 0.912 | 62 |
| login_access | 0.941 | 0.914 | 0.927 | 104 |
| refund_return | 0.934 | 0.864 | 0.898 | 66 |
| shipping | 0.856 | 0.963 | 0.906 | 80 |

| priority | precision | recall | F1 | support |
|---|---|---|---|---|
| low | 0.662 | 0.803 | 0.726 | 127 |
| medium | 0.810 | 0.703 | 0.753 | 236 |
| high | 0.707 | 0.682 | 0.695 | 170 |
| urgent | 0.558 | 0.642 | 0.597 | 67 |

Confusion matrices: `reports/confusion_category.png`, `reports/confusion_priority.png`.

On the 557 test rows whose labels were not flipped, category accuracy is 0.952 and priority accuracy 0.733,
so about 20 of the 47 category errors fall on the 43 test rows with a flipped label.

### Review threshold

The API flags `needs_human_review` when category confidence < 0.6 **or** priority confidence < 0.4
(both configurable). On the held-out split this flags 69 of 600 tickets (11.5%). Category accuracy is
0.945 on the auto-routed tickets and 0.739 on the flagged ones, so the threshold does separate easy
from hard tickets. `reports/metrics.json` has a coverage / accuracy table for thresholds 0.3 to 0.9.

## Intended use

A portfolio demonstration of a classic-ML triage service with an explanation layer and an automation hook
(n8n). It is not trained on real data and should not route real customer tickets without retraining on
the target queue's own history and re-measuring.

## Known limitations

- Synthetic, template-generated data (see above). Expect large drops on real tickets.
- Priority is hard by construction: 22% of tickets carry no urgency cue and some carry a misleading one.
  `urgent` is the weakest class (F1 0.60).
- Top terms only cover word features. Char n-grams also influence the prediction but are hidden because
  fragments like `ogi` are unreadable, so the evidence list is a partial view of the model.
- Some top terms are filler words (`was`, `in`, `but`). They are genuine model evidence on this data,
  which is itself a sign of template artefacts.
- English only. No handling of quoted email threads, HTML or attachments.

## Explanation layer

The LLM receives only labels, the top three probabilities per target and the top terms; never the ticket
text. The review sentence is appended by code. On 12 held-out tickets with llama3.2 (CPU), every
explanation named the predicted category and quoted a top term, but a manual read found several that add
unsupported reasoning (see the README eval section and `reports/explain_eval.json`). The explanation is
a reading aid, not an audit trail; the structured `top_terms` and `probabilities` are the source of truth.
