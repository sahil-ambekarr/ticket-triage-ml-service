# Interview guide: ticket-triage-ml-service

## 1. How it works, in simple words

- A seeded script writes 3,000 fake support tickets from templates, adds typos, mixed-topic tickets and some wrong labels.
- `scripts/train.py` turns text into TF-IDF features (words and character chunks), compares three linear models with 5-fold CV, and saves a Logistic Regression for category and one for priority.
- The API loads that bundle. For each ticket it returns probabilities, the words that pushed the model toward its answer, and `needs_human_review` if either confidence is below its threshold.
- `/explain` and `/triage` hand only the labels, probabilities and top words to an LLM (Ollama by default) and ask for two sentences. If the LLM fails or times out, a template built from the same evidence is used. The review sentence is always written by code.
- n8n calls `/triage` and branches on `needs_human_review`.

## 2. Files to read first, in order

1. `app/pipeline.py`: the features and models, shared by training and serving.
2. `scripts/train.py`: split, CV, held-out metrics, threshold table, routing summary.
3. `app/classifier.py`: probabilities, top-term extraction, `review_decision`.
4. `app/explainer.py`: prompt, retries, fallback, why review is not left to the LLM.
5. `app/main.py`: endpoints and validation.
6. `n8n/ticket-triage-workflow.json` with `tests/test_n8n_workflow.py`.
7. `reports/metrics.json` and `reports/explain_eval*.json`: the real numbers.

## 3. Likely questions, with honest answers

**Why classic ML instead of asking an LLM to classify?**
It runs in milliseconds on CPU, gives probabilities I can threshold, and is fully reproducible. The LLM takes about 10 s per call here, so it only phrases the explanation, which is optional.

**How good is the model really?**
Category 92.2% accuracy and priority 71.2% on a held-out split. But the data is template-generated, so test tickets look like training tickets. On real tickets it would be lower; I have not measured how much.

**Why serve Logistic Regression when LinearSVC scored higher?**
LinearSVC is about half a point better on category and two points on priority. Logistic Regression gives `predict_proba` for the threshold and per-class coefficients for the top terms. Calibrating LinearSVC is on the roadmap.

**How are the top terms computed?**
For the predicted class, each non-zero feature's tf-idf value times that class's coefficient. I keep the five largest positive word features; char n-grams are hidden because they are unreadable, so the list is a partial view.

**How do you stop the LLM from making things up?**
It never sees the ticket, only the model's evidence. The first eval still showed it contradicting the review flag in 4 of 12 cases, so the review sentence moved into code. It still adds some unsupported reasoning; the structured fields are the source of truth.

**How did you pick the thresholds?**
From the coverage table in `metrics.json`. Category below 0.6 or priority below 0.4 flags 11.5% of test tickets; accuracy is 94.5% on the rest and 73.9% on the flagged ones. A priority threshold of 0.6 would flag 25%.

**What happens when Ollama is down or slow?**
httpx timeout, one retry with backoff, then the template explanation with `source: template_fallback` and the error text. The prediction and routing are unaffected. Tests cover timeout, HTTP 500 and empty replies.

**How is it tested without an LLM or a committed model?**
`LLM_PROVIDER=mock`, and a session fixture trains a 600-row model in about 10 s. The Claude provider is tested through an `httpx2.MockTransport`. CI also checks that the generator reproduces the committed CSV byte-for-byte.

**Is the n8n workflow proven to work?**
No. Its structure is tested, but it was never imported into a running n8n. Docker and the live Claude API are also unverified.

**What would you change for real data?**
Retrain on the queue's own history, use a time-based split, re-pick thresholds against the cost of misroutes, and monitor drift in the share of tickets flagged for review.

## 4. Small improvements to make yourself

1. **Calibrated LinearSVC.** Add a `linear_svc_calibrated` option in `app/pipeline.py` using `CalibratedClassifierCV`, then compare its threshold table in `scripts/train.py` with Logistic Regression.
2. **Stricter groundedness check.** In `scripts/eval_explain.py`, extract every percentage from the explanation with a regex and check it matches a probability in the prediction (rounded).
3. **Show the review reasons in n8n.** In the workflow's "Queue For Human Review" Set node, add a field with `{{ $json.review_reasons.join('; ') }}`, and extend `tests/test_n8n_workflow.py` to assert it.
