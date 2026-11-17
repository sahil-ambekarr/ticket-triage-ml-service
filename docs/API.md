# API reference

Base URL when running locally: `http://localhost:4103`. Interactive OpenAPI docs are served at `/docs`.

Every example below was captured with `curl` from the running service on 2026-11-17
(`LLM_PROVIDER=ollama`, `llama3.2` on a CPU-only laptop). Responses are pasted unedited, only re-indented.
LLM wording is not deterministic across machines or model versions; the classifier output is.

## Request body (all POST endpoints)

| field | type | rules |
|---|---|---|
| `body` | string | required, 1 to 5000 characters, not only whitespace |
| `subject` | string | optional, up to 300 characters |
| `ticket_id` | string | optional, up to 64 characters, echoed back |

Invalid input returns `422`. If the model file is missing, prediction endpoints return `503` and
`/health` reports `"status": "degraded"`.

## GET /health

```bash
curl -s http://localhost:4103/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "llm_provider": "ollama"
}
```

## GET /model-info

Model metadata, label sets, thresholds, explanation provider and held-out test metrics read from
`reports/metrics.json`.

```json
{
  "metadata": {
    "trained_on": "2026-11-17",
    "served_model": "logreg",
    "rows_used": 3000,
    "scikit_learn": "1.9.1",
    "test_accuracy": {
      "category": 0.9217,
      "priority": 0.7117
    },
    "test_macro_f1": {
      "category": 0.9165,
      "priority": 0.6927
    }
  },
  "labels": {
    "category": [
      "account",
      "billing",
      "bug",
      "feature_request",
      "login_access",
      "refund_return",
      "shipping"
    ],
    "priority": [
      "high",
      "low",
      "medium",
      "urgent"
    ]
  },
  "thresholds": {
    "category": 0.6,
    "priority": 0.4
  },
  "llm": {
    "provider": "ollama",
    "model": "llama3.2"
  },
  "data": "synthetic, template-generated tickets; metrics overstate real-world performance",
  "held_out_test": {
    "category": {
      "accuracy": 0.9217,
      "macro_f1": 0.9165,
      "fit_seconds": 2.88
    },
    "priority": {
      "accuracy": 0.7117,
      "macro_f1": 0.6927,
      "fit_seconds": 3.64
    }
  }
}
```

## POST /predict

Classifier only, no LLM call. `top_terms` are the word and word-bigram features with the largest positive
contribution (tf-idf value times class coefficient) to the predicted label.

```bash
curl -s -X POST http://localhost:4103/predict -H "Content-Type: application/json"   -d '{"ticket_id":"demo-1","subject":"Charged twice this month","body":"I was charged $49 twice on my card for the Pro plan. This is blocking my work today."}'
```

```json
{
  "ticket_id": "demo-1",
  "category": {
    "label": "billing",
    "confidence": 0.8685,
    "probabilities": {
      "billing": 0.8685,
      "feature_request": 0.0728,
      "shipping": 0.02,
      "account": 0.0121,
      "refund_return": 0.011,
      "login_access": 0.0097,
      "bug": 0.0058
    },
    "top_terms": [
      {
        "term": "charged",
        "weight": 0.156
      },
      {
        "term": "card",
        "weight": 0.151
      },
      {
        "term": "plan",
        "weight": 0.1377
      },
      {
        "term": "49",
        "weight": 0.1208
      },
      {
        "term": "my card",
        "weight": 0.0926
      }
    ]
  },
  "priority": {
    "label": "high",
    "confidence": 0.9232,
    "probabilities": {
      "high": 0.9232,
      "urgent": 0.0613,
      "medium": 0.0083,
      "low": 0.0071
    },
    "top_terms": [
      {
        "term": "today",
        "weight": 0.2486
      },
      {
        "term": "my work",
        "weight": 0.2244
      },
      {
        "term": "blocking my",
        "weight": 0.2222
      },
      {
        "term": "work today",
        "weight": 0.2117
      },
      {
        "term": "blocking",
        "weight": 0.1702
      }
    ]
  },
  "needs_human_review": false,
  "review_reasons": []
}
```

## POST /explain

Same as `/predict` plus an `explanation` object. `source` is `llm`, `template` (mock provider) or
`template_fallback` (the LLM failed or timed out; `error` then says why). The sentence about review is
appended by code, not written by the LLM. This call took 9.4 s on CPU.

```bash
curl -s -X POST http://localhost:4103/explain -H "Content-Type: application/json"   -d '{"ticket_id":"demo-1","subject":"Charged twice this month","body":"I was charged $49 twice on my card for the Pro plan. This is blocking my work today."}'
```

```json
{
  "ticket_id": "demo-1",
  "category": {
    "label": "billing",
    "confidence": 0.8685,
    "probabilities": {
      "billing": 0.8685,
      "feature_request": 0.0728,
      "shipping": 0.02,
      "account": 0.0121,
      "refund_return": 0.011,
      "login_access": 0.0097,
      "bug": 0.0058
    },
    "top_terms": [
      {
        "term": "charged",
        "weight": 0.156
      },
      {
        "term": "card",
        "weight": 0.151
      },
      {
        "term": "plan",
        "weight": 0.1377
      },
      {
        "term": "49",
        "weight": 0.1208
      },
      {
        "term": "my card",
        "weight": 0.0926
      }
    ]
  },
  "priority": {
    "label": "high",
    "confidence": 0.9232,
    "probabilities": {
      "high": 0.9232,
      "urgent": 0.0613,
      "medium": 0.0083,
      "low": 0.0071
    },
    "top_terms": [
      {
        "term": "today",
        "weight": 0.2486
      },
      {
        "term": "my work",
        "weight": 0.2244
      },
      {
        "term": "blocking my",
        "weight": 0.2222
      },
      {
        "term": "work today",
        "weight": 0.2117
      },
      {
        "term": "blocking",
        "weight": 0.1702
      }
    ]
  },
  "needs_human_review": false,
  "review_reasons": [],
  "explanation": {
    "text": "The support ticket has been classified as a billing issue with a predicted probability of 87% due to the presence of terms such as \"charged\" and \"card\". The high priority classification is also supported by the top term \"today\", which suggests that the issue is urgent and requires immediate attention, with a predicted probability of 92%. Both confidences are above the review threshold, so it can be routed automatically.",
    "source": "llm",
    "provider": "ollama",
    "model": "llama3.2",
    "latency_ms": 9353.4,
    "error": null
  }
}
```

## POST /triage

A flat response meant for automations (the n8n workflow calls this). `route_to_queue` is
`human_review` when `needs_human_review` is true, otherwise the predicted category.
Example with a deliberately ambiguous ticket:

```bash
curl -s -X POST http://localhost:4103/triage -H "Content-Type: application/json"   -d '{"ticket_id":"demo-2","subject":"Need help","body":"I returned the headphones but my card was billed again for the monthly plan. Can somebody contact me about this?"}'
```

```json
{
  "ticket_id": "demo-2",
  "category": "billing",
  "category_confidence": 0.3431,
  "priority": "medium",
  "priority_confidence": 0.3315,
  "needs_human_review": true,
  "review_reasons": [
    "category confidence 0.34 < threshold 0.60",
    "priority confidence 0.33 < threshold 0.40"
  ],
  "route_to_queue": "human_review",
  "top_terms": {
    "category": [
      {
        "term": "card",
        "weight": 0.1594
      },
      {
        "term": "plan",
        "weight": 0.1454
      },
      {
        "term": "card was",
        "weight": 0.1267
      },
      {
        "term": "billed",
        "weight": 0.1044
      },
      {
        "term": "was billed",
        "weight": 0.1011
      }
    ],
    "priority": [
      {
        "term": "can",
        "weight": 0.1193
      },
      {
        "term": "headphones",
        "weight": 0.0926
      },
      {
        "term": "but",
        "weight": 0.0846
      },
      {
        "term": "me",
        "weight": 0.0812
      },
      {
        "term": "returned",
        "weight": 0.0611
      }
    ]
  },
  "explanation": "The predicted category for the support ticket is billing, with a moderate level of confidence due to the presence of terms like \"card\" and \"plan\". The top priority assigned to the ticket is medium, reflecting its relative severity, despite being slightly less urgent than high-priority tickets. Confidence is below the review threshold, so a person should check it before it is routed.",
  "explanation_source": "llm",
  "explanation_provider": "ollama"
}
```

## Validation error example

```bash
curl -s -X POST http://localhost:4103/predict -H "Content-Type: application/json" -d '{"body":"   "}'
```

HTTP 422:

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": [
        "body",
        "body"
      ],
      "msg": "Value error, body must contain non-whitespace text",
      "input": "   ",
      "ctx": {
        "error": {}
      }
    }
  ]
}
```
