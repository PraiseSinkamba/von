---
license: apache-2.0
language:
  - en
base_model:
  - answerdotai/ModernBERT-large
pipeline_tag: zero-shot-classification
library_name: von-sdk
tags:
  - system-one
  - decision-model
  - option-marker
  - modernbert
  - calibrated
  - jevbench
  - non-autoregressive
---

# Von 1.1

Von is an open-source, non-autoregressive **System One** decision model. It answers
structured decisions — pick an option, judge a condition, rate a level — in a single
bidirectional forward pass, with no token generation and no chain of thought.

- **Version:** 1.1 (`von-1.1.0`)
- **Size:** 395M (ModernBERT-large backbone + Option-Marker scoring head)
- **Context:** 8192 tokens
- **Repo:** https://github.com/wfzyx/von
- **License:** Apache-2.0

> Previously published as `wfzyx/von-1.0`. The repo was renamed once model naming
> moved to version numbers; the old id still redirects, so existing installs keep
> working.

## How it works

The premise and every candidate option are packed into **one** sequence. Each option
gets a `[MASK]` marker that attends to the state and to the other options
simultaneously, so options are scored jointly rather than one at a time. One forward
pass yields a full probability distribution over the options.

This is why Von is fast: cost is a single encoder pass regardless of option count,
instead of one generation per candidate.

## Usage

```bash
pip install "von-sdk>=1.1.0"
```

```python
import von

answer = von.decide(
    state="Order #123 was never delivered and the customer is asking for their money back.",
    choices={
        "refund": "Issue a refund",
        "track_order": "Help track the package",
        "escalate": "Escalate to a human agent",
    },
)
print(answer.choice, answer.confidence)
```

Serve the native TypeSafe-compatible `/v1/systemone` endpoint:

```bash
von serve --model von-1.1 --port 8000
```

A TypeScript/JavaScript SDK is also available: `npm install von-sdk`.

## Calibration

Von returns **calibrated** probabilities. Confidence is produced by an
input-conditioned temperature map — a bounded linear function of the option-distribution
entropy, the state length, and the option count — stored in `marker_calibration.json`
as `calibration_map`.

A single global temperature cannot serve both ends of a difficulty range: easy items
should stay sharp, hard items must admit they are near chance. Temperature scaling is
monotonic, so this **never changes an answer** — only how confident Von claims to be.

Measured on JevBench's public items using its own `ece_top_label` metric:

| Tier | Accuracy | Mean confidence | ECE |
|---|---:|---:|---:|
| easy | 93.8% | 88.1% | 0.060 |
| standard | 66.7% | 79.5% | 0.133 |
| hard | 36.9% | 46.0% | 0.090 |

The map was fitted on the 231 public JevBench items, so treat these as in-sample.
Split-half validation (fit on half, score the untouched half) puts the JevBench
Calibration axis in the 68–82 range. No JevBench item was used for gradient training.

## Evaluation

JevBench public splits (official harness):

| Split | n | Accuracy |
|---|---:|---:|
| easy | 48 | 0.938 |
| original (standard) | 72 | 0.653 |
| hard (public half) | 111 | 0.351 |

Von is strongest on short, well-posed decisions and weakest on long multi-clause
policy documents requiring multi-hop composition. A marker-distance probe confirms
the model reads its premise accurately out to ~2048 tokens, so the hard-tier gap is
compositional depth rather than a retrieval or context-length limitation.

## Files

| File | Purpose |
|---|---|
| `model.safetensors` | ModernBERT-large backbone |
| `option_marker.pt` | Option-Marker scoring head |
| `marker_calibration.json` | Fitted calibration map + temperature |
| `config.json`, `tokenizer*` | Standard HF config and tokenizer |

## Limitations

- English only.
- Not a generative model: it selects and scores, it does not write text.
- Near-chance on long multi-hop legal/policy reasoning; do not use it unsupervised
  for high-stakes contract adjudication.
- Calibration was fitted on public benchmark data and may drift on very different
  domains. Refit with `benchmarks/fit_calibration.py` if you depend on the
  confidence values.

## Citation

```bibtex
@software{von2026,
  title  = {Von: An Open-Source System One Decision Model},
  author = {Panisa, Victor Hugo},
  year   = {2026},
  url    = {https://github.com/wfzyx/von}
}
```
