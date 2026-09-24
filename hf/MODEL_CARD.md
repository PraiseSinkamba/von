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

# Von 1.2

Von is an open-source, non-autoregressive **System One** decision model. It answers
structured decisions — pick an option, judge a condition, rate a level — in a single
bidirectional forward pass, with no token generation and no chain of thought.

- **Version:** 1.2 (`von-1.2.0`)
- **Size:** 395M (ModernBERT-large backbone + Option-Marker scoring head)
- **Context:** 8192 tokens
- **Repo:** https://github.com/wfzyx/von
- **License:** Apache-2.0

> Previously published as `wfzyx/von-1.0`. The repo was renamed once model naming
> moved to version numbers; the old id still redirects, so existing installs keep
> working.

## How it works

The premise and every candidate option are packed into **one** sequence. Each option
gets a `[MASK]` marker whose final hidden state is scored by a small head. One forward
pass yields a full probability distribution over the options.

This is why Von is fast: cost is a single encoder pass regardless of option count,
instead of one generation per candidate.

### New in 1.2: order-invariant option scoring

In 1.1, every option marker also attended to the *other* options, and its rotary
position depended on how many options preceded it. That made the answer depend on the
order the options happened to be listed in: on JevBench's option-order diagnostic,
**49.5%** of Von 1.1's hard-tier answers changed when the options were reordered
(reference models: 0–5%).

1.2 removes that dependence at the architecture level. Inside the encoder:

- an option's tokens attend only to the shared premise and to that option's own tokens
  (never to another option),
- every option's position ids restart at the end of the premise, as if it were the
  only option present,
- ModernBERT's local sliding-window attention is computed from those position ids,
  not from raw sequence index.

Each option's logit is therefore a function of *(premise, that option)* alone. This
is a mathematical guarantee, not a training tendency: permuting the options permutes the
logits and changes nothing else. Measured on the trained weights, JevBench hard tier,
111 items × 4 orderings: **0 flips**.

The trade: options can no longer "look at each other" inside the encoder. Comparison
happens in the softmax over per-option logits instead. 1.2 was retrained from the 1.1
weights under the new mask and recovers 1.1's accuracy on every public tier.

## Usage

```bash
pip install "von-sdk>=1.2.0"
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
von serve --model von-1.2 --port 8000
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
| easy | 100.0% | 89.1% | 0.109 |
| standard | 63.9% | 65.9% | 0.045 |
| hard | 36.9% | 44.4% | 0.089 |

The map was fitted on the 231 public JevBench items, so treat these as in-sample.
In-sample JevBench Calibration axis 77.4; split-half validation (fit on half, score
the untouched half) gives 67.6. No JevBench item was used for gradient training.

## Evaluation

JevBench public splits (official harness):

| Split | n | Von 1.1 | **Von 1.2** | Order flips (4 orderings) |
|---|---:|---:|---:|---:|
| easy | 48 | 0.938 | **1.000** | — |
| original (standard) | 72 | 0.653 | 0.639 | — |
| hard (public half) | 111 | 0.387 | **0.387** | 1.1: 49.5% → **1.2: 0.0%** |

Von is strongest on short, well-posed decisions and weakest on long multi-clause
policy documents requiring multi-hop composition. A marker-distance probe confirms
the model reads its premise accurately out to ~2048 tokens, so the hard-tier gap is
compositional depth rather than a retrieval or context-length limitation.

## Files

| File | Purpose |
|---|---|
| `model.safetensors` | ModernBERT-large backbone |
| `option_marker.pt` | Option-Marker scoring head |
| `marker_calibration.json` | Fitted calibration map, zero-shot noul prior, and `independent_options: true` (tells the SDK to run the order-invariant attention mode — required for these weights) |
| `config.json`, `tokenizer*` | Standard HF config and tokenizer |

## Limitations

- English only.
- Not a generative model: it selects and scores, it does not write text.
- Near-chance on long multi-hop legal/policy reasoning; do not use it unsupervised
  for high-stakes contract adjudication.
- Zero-shot Noul (no criteria given) uses a fitted context-free prior correction;
  on a held-out 175-item dev set it scores 85.1% (1.1: 81.7%).
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
