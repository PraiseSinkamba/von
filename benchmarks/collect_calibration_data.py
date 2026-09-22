"""Dump Von's raw option logits on every public JevBench item.

Calibration fitting needs to be repeatable without re-running the model, so this
captures the *logits* (not just the final probabilities). Any temperature or
input-conditioned map can then be fitted and re-fitted offline in milliseconds.

JevBench scores Calibration as the mean of two halves, both of which punish
overconfidence:
  - ECE on the hard tier, top-label, 10 equal-width bins.
  - 1 - mean TVD against `provenance.gold_probs` on the probability items.

Both are fixed by the same thing: probability mass that reflects real
uncertainty. Neither is affected by argmax, so calibration never changes an
answer.

Usage:
    uv run python benchmarks/collect_calibration_data.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, "src")

TIER_FILES = {
    "easy": "/tmp/jevbench/datasets/public/easy.jsonl",
    "standard": "/tmp/jevbench/datasets/public/original.jsonl",
    "hard": "/tmp/jevbench/datasets/public/hard.jsonl",
}


def as_text(state) -> str:
    return json.dumps(state) if isinstance(state, (dict, list)) else str(state)


def raw_logits(backend, case: dict, tokenizer):
    """Run the packed forward pass once and return (labels, raw logits)."""
    import torch

    q = case["question"]
    qtype = q.get("type", "choice")
    state = as_text(case["state"])
    crit = q.get("criteria") or {}

    if qtype == "noul":
        # Von packs an explicit true/false pair for Noul.
        labels = ["no", "yes"]
        descs = [
            crit.get("false", "No, condition is false."),
            crit.get("true", "Yes, condition holds true."),
        ]
    else:
        # Score criteria arrive as an ordered LIST of level descriptions, while
        # Choice criteria arrive as a dict of label -> description. Both reach
        # here, so normalise instead of assuming a mapping.
        if isinstance(crit, dict):
            labels = list(crit.keys())
            descs = [str(crit[k]) for k in labels]
        else:
            labels = [str(i) for i in range(len(crit))]
            descs = [str(c) for c in crit]

    model = backend._get_model()
    packed = model.pack_sequence(state, q.get("instructions", ""), descs)
    enc = tokenizer(packed, return_tensors="pt", truncation=True, max_length=8192)
    mask_id = tokenizer.mask_token_id
    positions = (enc["input_ids"][0] == mask_id).nonzero(as_tuple=True)[0].tolist()
    if len(positions) != len(labels):
        raise ValueError(f"marker/option mismatch: {len(positions)} masks vs {len(labels)} options")

    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"].to(backend.device),
            attention_mask=enc["attention_mask"].to(backend.device),
            mask_positions=[positions],
        )
    return labels, [float(x) for x in out[0]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/von-option-marker-universal")
    parser.add_argument("--out", default="benchmarks/data/calibration_raw.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from von.backends.option_marker_backend import OptionMarkerBackend

    backend = OptionMarkerBackend(checkpoint_dir=args.checkpoint, device="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    records: List[dict] = []
    failures: List[dict] = []
    for tier, path in TIER_FILES.items():
        try:
            with open(path, "r", encoding="utf-8") as f:
                cases = [json.loads(l) for l in f if l.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read tier file {path}: {exc}") from exc
        print(f"\n=== {tier}: {len(cases)} cases", flush=True)
        t0 = time.time()
        for i, case in enumerate(cases, 1):
            try:
                labels, logits = raw_logits(backend, case, tokenizer)
            except Exception as exc:
                # Never silently drop cases: a biased subset produces a
                # calibration map fitted on the easy half of the distribution.
                failures.append({"id": case.get("id"), "error": f"{type(exc).__name__}: {exc}"})
                print(f"  case {i} ({case.get('id')}) FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            state_txt = as_text(case["state"])
            records.append({
                "tier": tier,
                "id": case.get("id", f"{tier}-{i}"),
                "family": case.get("family"),
                "qtype": case["question"].get("type", "choice"),
                "labels": labels,
                "logits": logits,
                "expected": str(case["expected"]),
                "gold_probs": (case.get("provenance") or {}).get("gold_probs"),
                "n_options": len(labels),
                "state_tokens": len(tokenizer.encode(state_txt, add_special_tokens=False)),
                "state_is_json": isinstance(case["state"], (dict, list)),
            })
            if i % 20 == 0:
                print(f"  {i}/{len(cases)} ({time.time()-t0:.0f}s)", flush=True)
        print(f"  {tier} done in {time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    except OSError as exc:
        raise RuntimeError(f"cannot write {args.out}: {exc}") from exc

    n_prob = sum(1 for r in records if r["gold_probs"])
    print(f"\nwrote {args.out}: {len(records)} records, {n_prob} with gold_probs")
    expected_total = 0
    for tier_path in TIER_FILES.values():
        with open(tier_path, "r", encoding="utf-8") as f:
            expected_total += sum(1 for line in f if line.strip())
    if failures or len(records) != expected_total:
        print(f"\n!! INCOMPLETE: {len(records)}/{expected_total} collected, {len(failures)} failures")
        for f_ in failures[:10]:
            print(f"   {f_['id']}: {f_['error']}")
        raise SystemExit(1)
    print(f"COMPLETE: {len(records)}/{expected_total} cases, zero failures")


if __name__ == "__main__":
    main()
