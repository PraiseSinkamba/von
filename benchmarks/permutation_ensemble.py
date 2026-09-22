"""Does running Von N times with shuffled option order help?

Two questions, one experiment.

1. N passes. Von answers in ~18ms on GPU, so several passes still undercut the
   fastest competitor. But repeating a deterministic model on identical input
   returns an identical answer: extra passes buy nothing unless something
   varies. Option order is the natural thing to vary, since the Option-Marker
   packs all candidates into one sequence and each marker sits at a different
   absolute position.

2. Position bias. If Von is systematically drawn to a position rather than to
   the evidence, accuracy will depend on where the correct option sits, and
   averaging over permutations will recover real accuracy. That is the direct
   test of whether the below-chance families (multi_hop, long_policy, tradeoff)
   are losing to distractor attraction or to genuine reasoning failure.

Reports single-pass accuracy, order-averaged ensemble accuracy, and how often
the answer changes at all when only the option order moves -- an answer that
flips on a permutation was never grounded in the premise.

Usage:
    uv run python benchmarks/permutation_ensemble.py --tier hard --perms 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
from collections import Counter
from typing import Dict, List

sys.path.insert(0, "src")

TIER_FILES = {
    "easy": "/tmp/jevbench/datasets/public/easy.jsonl",
    "standard": "/tmp/jevbench/datasets/public/original.jsonl",
    "hard": "/tmp/jevbench/datasets/public/hard.jsonl",
}


def as_text(state) -> str:
    return json.dumps(state) if isinstance(state, (dict, list)) else str(state)


def labels_and_descs(case: dict):
    q = case["question"]
    crit = q.get("criteria") or {}
    qtype = q.get("type", "choice")
    if qtype == "noul":
        return ["no", "yes"], [
            crit.get("false", "No, condition is false."),
            crit.get("true", "Yes, condition holds true."),
        ]
    if isinstance(crit, dict):
        labels = list(crit.keys())
        return labels, [str(crit[k]) for k in labels]
    return [str(i) for i in range(len(crit))], [str(c) for c in crit]


def run_order(backend, tokenizer, case, labels, descs, order):
    """Score one permutation; returns probabilities keyed by original label."""
    import torch

    q = case["question"]
    state = as_text(case["state"])
    perm_descs = [descs[i] for i in order]

    model = backend._get_model()
    packed = model.pack_sequence(state, q.get("instructions", ""), perm_descs)
    enc = tokenizer(packed, return_tensors="pt", truncation=True, max_length=8192)
    positions = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].tolist()
    if len(positions) != len(order):
        raise ValueError(f"marker mismatch: {len(positions)} vs {len(order)}")

    with torch.no_grad():
        logits = backend._get_model()(
            input_ids=enc["input_ids"].to(backend.device),
            attention_mask=enc["attention_mask"].to(backend.device),
            mask_positions=[positions],
        )[0]
    temp = backend._effective_temperature(logits, state, len(order), tokenizer)
    probs = torch.softmax(logits / max(temp, 1e-4), dim=-1).tolist()
    # Undo the permutation so every pass is comparable in original label space.
    return {labels[order[i]]: probs[i] for i in range(len(order))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="hard", choices=list(TIER_FILES))
    parser.add_argument("--perms", type=int, default=4)
    parser.add_argument("--checkpoint", default="checkpoints/von-option-marker-universal")
    parser.add_argument("--out", default="benchmarks/data/permutation_hard.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from von.backends.option_marker_backend import OptionMarkerBackend

    backend = OptionMarkerBackend(checkpoint_dir=args.checkpoint, device="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    try:
        with open(TIER_FILES[args.tier], "r", encoding="utf-8") as f:
            cases = [json.loads(l) for l in f if l.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read tier file {TIER_FILES[args.tier]}: {exc}") from exc

    rng = random.Random(0)
    records: List[dict] = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        labels, descs = labels_and_descs(case)
        k = len(labels)
        identity = list(range(k))
        orders = [identity]
        # Reversal is the most informative second order; the rest are random.
        if k > 1:
            orders.append(list(reversed(identity)))
        while len(orders) < args.perms:
            cand = identity[:]
            rng.shuffle(cand)
            orders.append(cand)
        orders = orders[: args.perms]

        try:
            per_order = [run_order(backend, tokenizer, case, labels, descs, o) for o in orders]
        except Exception as exc:
            print(f"  case {case.get('id')} FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue

        records.append({
            "id": case.get("id"),
            "family": case.get("family"),
            "labels": labels,
            "expected": str(case["expected"]),
            "per_order": per_order,
            "orders": orders,
        })
        if i % 10 == 0:
            print(f"  {i}/{len(cases)} ({time.time()-t0:.0f}s)", flush=True)

    try:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(records, f)
    except OSError as exc:
        raise RuntimeError(f"cannot write {args.out}: {exc}") from exc
    print(f"wrote {args.out}: {len(records)} cases x {args.perms} orders in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
