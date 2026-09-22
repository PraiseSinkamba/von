"""Measure whether Von's confidence can route cases to a second-stage model.

A confidence-gated cascade only works if confidence predicts correctness. If
Von is equally confident when right and when wrong, no threshold can select the
cases worth escalating, and the cascade degenerates into "escalate everything"
-- which forfeits the Speed and Cost axes that make Von competitive.

This dumps per-case confidence and correctness for every public JevBench tier,
then sweeps the escalation threshold to produce the fire-rate / blended-accuracy
curve a cascade design actually needs.

Usage:
    uv run python benchmarks/routing_gate.py [--checkpoint DIR] [--tiers easy,standard,hard]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List

sys.path.insert(0, "src")

TIER_FILES = {
    "easy": "/tmp/jevbench/datasets/public/easy.jsonl",
    "standard": "/tmp/jevbench/datasets/public/original.jsonl",
    "hard": "/tmp/jevbench/datasets/public/hard.jsonl",
}


def load(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_text(state) -> str:
    return json.dumps(state) if isinstance(state, (dict, list)) else str(state)


def predict(backend, case: dict):
    """Return (prediction, confidence, top1_minus_top2 margin)."""
    from von.types import Choice, Noul, Score

    q = case["question"]
    qtype = q.get("type", "choice")
    state = as_text(case["state"])

    if qtype == "choice":
        ans = backend.evaluate_choice("d", state, Choice(instructions=q["instructions"], criteria=q["criteria"]))
        probs = sorted(ans.probabilities.values(), reverse=True)
        margin = probs[0] - probs[1] if len(probs) > 1 else probs[0]
        return ans.choice, ans.confidence, margin
    if qtype == "noul":
        ans = backend.evaluate_noul("d", state, Noul(instructions=q["instructions"], criteria=q.get("criteria") or {}))
        p = ans.probability if hasattr(ans, "probability") else (1.0 if ans.noul else 0.0)
        return ("yes" if ans.noul else "no"), ans.confidence, abs(2 * p - 1)
    ans = backend.evaluate_score("d", state, Score(instructions=q["instructions"], criteria=q["criteria"]))
    probs_map = ans.probabilities or {}
    pred = max(probs_map, key=probs_map.get) if probs_map else str(int(round(ans.score)))
    probs = sorted(probs_map.values(), reverse=True) or [1.0]
    margin = probs[0] - probs[1] if len(probs) > 1 else probs[0]
    return pred, ans.confidence, margin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/von-option-marker-universal")
    parser.add_argument("--tiers", default="easy,standard,hard")
    parser.add_argument("--out", default="benchmarks/data/routing_gate.json")
    args = parser.parse_args()

    from von.backends.option_marker_backend import OptionMarkerBackend

    backend = OptionMarkerBackend(checkpoint_dir=args.checkpoint, device="cpu")
    records: List[dict] = []

    for tier in args.tiers.split(","):
        cases = load(TIER_FILES[tier])
        print(f"\n=== {tier}: {len(cases)} cases", flush=True)
        t0 = time.time()
        for i, case in enumerate(cases, 1):
            try:
                pred, conf, margin = predict(backend, case)
            except Exception as exc:
                print(f"  case {i} failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            ok = str(pred).strip().lower() == str(case["expected"]).strip().lower()
            records.append({
                "tier": tier,
                "id": case.get("id", f"{tier}-{i}"),
                "correct": bool(ok),
                "confidence": float(conf),
                "margin": float(margin),
                "n_tokens_state": len(as_text(case["state"])) // 4,
            })
            if i % 25 == 0:
                acc = sum(r["correct"] for r in records if r["tier"] == tier) / i
                print(f"  {i}/{len(cases)} acc {acc:.1%} ({time.time()-t0:.0f}s)", flush=True)
        acc = sum(r["correct"] for r in records if r["tier"] == tier) / len(cases)
        print(f"  {tier} OVERALL {acc:.1%}", flush=True)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nwrote {args.out} ({len(records)} records)")

    # Does confidence separate right from wrong?
    import statistics
    for tier in args.tiers.split(","):
        rs = [r for r in records if r["tier"] == tier]
        right = [r["confidence"] for r in rs if r["correct"]]
        wrong = [r["confidence"] for r in rs if not r["correct"]]
        if right and wrong:
            # AUC via rank comparison: P(conf_right > conf_wrong)
            wins = sum(1 for a in right for b in wrong if a > b)
            ties = sum(1 for a in right for b in wrong if a == b)
            auc = (wins + 0.5 * ties) / (len(right) * len(wrong))
            print(f"\n{tier}: conf(correct) median {statistics.median(right):.3f} | "
                  f"conf(wrong) median {statistics.median(wrong):.3f} | AUC {auc:.3f}")


if __name__ == "__main__":
    main()
