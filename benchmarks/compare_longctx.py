"""Compare two Option-Marker checkpoints on JevBench's hard tier by premise length.

The long-context corpus targeted one specific measured failure: accuracy decaying
from 40.7% on short premises to 23.5% on 2048-4096 token premises. This scores
both checkpoints on the same fixed external set and buckets by real token count,
so the comparison answers whether that decay moved -- not whether validation
accuracy moved on a validation set that itself changed.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, "src")

from von.backends.option_marker_backend import OptionMarkerBackend
from von.types import Choice, Noul, Score

HARD = "/tmp/jevbench/datasets/public/hard.jsonl"
BUCKETS = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 1 << 30)]


def load_cases() -> List[dict]:
    with open(HARD, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def predict(backend: OptionMarkerBackend, case: dict):
    q = case["question"]
    qtype = q.get("type", "choice")
    state = case["state"]
    if isinstance(state, (dict, list)):
        state = json.dumps(state)

    if qtype == "choice":
        obj = Choice(instructions=q["instructions"], criteria=q["criteria"])
        return backend.evaluate_choice("d", state, obj).choice
    if qtype == "noul":
        obj = Noul(instructions=q["instructions"], criteria=q.get("criteria") or {})
        ans = backend.evaluate_noul("d", state, obj)
        return "yes" if ans.noul else "no"
    obj = Score(instructions=q["instructions"], criteria=q["criteria"])
    ans = backend.evaluate_score("d", state, obj)
    probs = ans.probabilities or {}
    return max(probs, key=probs.get) if probs else str(int(round(ans.score)))


def normalise(v) -> str:
    return str(v).strip().lower()


def run(checkpoint: str, cases: List[dict], tokenizer) -> Dict[str, object]:
    backend = OptionMarkerBackend(checkpoint_dir=checkpoint, device="cpu")
    per_bucket = defaultdict(lambda: [0, 0])
    correct = 0
    t0 = time.time()

    for i, case in enumerate(cases, 1):
        state = case["state"]
        if isinstance(state, (dict, list)):
            state = json.dumps(state)
        n_tok = len(tokenizer.encode(state, add_special_tokens=False))

        try:
            pred = predict(backend, case)
        except Exception as exc:  # a crash on one case must not void the run
            print(f"  case {i} failed: {type(exc).__name__}: {exc}", flush=True)
            pred = "<error>"

        ok = normalise(pred) == normalise(case["expected"])
        correct += ok
        for lo, hi in BUCKETS:
            if lo <= n_tok < hi:
                per_bucket[(lo, hi)][0] += ok
                per_bucket[(lo, hi)][1] += 1
                break
        if i % 25 == 0:
            print(f"  {i}/{len(cases)}  running acc {correct/i:.1%}  ({time.time()-t0:.0f}s)", flush=True)

    return {"correct": correct, "total": len(cases), "buckets": dict(per_bucket)}


def main() -> None:
    from transformers import AutoTokenizer

    cases = load_cases()
    tokenizer = AutoTokenizer.from_pretrained("checkpoints/von-option-marker-universal")
    print(f"Loaded {len(cases)} hard-tier cases\n")

    results = {}
    for label, ckpt in [
        ("baseline (3ep, short-only)", "checkpoints/von-option-marker-universal"),
        ("long-context (1ep)", "checkpoints/von-longctx-e1"),
    ]:
        print(f"=== {label} :: {ckpt}")
        results[label] = run(ckpt, cases, tokenizer)
        r = results[label]
        print(f"  OVERALL {r['correct']}/{r['total']} = {r['correct']/r['total']:.1%}\n", flush=True)

    print("\n================ HARD TIER BY PREMISE LENGTH ================")
    header = f"{'bucket':>12} {'n':>4}"
    for label in results:
        header += f" {label[:22]:>24}"
    print(header)
    for lo, hi in BUCKETS:
        label_txt = f"{lo}-{hi}" if hi < (1 << 30) else f"{lo}+"
        counts = [results[k]["buckets"].get((lo, hi), [0, 0]) for k in results]
        if not counts or counts[0][1] == 0:
            continue
        row = f"{label_txt:>12} {counts[0][1]:>4}"
        for c in counts:
            row += f" {c[0]/c[1]:>23.1%}" if c[1] else f" {'-':>23}"
        print(row)

    print("\nOVERALL:")
    for label, r in results.items():
        print(f"  {label:<28} {r['correct']:>3}/{r['total']} = {r['correct']/r['total']:.1%}")


if __name__ == "__main__":
    main()
