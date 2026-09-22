"""Measure how far Von's option markers can actually read.

ModernBERT-large uses a 128-token sliding window on ~19 of its 28 layers, with
global attention only every 3rd layer. Von's 290k training corpus had a maximum
premise of 116 tokens, so in 100% of training examples the premise, question and
every option marker fit inside a single local window. The global-attention
pathway was therefore never exercised.

This probe isolates reading distance from reasoning. The task is pure lookup --
a single literal token to recover, no composition, no inference. If accuracy
collapses as the answer is pushed away from the markers, the model physically
cannot read its own premise past a certain distance, and no amount of reasoning
data can fix that.

Usage:
    uv run python benchmarks/probe_marker_distance.py [--checkpoint DIR] [--samples N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, "src")

CODE_WORDS = [
    "ZEBRA", "FALCON", "MARLIN", "OTTER", "IBEX", "HERON",
    "JACKAL", "LEMUR", "NARWHAL", "QUAIL", "TAPIR", "VIPER",
]

# Neutral filler with no lexical overlap with the codes or the question.
FILLER_SENTENCES = [
    "The quarterly review was scheduled for the second week of the period.",
    "All submitted forms are retained according to the standard retention plan.",
    "Participants may request a copy of the summary at any reasonable time.",
    "The committee meets on alternating weeks unless otherwise announced.",
    "Supporting material should be filed before the close of business.",
    "Routine maintenance is performed without prior notice to occupants.",
    "Printed copies are available at the front desk during opening hours.",
    "The archive is organised by year and then by department reference.",
    "Attendance is recorded for planning purposes only and is not binding.",
    "Any revision supersedes the previously circulated draft of the notice.",
]


def build_filler(rng: random.Random, target_tokens: int, tokenizer) -> str:
    """Grow neutral text until it reaches roughly `target_tokens` tokens."""
    if target_tokens <= 0:
        return ""
    out: List[str] = []
    while True:
        out.append(rng.choice(FILLER_SENTENCES))
        if len(out) % 4 == 0:
            n = len(tokenizer.encode(" ".join(out), add_special_tokens=False))
            if n >= target_tokens:
                break
        if len(out) > 4000:
            break
    return " ".join(out)


def make_case(rng: random.Random, distance_tokens: int, tokenizer) -> Tuple[str, List[str], str]:
    """A single-fact lookup with `distance_tokens` of filler after the fact.

    The fact is placed FIRST and filler follows, so the distance between the
    evidence and the option markers (which sit at the end of the packed
    sequence) is what the sweep actually varies.
    """
    options = rng.sample(CODE_WORDS, 4)
    answer = rng.choice(options)
    fact = f"The authorized access code is {answer}."
    filler = build_filler(rng, distance_tokens, tokenizer)
    state = f"{fact} {filler}".strip()
    return state, options, answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/von-option-marker-universal")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--out", default="benchmarks/data/marker_distance.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from von.backends.option_marker_backend import OptionMarkerBackend
    from von.types import Choice

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    backend = OptionMarkerBackend(checkpoint_dir=args.checkpoint, device="cpu")

    distances = [0, 32, 64, 128, 192, 256, 384, 512, 1024, 2048, 4096]
    results: Dict[str, Dict[str, float]] = {}

    print(f"checkpoint: {args.checkpoint}")
    print(f"{'distance':>9} {'n':>4} {'accuracy':>9} {'chance':>7} {'sec/case':>9}")

    for dist in distances:
        rng = random.Random(1234 + dist)
        n = args.samples if dist <= 512 else max(20, args.samples // 2)
        correct = 0
        t0 = time.time()
        for _ in range(n):
            state, options, answer = make_case(rng, dist, tokenizer)
            q = Choice(
                instructions="What is the authorized access code?",
                criteria={o: o for o in options},
            )
            try:
                pred = backend.evaluate_choice("d", state, q).choice
            except Exception as exc:
                print(f"    case failed: {type(exc).__name__}: {exc}")
                continue
            correct += (pred == answer)
        elapsed = time.time() - t0
        acc = correct / n
        results[str(dist)] = {"n": n, "accuracy": acc, "sec_per_case": elapsed / n}
        print(f"{dist:>9} {n:>4} {acc:>8.1%} {0.25:>7.0%} {elapsed/n:>9.2f}", flush=True)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
