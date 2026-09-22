"""Can Von's own confidence tell its right answers from its wrong ones?

This is the load-bearing question for any cascade. If confidence does not
separate correct from incorrect, no threshold can choose what to escalate, and
the only options left are escalate-everything (which destroys the Speed and Cost
axes that are Von's whole advantage) or retraining.

Measured as AUC = P(score of a correct item > score of an incorrect item), the
probability that a randomly chosen correct answer outranks a randomly chosen
wrong one. 0.5 is coin-flip separation, 1.0 is perfect.

Two candidate routing signals are compared:
  top1        - the calibrated confidence of the chosen option
  margin      - top1 minus top2, which measures how nearly the model tied

Margin is the one to beat: a model can be uniformly overconfident (bad top1
separation) while still hesitating measurably on the items it gets wrong.

Runs entirely off cached logits from collect_calibration_data.py, so it costs
nothing and is re-runnable.

Usage:
    uv run python benchmarks/routing_auc.py
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Sequence, Tuple

from fit_calibration import softmax, temp_for


def auc(scores_correct: Sequence[float], scores_wrong: Sequence[float]) -> float:
    """Mann-Whitney U / rank-based AUC, ties counted as half."""
    if not scores_correct or not scores_wrong:
        return float("nan")
    wins = ties = 0
    for c in scores_correct:
        for w in scores_wrong:
            if c > w:
                wins += 1
            elif c == w:
                ties += 1
    return (wins + 0.5 * ties) / (len(scores_correct) * len(scores_wrong))


def score_records(records: List[dict], params: Dict[str, float]) -> List[dict]:
    out = []
    for rec in records:
        probs = softmax(rec["logits"], temp_for(rec, params))
        pmap = dict(zip(rec["labels"], probs))
        pred = max(pmap, key=lambda k: pmap[k])
        ordered = sorted(probs, reverse=True)
        top1 = ordered[0]
        top2 = ordered[1] if len(ordered) > 1 else 0.0
        out.append({
            "tier": rec["tier"],
            "family": rec.get("family"),
            "correct": pred.strip().lower() == rec["expected"].strip().lower(),
            "top1": top1,
            "margin": top1 - top2,
            "n_options": rec["n_options"],
        })
    return out


def report_auc(rows: List[dict]) -> None:
    print(f"{'tier':<10} {'n':>4} {'acc':>7} {'AUC top1':>9} {'AUC margin':>11}")
    for tier in ("easy", "standard", "hard", "ALL"):
        sub = rows if tier == "ALL" else [r for r in rows if r["tier"] == tier]
        if not sub:
            continue
        ok = [r for r in sub if r["correct"]]
        bad = [r for r in sub if not r["correct"]]
        acc = len(ok) / len(sub)
        a1 = auc([r["top1"] for r in ok], [r["top1"] for r in bad])
        am = auc([r["margin"] for r in ok], [r["margin"] for r in bad])
        print(f"{tier:<10} {len(sub):>4} {acc:>6.1%} {a1:>9.3f} {am:>11.3f}")


def cascade_sweep(hard_rows: List[dict], signal: str, second_stage_acc: float) -> None:
    """What hard-tier accuracy results from escalating the least-confident items.

    Escalated items are assumed to be answered by a stronger model at a fixed
    accuracy. That is optimistic -- a real second stage is not uniformly good --
    but it bounds what any router on this signal can deliver.
    """
    ordered = sorted(hard_rows, key=lambda r: r[signal])
    n = len(ordered)
    base = sum(1 for r in ordered if r["correct"]) / n
    print(f"\n  escalating lowest-{signal}, second stage at {second_stage_acc:.0%} accuracy "
          f"(hard tier baseline {base:.1%})")
    print(f"  {'fire rate':>10} {'escalated':>10} {'kept acc':>9} {'final acc':>10}")
    for rate in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0):
        k = int(round(n * rate))
        escalated, kept = ordered[:k], ordered[k:]
        kept_correct = sum(1 for r in kept if r["correct"])
        final = (kept_correct + second_stage_acc * k) / n
        kept_acc = kept_correct / len(kept) if kept else float("nan")
        print(f"  {rate:>9.0%} {k:>10} {kept_acc:>8.1%} {final:>10.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="benchmarks/data/calibration_raw.json")
    parser.add_argument("--calib", default="checkpoints/von-option-marker-universal/marker_calibration.json")
    args = parser.parse_args()

    try:
        with open(args.data, "r", encoding="utf-8") as f:
            records = json.load(f)
        with open(args.calib, "r", encoding="utf-8") as f:
            params = json.load(f)["calibration_map"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load routing inputs: {exc}") from exc

    rows = score_records(records, params)
    print("Can confidence separate right from wrong? (AUC 0.5 = useless)\n")
    report_auc(rows)

    hard = [r for r in rows if r["tier"] == "hard"]
    best = "margin" if auc([r["margin"] for r in hard if r["correct"]],
                           [r["margin"] for r in hard if not r["correct"]]) >= \
                       auc([r["top1"] for r in hard if r["correct"]],
                           [r["top1"] for r in hard if not r["correct"]]) else "top1"
    print(f"\nbetter hard-tier signal: {best}")

    for acc in (0.66, 0.85, 1.0):
        cascade_sweep(hard, best, acc)


if __name__ == "__main__":
    main()
