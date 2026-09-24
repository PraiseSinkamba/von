"""Refit Von's calibration map to be honest on every tier, not just the scored one.

JevBench measures ECE on the hard tier only. Optimising that alone produces a
map that is well calibrated exactly where the benchmark looks and badly
calibrated everywhere else: the JevBench-only fit reports ~48% mean confidence
on easy items Von answers correctly 93.8% of the time (ECE 0.453).

That is a metric-gaming artifact, not calibration. A confidence number is a
product promise, so this fits against a combined objective:

    objective = w_jev * jevbench_calibration_axis
              + (1 - w_jev) * mean over tiers of 100 * (1 - ECE_tier / 0.5)

The second term forces the map to be honest on easy and standard too. The cost
on the JevBench axis is reported explicitly so the tradeoff is a decision, not
an accident.

Usage:
    uv run python benchmarks/fit_calibration_alltier.py
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List, Tuple

from fit_calibration import (
    calibration_score,
    ece_top_label,
    softmax,
    temp_for,
    tvd,
)

TIERS = ("easy", "standard", "hard")


def tier_stats(records: List[dict], params: Dict[str, float]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for tier in TIERS:
        rows = [r for r in records if r["tier"] == tier]
        if not rows:
            continue
        pairs: List[Tuple[float, bool]] = []
        for r in rows:
            probs = softmax(r["logits"], temp_for(r, params))
            pmap = dict(zip(r["labels"], probs))
            pred = max(pmap, key=lambda k: pmap[k])
            pairs.append((max(probs), pred.strip().lower() == r["expected"].strip().lower()))
        out[tier] = {
            "n": len(pairs),
            "acc": sum(1 for _, c in pairs if c) / len(pairs),
            "conf": sum(c for c, _ in pairs) / len(pairs),
            "ece": ece_top_label(pairs),
        }
    return out


def measure(records: List[dict], params: Dict[str, float]) -> Dict[str, Any]:
    stats = tier_stats(records, params)
    tvds = []
    for r in records:
        if r.get("gold_probs"):
            probs = softmax(r["logits"], temp_for(r, params))
            tvds.append(tvd(dict(zip(r["labels"], probs)), r["gold_probs"]))
    mean_tvd = sum(tvds) / len(tvds) if tvds else None
    hard_ece = stats.get("hard", {}).get("ece", 1.0)
    jev = calibration_score(hard_ece, mean_tvd)
    honesty = sum(max(0.0, 100 * (1 - s["ece"] / 0.5)) for s in stats.values()) / len(stats)
    return {
        "jev": jev,
        "honesty": honesty,
        "mean_tvd": float("nan") if mean_tvd is None else mean_tvd,
        "stats": stats,
    }


def objective(records: List[dict], params: Dict[str, float], w_jev: float) -> float:
    m = measure(records, params)
    return w_jev * m["jev"] + (1 - w_jev) * m["honesty"]


def fit(records: List[dict], w_jev: float, seed: int = 0, iters: int = 6000) -> Dict[str, float]:
    rng = random.Random(seed)
    best = {"bias": 2.0, "entropy": 6.0, "log_tokens": 0.0, "n_options": 0.0, "lo": 0.3, "hi": 12.0}
    best_score = objective(records, best, w_jev)
    keys = ["bias", "entropy", "log_tokens", "n_options"]
    for i in range(iters):
        cand = dict(best)
        k = rng.choice(keys)
        cand[k] = cand[k] + rng.gauss(0, 1.5 * (1 - i / iters) + 0.1)
        score = objective(records, cand, w_jev)
        if score > best_score:
            best, best_score = cand, score
    return best


def report(label: str, records: List[dict], params: Dict[str, float]) -> Dict[str, Any]:
    m = measure(records, params)
    print(f"\n{label}")
    print("  " + " ".join(f"{k}={params[k]:.3f}" for k in ("bias", "entropy", "log_tokens", "n_options")))
    for tier, s in m["stats"].items():
        print(f"    {tier:<9} acc {s['acc']:>6.1%}  conf {s['conf']:>6.1%}  ECE {s['ece']:.3f}")
    print(f"    JevBench Calibration axis {m['jev']:.1f}   (TVD {m['mean_tvd']:.3f})   honesty {m['honesty']:.1f}")
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="benchmarks/data/calibration_raw.json")
    parser.add_argument("--out", default="benchmarks/data/calibration_map.json")
    parser.add_argument("--w_jev", type=float, default=0.5)
    args = parser.parse_args()

    try:
        with open(args.data, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read calibration data {args.data}: {exc}") from exc

    for w in (1.0, 0.7, 0.5, 0.3):
        report(f"w_jev={w}", records, fit(records, w))

    # The random-search optimiser is seed-sensitive (same w_jev, several points
    # apart between seeds), so take the best of several restarts by objective
    # instead of trusting one draw.
    candidates = [fit(records, args.w_jev, seed=s) for s in (0, 1, 2, 3, 4, 5, 6, 7)]
    final = max(candidates, key=lambda p: objective(records, p, args.w_jev))
    m = report(f"FINAL (w_jev={args.w_jev}, best of {len(candidates)} restarts)", records, final)

    # Split-half honesty check: fit on one half, score the untouched half.
    rng = random.Random(11)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    half = len(idx) // 2
    dev = [records[i] for i in idx[:half]]
    reserve = [records[i] for i in idx[half:]]
    m_res = measure(reserve, fit(dev, args.w_jev, seed=5))
    print(f"\n  split-half RESERVE: JevBench axis {m_res['jev']:.1f}, honesty {m_res['honesty']:.1f}  <-- generalisation")

    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"map": final, "jev_axis": m["jev"], "honesty": m["honesty"],
                       "reserve_jev": m_res["jev"]}, f, indent=2)
    except OSError as exc:
        raise RuntimeError(f"cannot write calibration map {args.out}: {exc}") from exc
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
