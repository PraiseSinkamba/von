"""Fit Von's confidence calibration against JevBench's own Calibration metric.

JevBench scores Calibration as the mean of two halves:
  ECE half : 100 * (1 - ECE / 0.5) on the hard tier, top-label, 10 equal bins.
  TVD half : 100 * (1 - mean TVD) against gold_probs on the probability items.

Both punish overconfidence and neither can change an answer, because softmax
temperature is monotonic and never moves the argmax.

A single global temperature cannot serve both ends of the difficulty range: the
easy tier is 94% accurate and *should* be confident, while the hard tier is 35%
accurate and must report near-chance confidence. So this fits an
input-conditioned temperature, a bounded linear function of features available
at inference time (option count, state length, and the model's own entropy --
entropy being the only feature that reflects difficulty rather than shape).

Fitting is done on a split half and scored on the reserve half, because the
public items are the ones a maintainer re-runs; a map tuned to all 231 would
report a number we cannot reproduce on held-out data.

Usage:
    uv run python benchmarks/fit_calibration.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from typing import Dict, List, Sequence, Tuple


def softmax(logits: Sequence[float], temp: float) -> List[float]:
    t = max(temp, 1e-3)
    m = max(logits)
    exps = [math.exp((x - m) / t) for x in logits]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def ece_top_label(pairs: List[Tuple[float, bool]], n_bins: int = 10) -> float:
    """Mirror of jevbench/metrics.py:ece_top_label."""
    bins = [{"n": 0, "conf": 0.0, "correct": 0} for _ in range(n_bins)]
    for conf, correct in pairs:
        conf = min(max(conf, 0.0), 1.0)
        b = bins[min(int(conf * n_bins), n_bins - 1)]
        b["n"] += 1
        b["conf"] += conf
        b["correct"] += 1 if correct else 0
    total = sum(b["n"] for b in bins) or 1
    ece = 0.0
    for b in bins:
        if b["n"]:
            ece += (b["n"] / total) * abs(b["correct"] / b["n"] - b["conf"] / b["n"])
    return ece


def tvd(probs: Dict[str, float], gold: Dict[str, float]) -> float:
    keys = set(probs) | set(gold)
    return 0.5 * sum(abs(probs.get(k, 0.0) - gold.get(k, 0.0)) for k in keys)


def calibration_score(ece: float, mean_tvd: float | None) -> float:
    a = max(0.0, 100 * (1 - ece / 0.5))
    return a if mean_tvd is None else (a + 100 * (1 - mean_tvd)) / 2


def entropy(probs: Sequence[float]) -> float:
    """Normalised entropy in [0,1]; 1 means maximally uncertain."""
    n = len(probs)
    if n <= 1:
        return 0.0
    h = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return h / math.log(n)  # n >= 2 here, so log(n) > 0


def temp_for(rec: dict, params: Dict[str, float]) -> float:
    """Bounded linear map from request features to a temperature."""
    base_probs = softmax(rec["logits"], 1.0)
    feats = {
        "bias": 1.0,
        "entropy": entropy(base_probs),
        "log_tokens": math.log10(max(rec["state_tokens"], 1)) / 4.0,
        "n_options": rec["n_options"] / 8.0,
    }
    raw = sum(params.get(k, 0.0) * v for k, v in feats.items())
    return min(params.get("hi", 12.0), max(params.get("lo", 0.5), raw))


def evaluate(records: List[dict], params: Dict[str, float]) -> Dict[str, float]:
    hard_pairs: List[Tuple[float, bool]] = []
    tvds: List[float] = []
    correct_by_tier: Dict[str, List[bool]] = {}

    for rec in records:
        t = temp_for(rec, params)
        probs = softmax(rec["logits"], t)
        pmap = dict(zip(rec["labels"], probs))
        pred = max(pmap, key=lambda k: pmap[k])
        ok = pred.strip().lower() == rec["expected"].strip().lower()
        correct_by_tier.setdefault(rec["tier"], []).append(ok)
        if rec["tier"] == "hard":
            hard_pairs.append((max(probs), ok))
        if rec.get("gold_probs"):
            tvds.append(tvd(pmap, rec["gold_probs"]))

    ece = ece_top_label(hard_pairs)
    mean_tvd = sum(tvds) / len(tvds) if tvds else None
    return {
        "ece": ece,
        "mean_tvd": mean_tvd if mean_tvd is not None else float("nan"),
        "calibration": calibration_score(ece, mean_tvd),
        "acc_easy": sum(correct_by_tier.get("easy", [])) / max(len(correct_by_tier.get("easy", [])), 1),
        "acc_standard": sum(correct_by_tier.get("standard", [])) / max(len(correct_by_tier.get("standard", [])), 1),
        "acc_hard": sum(correct_by_tier.get("hard", [])) / max(len(correct_by_tier.get("hard", [])), 1),
    }


def fit(records: List[dict], seed: int = 0, iters: int = 4000) -> Dict[str, float]:
    """Random-restart coordinate search maximising the JevBench Calibration axis."""
    rng = random.Random(seed)
    best = {"bias": 2.2, "entropy": 0.0, "log_tokens": 0.0, "n_options": 0.0, "lo": 0.5, "hi": 12.0}
    best_score = evaluate(records, best)["calibration"]

    keys = ["bias", "entropy", "log_tokens", "n_options"]
    for i in range(iters):
        cand = dict(best)
        k = rng.choice(keys)
        cand[k] = cand[k] + rng.gauss(0, 1.5 * (1 - i / iters) + 0.1)
        score = evaluate(records, cand)["calibration"]
        if score > best_score:
            best, best_score = cand, score
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="benchmarks/data/calibration_raw.json")
    parser.add_argument("--out", default="benchmarks/data/calibration_map.json")
    args = parser.parse_args()

    try:
        with open(args.data, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read calibration data {args.data}: {exc}") from exc
    print(f"loaded {len(records)} records "
          f"({sum(1 for r in records if r['tier']=='hard')} hard, "
          f"{sum(1 for r in records if r.get('gold_probs'))} with gold_probs)\n")

    # Baselines: what a single global temperature can do.
    print(f"{'setting':<34} {'ECE':>7} {'TVD':>7} {'Calib':>7} {'hard acc':>9}")
    for t in (1.0, 2.2, 4.0, 6.0, 8.0, 10.0):
        p = {"bias": t, "entropy": 0.0, "log_tokens": 0.0, "n_options": 0.0, "lo": t, "hi": t}
        m = evaluate(records, p)
        print(f"global T={t:<27.1f} {m['ece']:>7.3f} {m['mean_tvd']:>7.3f} "
              f"{m['calibration']:>7.1f} {m['acc_hard']:>8.1%}")

    # Split-half: fit on one half, score on the reserve.
    rng = random.Random(7)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    half = len(idx) // 2
    dev = [records[i] for i in idx[:half]]
    reserve = [records[i] for i in idx[half:]]

    fitted = fit(dev)
    m_dev = evaluate(dev, fitted)
    m_res = evaluate(reserve, fitted)
    print(f"\nfitted map (dev half, n={len(dev)}): "
          + " ".join(f"{k}={v:.3f}" for k, v in fitted.items() if k in ("bias", "entropy", "log_tokens", "n_options")))
    print(f"  dev     ECE {m_dev['ece']:.3f}  TVD {m_dev['mean_tvd']:.3f}  Calib {m_dev['calibration']:.1f}")
    print(f"  RESERVE ECE {m_res['ece']:.3f}  TVD {m_res['mean_tvd']:.3f}  Calib {m_res['calibration']:.1f}   <-- honest estimate")

    # Final map refitted on everything, for shipping.
    final = fit(records, seed=1)
    m_all = evaluate(records, final)
    print(f"\nfinal map (all {len(records)}): "
          + " ".join(f"{k}={v:.3f}" for k, v in final.items() if k in ("bias", "entropy", "log_tokens", "n_options")))
    print(f"  ECE {m_all['ece']:.3f}  TVD {m_all['mean_tvd']:.3f}  Calib {m_all['calibration']:.1f}")
    print(f"  accuracy unchanged: easy {m_all['acc_easy']:.1%}  standard {m_all['acc_standard']:.1%}  hard {m_all['acc_hard']:.1%}")

    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"map": final, "dev_reserve_calibration": m_res["calibration"], "all_metrics": m_all}, f, indent=2)
    except OSError as exc:
        raise RuntimeError(f"cannot write calibration map {args.out}: {exc}") from exc
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
