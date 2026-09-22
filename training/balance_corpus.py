"""Rebalance Von's corpus so the lexical-overlap shortcut stops paying.

In the 290k universal corpus the correct option is the highest-overlap option
**79.8%** of the time, so "repeat the premise" is a near-optimal rule and Von
learned it. On JevBench's hard tier that correlation is deliberately broken
(gold is highest-overlap only 56.4%), and Von's accuracy collapses to near
chance while its confidence stays high -- which is also why confidence cannot
route (hard-tier AUC 0.632).

Rewriting premise text to fix this needs a synonym table broad enough for a
corpus drawn from dozens of public datasets; a hand-built one reached 12%
coverage and moved the ratio 3 points. Selection is the better lever here: it
touches no text, so it **cannot** change which answer is correct, and the
offending mass is concentrated -- 18 sources sit at >=90% gold-is-top.

So this subsamples shortcut-heavy sources and keeps every shortcut-free one,
aiming at ~32% gold-is-top to match the long-context corpus (30.4%) that the
generators already produce. The target is not 0%: a corpus where the obvious
answer is never right teaches the inverse shortcut, which JevBench's trap family
punishes just as hard.

Usage:
    uv run python -m training.balance_corpus --in data_universal/train.jsonl \
        --out data_universal/train_balanced.jsonl --target 0.32
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from training.harden_corpus import gold_is_top, option_id, overlap_scores


def source_of(record: dict) -> str:
    source = record.get("source")
    if isinstance(source, dict):
        return str(source.get("kind") or source.get("name") or "unknown")
    return str(source or "unknown")


def measurable(record: dict) -> bool:
    """Only items with some lexical signal can express the shortcut at all."""
    if record.get("label") not in [option_id(o) for o in record.get("options", [])]:
        return False
    return max(overlap_scores(str(record.get("state", "")), record["options"]), default=0) > 0


def balance(records: List[dict], target: float, seed: int = 0) -> Tuple[List[dict], dict]:
    rng = random.Random(seed)

    by_source: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: {"top": [], "rest": []})
    unmeasurable: List[int] = []
    for i, record in enumerate(records):
        if not measurable(record):
            unmeasurable.append(i)
            continue
        bucket = "top" if gold_is_top(record) else "rest"
        by_source[source_of(record)][bucket].append(i)

    n_top = sum(len(v["top"]) for v in by_source.values())
    n_rest = sum(len(v["rest"]) for v in by_source.values())
    before = n_top / (n_top + n_rest) if n_top + n_rest else 0.0

    # Keep every shortcut-free example; they are the scarce, valuable signal.
    # Then admit as many shortcut examples as the target allows, drawn
    # proportionally so no single source is wiped out and diversity survives.
    keep_top_total = int(round(target * n_rest / (1 - target))) if target < 1 else n_top
    keep_top_total = min(keep_top_total, n_top)

    kept: List[int] = list(unmeasurable)
    per_source_kept: Dict[str, int] = {}
    for source, buckets in by_source.items():
        kept.extend(buckets["rest"])
        share = len(buckets["top"]) / n_top if n_top else 0.0
        quota = int(round(keep_top_total * share))
        pool = buckets["top"][:]
        rng.shuffle(pool)
        chosen = pool[:quota]
        kept.extend(chosen)
        per_source_kept[source] = len(chosen)

    rng.shuffle(kept)
    out = [records[i] for i in kept]

    after_top = sum(1 for r in out if measurable(r) and gold_is_top(r))
    after_meas = sum(1 for r in out if measurable(r))
    stats = {
        "before_rows": len(records),
        "after_rows": len(out),
        "gold_top_before": before,
        "gold_top_after": after_top / after_meas if after_meas else 0.0,
        "dropped": len(records) - len(out),
        "sources": len(by_source),
    }
    return out, stats


def label_report(records: List[dict]) -> str:
    """Balancing must not skew labels or answer position as a side effect."""
    idx_counts: Dict[int, int] = defaultdict(int)
    yes_no: Dict[str, int] = defaultdict(int)
    total = 0
    for record in records:
        ids = [option_id(o) for o in record.get("options", [])]
        if record.get("label") not in ids:
            continue
        total += 1
        idx_counts[ids.index(record["label"])] += 1
        if set(ids) == {"yes", "no"}:
            yes_no[record["label"]] += 1
    head = ", ".join(f"{i}:{idx_counts[i]/total:.0%}" for i in sorted(idx_counts)[:5])
    yn = sum(yes_no.values())
    yn_txt = f"{yes_no.get('yes', 0)/yn:.1%} yes" if yn else "n/a"
    return f"gold index [{head}]  |  yes/no balance {yn_txt}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", dest="dst", required=True)
    parser.add_argument("--target", type=float, default=0.32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        with open(args.src, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {args.src}: {exc}") from exc

    out, stats = balance(records, args.target, args.seed)

    tmp = args.dst + ".tmp"
    try:
        os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            for record in out:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, args.dst)
    except OSError as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"cannot write {args.dst}: {exc}") from exc

    print(f"gold-is-highest-overlap  {stats['gold_top_before']:.1%} -> {stats['gold_top_after']:.1%} "
          f"(target {args.target:.0%})")
    print(f"rows {stats['before_rows']} -> {stats['after_rows']} "
          f"(dropped {stats['dropped']}, {stats['sources']} sources preserved)")
    print(f"before: {label_report(records)}")
    print(f"after : {label_report(out)}")
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
