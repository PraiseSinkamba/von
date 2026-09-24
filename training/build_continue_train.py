"""Assemble the von-1.3 continue-train set: numeric + adequacy + replay.

Per VON_NEXT.md's Phase 1 step 5: numeric 40k + adequacy 10k + 50k replay,
95k train / 5k val (held out proportionally from each pool, never leaked
across the split).

Replay deviation from the original plan (disclosed, not silent): the plan
called for "50k sampled from the von-1.2 universal mix", but that corpus
(`data_universal/train.jsonl`) was built and consumed entirely on the
ephemeral EC2 training instance and never uploaded anywhere -- S3 has the
resulting checkpoints and run logs, not the training rows. Re-harvesting it
would mean re-pulling every external HF source (ANLI/WANLI, banking77,
emotion, legalbench, PAWS, ...) over the network with no guarantee of
reproducing the original shuffle/composition anyway. Standing in instead:
`training.generate_synthetic_decisions.generate()`, the two-hop/noul/choice/
score reasoning generator -- local, network-free, already audited this
session (no lexical-overlap shortcut baked in), and a legitimate general-
reasoning proxy for "don't forget everything else" during a 1-epoch,
half-original-lr continue-train.

Usage:
    uv run python -m training.build_continue_train \
        --numeric data_numeric/numeric.jsonl --adequacy data_adequacy/adequacy.jsonl \
        --replay-n 50000 --val-n 5000 --out-dir data_continue --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List

from training.generate_numeric_decisions import write_jsonl
from training.generate_synthetic_decisions import generate as generate_replay


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_val(records: List[dict], frac: float, rng: random.Random) -> tuple:
    records = list(records)
    rng.shuffle(records)
    n_val = max(1, int(round(len(records) * frac)))
    return records[n_val:], records[:n_val]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numeric", default="data_numeric/numeric.jsonl")
    ap.add_argument("--adequacy", default="data_adequacy/adequacy.jsonl")
    ap.add_argument("--replay-n", type=int, default=50000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--out-dir", default="data_continue")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    numeric = load_jsonl(args.numeric)
    adequacy = load_jsonl(args.adequacy)
    print(f"numeric: {len(numeric):,}  adequacy: {len(adequacy):,}")

    print(f"generating {args.replay_n:,} replay rows "
          f"(training.generate_synthetic_decisions, network-free)...")
    replay = generate_replay(args.replay_n, seed=args.seed + 1)
    for r in replay:
        r.setdefault("family", "replay_synth_twohop")
    print(f"replay: {len(replay):,}")

    train: List[dict] = []
    val: List[dict] = []
    for pool, name in ((numeric, "numeric"), (adequacy, "adequacy"), (replay, "replay")):
        t, v = split_val(pool, args.val_frac, rng)
        train.extend(t)
        val.extend(v)
        print(f"  {name}: {len(t):,} train / {len(v):,} val")

    rng.shuffle(train)
    rng.shuffle(val)

    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path = os.path.join(args.out_dir, "val.jsonl")
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)
    print(f"\nWrote {len(train):,} train rows to {train_path}")
    print(f"Wrote {len(val):,} val rows to {val_path}")


if __name__ == "__main__":
    main()
