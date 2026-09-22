"""Harvest SargeDev/jev-distill-corpus-v3 into Von's record format.

The corpus ships three streams and they are not equally useful. Measured over
all 655,806 train rows:

    stream       rows      uniform/no-signal   soft targets
    yuri_v3      443,977               0.3%          94.1%
    yuri_v1      137,203             100.0%              -
    openjev_v2    74,626               1.4%           4.2%

`yuri_v1` is dropped outright: every one of its 137,203 rows carries the target
[0.5, 0.5]. A uniform distribution states that both answers are equally likely,
so there is no gradient to learn from and 21% of the dataset teaches nothing.

`yuri_v3` carries genuinely soft distributions distilled from Jev 1.13, which is
the point of using it: Von currently trains on one-hot labels, so it can only
ever learn to be certain. Soft targets are what the TVD half of JevBench's
Calibration axis actually scores against.

Because those labels come from a competitor's model, any benchmark submission
trained on them must say so. Disclosure is cheap; discovery is not.

Score rows carry bare digit options ('0'..'5') whose meaning lives in the
question rather than the option, so they get a synthesised ordinal description
-- the Option-Marker scores option text and a bare digit gives it nothing to
read.

Usage:
    uv run python -m training.prepare_distill_dataset \
        --src /tmp/jd_train.jsonl --out data_distill/distill.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional

# yuri_v1 is excluded by default: 100% uniform targets, zero learnable signal.
DEFAULT_STREAMS = ("yuri_v3", "openjev_v2")

UNIFORM_EPS = 1e-9


def render_state(state) -> str:
    if isinstance(state, str):
        stripped = state.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                state = json.loads(stripped)
            except json.JSONDecodeError:
                return state
        else:
            return state
    if isinstance(state, dict):
        return "\n".join(
            f"{k.replace('_', ' ')}: {' '.join(str(x) for x in v) if isinstance(v, list) else v}"
            for k, v in state.items()
        )
    return str(state)


def describe(kind: str, option: str, index: int, total: int) -> str:
    """Option text the marker can actually read."""
    text = str(option).strip()
    if kind == "score":
        # '0'..'5' carry no meaning on their own; the scale lives in the question.
        return f"Rating level {text} on a 0-{total - 1} scale"
    if kind == "noul":
        return ("Yes, the described condition holds."
                if text.lower() in ("true", "yes", "1")
                else "No, the described condition does not hold.")
    return text.replace("_", " ")


def convert(row: dict, keep_soft: bool) -> Optional[dict]:
    kind = row.get("kind")
    options = row.get("options") or []
    target = row.get("target") or []
    if kind not in ("noul", "choice", "score"):
        return None
    if len(options) < 2 or len(target) != len(options):
        return None
    if max(target) - min(target) < UNIFORM_EPS:
        return None  # uniform: no signal to learn

    state = render_state(row.get("state", ""))
    question = str(row.get("question") or "").strip()
    if not state.strip() or not question:
        return None

    total = len(options)
    built = [
        {"id": str(o), "description": describe(kind, o, i, total)}
        for i, o in enumerate(options)
    ]
    gold = str(options[target.index(max(target))])

    record = {
        "state": state,
        "question": question,
        "options": built,
        "label": gold,
        "source": {"kind": f"distill_{row.get('source')}_{kind}"},
    }
    if keep_soft:
        # Kept aligned with `options`; the trainer uses it when present and
        # falls back to the hard label when it is not.
        record["target"] = [float(t) for t in target]
    return record


def harvest(src: str, streams: List[str], per_stream: int, keep_soft: bool) -> tuple:
    out: List[dict] = []
    counts: Counter = Counter()
    dropped: Counter = Counter()
    try:
        handle = open(src, "r", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot read {src}: {exc}") from exc
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            stream = row.get("source")
            if isinstance(stream, dict):
                # Already converted to Von's format (source is {"kind": ...});
                # pass it through rather than re-deriving it.
                out.append(row)
                counts[str(stream.get("kind", "preconverted"))] += 1
                continue
            stream = str(stream)
            if stream not in streams:
                dropped[stream] += 1
                continue
            if counts[stream] >= per_stream:
                continue
            record = convert(row, keep_soft)
            if not record:
                dropped[f"{stream}:unusable"] += 1
                continue
            out.append(record)
            counts[stream] += 1
    return out, counts, dropped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/tmp/jd_train.jsonl")
    parser.add_argument("--out", default="data_distill/distill.jsonl")
    parser.add_argument("--streams", default=",".join(DEFAULT_STREAMS))
    parser.add_argument("--per_stream", type=int, default=200000)
    parser.add_argument("--hard_only", action="store_true",
                        help="drop soft targets and keep only the argmax label")
    args = parser.parse_args()

    streams = [s.strip() for s in args.streams.split(",") if s.strip()]
    records, counts, dropped = harvest(args.src, streams, args.per_stream, not args.hard_only)

    tmp = args.out + ".tmp"
    try:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, args.out)
    except OSError as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"cannot write {args.out}: {exc}") from exc

    print(f"kept {len(records):,} records")
    for stream, n in counts.most_common():
        print(f"  {stream:<14} {n:>8,}")
    skipped = {k: v for k, v in dropped.items() if str(k).endswith("unusable")}
    if skipped:
        print("unusable (uniform/malformed):")
        for k, v in skipped.items():
            print(f"  {k:<24} {v:>8,}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
