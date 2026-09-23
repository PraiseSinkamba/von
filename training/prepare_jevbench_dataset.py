"""Harvest the public `Praveenrajus/jev-bench` corpus into Von's record format.

Two of its tasks target Von's measured failure modes directly:

  paws        8k paraphrase-adversary pairs. Lexical similarity carries *zero*
              signal about the label -- measured mean Jaccard 0.890 for "not a
              paraphrase" against 0.893 for "paraphrase", medians equal at
              0.889. A model cannot shortcut it on word overlap, which is
              exactly the shortcut Von learned (correct option is the
              highest-overlap option 79.8% of the time in our own corpus).

  strategyqa  implicit multi-hop questions, the composition depth Von lacks on
              JevBench's hard tier.

The rest are kept for breadth; the overlap rebalance in balance_corpus decides
how much of each actually survives into training.

Licences are per-row in `meta`, so they are checked per record rather than
assumed from the dataset card, and the licence set is reported at the end.

Noul options get descriptive text rather than bare "yes"/"no": the Option-Marker
scores option *descriptions*, so a bare polarity token carries no signal for it
to read.

Usage:
    uv run python -m training.prepare_jevbench_dataset --out data_jevbench/extra.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from collections import Counter
from typing import Dict, List, Optional

BASE = "https://huggingface.co/datasets/Praveenrajus/jev-bench/resolve/main/data"

# Tasks worth pulling, with the per-task yes/no phrasing used for Noul items.
# (positive description, negative description)
TASKS: Dict[str, Optional[tuple]] = {
    "paws": ("The two sentences state the same thing in different words.",
             "The two sentences differ in meaning despite sharing wording."),
    "strategyqa_closed": ("Yes, the answer to the question is affirmative.",
                          "No, the answer to the question is negative."),
    "strategyqa_grounded": ("Yes, the answer to the question is affirmative.",
                            "No, the answer to the question is negative."),
    "fever_evidence": ("The evidence supports the claim.",
                       "The evidence does not support the claim."),
    "boolq": ("Yes, the passage supports an affirmative answer.",
              "No, the passage does not support an affirmative answer."),
    # arc_challenge and mmlu deliberately excluded: both are scored benchmarks
    # (arc_challenge is display-only, mmlu is a scored panel benchmark) on the
    # Decision Index, drawn from the same public test splits jev-bench mirrors.
    # mnli stays excluded: research-use-only licence (see harden_corpus notes),
    # never actually intended to be live here despite being listed before.
}

TRUE_LABELS = {"1", "yes", "true", "entailment"}


def fetch(task: str, split: str, cache_dir: str) -> List[dict]:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{task}_{split}.jsonl")
    if not os.path.exists(path):
        url = f"{BASE}/{task}/{split}.jsonl"
        try:
            urllib.request.urlretrieve(url, path)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            print(f"  {task}/{split}: unavailable ({exc})")
            return []
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        print(f"  {task}/{split}: unreadable ({exc})")
        return []
    return rows


def render_state(state) -> str:
    """Flatten the JSON state into prose-ish text, matching the rest of the corpus."""
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except json.JSONDecodeError:
            return state
    if not isinstance(state, dict):
        return str(state)
    parts = []
    for key, value in state.items():
        if isinstance(value, list):
            value = " ".join(str(v) for v in value)
        parts.append(f"{key.replace('_', ' ')}: {value}")
    return "\n".join(parts)


def convert(row: dict, task: str) -> Optional[dict]:
    try:
        question = row["question"]
        question = json.loads(question) if isinstance(question, str) else question
    except (KeyError, json.JSONDecodeError):
        return None

    state = render_state(row.get("state", ""))
    if not state.strip():
        return None
    instructions = str(question.get("instructions") or "").strip()
    label = str(row.get("label", "")).strip()
    primitive = row.get("primitive")

    if primitive == "noul":
        phrasing = TASKS.get(task)
        if not phrasing:
            return None
        pos, neg = phrasing
        options = [{"id": "yes", "description": pos}, {"id": "no", "description": neg}]
        gold = "yes" if label.lower() in TRUE_LABELS else "no"
    elif primitive == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) < 2:
            return None
        options = [{"id": str(k), "description": str(v)} for k, v in criteria.items()]
        gold = label
        if gold not in [o["id"] for o in options]:
            return None
    else:
        return None

    return {
        "state": state,
        "question": instructions or "Which option applies?",
        "options": options,
        "label": gold,
        "source": {"kind": f"jevbench_{task}"},
    }


def harvest(tasks: List[str], per_task: int, cache_dir: str) -> tuple:
    out: List[dict] = []
    licences: Counter = Counter()
    for task in tasks:
        rows = fetch(task, "train", cache_dir)
        kept = 0
        for row in rows:
            if kept >= per_task:
                break
            record = convert(row, task)
            if not record:
                continue
            meta = row.get("meta")
            try:
                meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
            except json.JSONDecodeError:
                meta = {}
            licences[str(meta.get("license", "unknown"))] += 1
            out.append(record)
            kept += 1
        print(f"  {task:<22} {kept:>6} records")
    return out, licences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data_jevbench/extra.jsonl")
    parser.add_argument("--per_task", type=int, default=8000)
    parser.add_argument("--cache_dir", default="/tmp/jevbench_src")
    parser.add_argument("--tasks", default=",".join(TASKS))
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"Harvesting {len(tasks)} tasks from Praveenrajus/jev-bench")
    records, licences = harvest(tasks, args.per_task, args.cache_dir)

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

    print(f"\nwrote {len(records):,} records to {args.out}")
    print("licences present:")
    for name, count in licences.most_common():
        print(f"  {count:>7,}  {name}")


if __name__ == "__main__":
    main()
