"""Break the lexical-overlap shortcut in Von's training corpus.

Measured problem: in the 290k universal corpus the correct option is the one
sharing the most words with the premise **79.8%** of the time. "Pick whatever
repeats the premise" is therefore a near-optimal rule on our training data, and
Von learned it. JevBench's hard tier deliberately decorrelates overlap from
truth (gold is highest-overlap only 56.4% there), and Von keeps applying the
rule -- picking the highest-overlap option 86.8% of the time on binary hard
items whose gold is highest-overlap only 63.2% of the time.

The fix is to make the shortcut stop paying: paraphrase premise wording so the
correct option is no longer the lexical twin of the premise.

Design constraint: this must not be able to change which answer is correct.
Rewriting option text can silently flip the label -- inject premise words into a
wrong option and it may become right. So only the *premise* is touched, and only
by substituting words with meaning-preserving synonyms. Facts, numbers,
negations and quantifiers are never rewritten, because those carry the decision.

Target is roughly 30% gold-is-highest-overlap, matching the long-context corpus
that already sits at 30.4% -- not 0%. Driving it to zero would just teach the
inverse shortcut ("the obvious option is always wrong"), which JevBench's trap
family would punish just as hard.

Usage:
    uv run python -m training.harden_corpus --in data_universal/train.jsonl \
        --out data_universal/train_hardened.jsonl --target 0.32
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Dict, List, Sequence, Set, Tuple

# Meaning-preserving substitutions only. Nothing here changes a fact, a
# quantity, a polarity or a temporal relation -- those decide the answer.
SYNONYMS: Dict[str, str] = {
    "purchase": "acquisition", "purchased": "acquired", "buy": "procure", "bought": "procured",
    "refund": "reimbursement", "refunded": "reimbursed", "payment": "remittance", "pay": "remit",
    "customer": "client", "customers": "clients", "user": "account holder", "users": "account holders",
    "order": "transaction", "orders": "transactions", "delivery": "shipment", "delivered": "dispatched",
    "damage": "impairment", "damaged": "impaired", "broken": "non-functional", "defective": "faulty",
    "request": "application", "requested": "applied for", "approve": "authorise", "approved": "authorised",
    "reject": "decline", "rejected": "declined", "cancel": "terminate", "cancelled": "terminated",
    "urgent": "time-critical", "severe": "acute", "critical": "grave", "mild": "slight",
    "error": "fault", "errors": "faults", "issue": "problem", "issues": "problems",
    "account": "profile", "invoice": "billing statement", "subscription": "recurring plan",
    "employee": "staff member", "employees": "staff members", "manager": "supervisor",
    "contract": "agreement", "agreement": "arrangement", "policy": "rulebook",
    "document": "record", "documents": "records", "report": "writeup", "reports": "writeups",
    "complaint": "grievance", "warning": "advisory", "notice": "notification",
    "increase": "rise", "increased": "rose", "decrease": "drop", "decreased": "fell",
    "begin": "commence", "began": "commenced", "start": "initiate", "started": "initiated",
    "finish": "conclude", "finished": "concluded", "end": "cessation",
    "help": "assistance", "support": "assistance", "problem": "difficulty",
    "message": "communication", "email": "electronic mail", "call": "telephone contact",
    "price": "rate", "cost": "charge", "fee": "levy", "discount": "reduction",
    "shipping": "carriage", "package": "parcel", "item": "article", "items": "articles",
    "repair": "rectification", "replace": "substitute", "replacement": "substitution",
    "claim": "submission", "claims": "submissions", "coverage": "protection",
    "device": "unit", "machine": "apparatus", "system": "platform",
    "patient": "individual", "doctor": "clinician", "treatment": "intervention",
    "vehicle": "automobile", "driver": "operator", "accident": "collision",
    "food": "foodstuff", "meal": "dish", "allergy": "hypersensitivity",
    "travel": "journey", "trip": "excursion", "flight": "air service", "hotel": "lodging",
    "meeting": "session", "schedule": "timetable", "deadline": "cut-off",
    "security": "safeguarding", "breach": "violation", "attack": "intrusion",
    "transfer": "transmission", "withdraw": "draw down", "deposit": "lodgement",
}

# Never rewritten: these decide answers rather than describe them.
PROTECTED = {
    "not", "no", "never", "none", "nor", "cannot", "without", "except", "unless",
    "all", "any", "some", "every", "only", "must", "shall", "may", "before", "after",
    "if", "then", "else", "and", "or", "but",
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")
TOKEN_RE = re.compile(r"[a-z]{4,}")


def content_tokens(text: str) -> Set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def option_text(option) -> str:
    if isinstance(option, dict):
        return str(option.get("description") or option.get("text") or option.get("id") or "")
    return str(option)


def option_id(option) -> str:
    return str(option.get("id")) if isinstance(option, dict) else str(option)


def overlap_scores(state: str, options: Sequence) -> List[int]:
    st = content_tokens(state)
    return [len(content_tokens(option_text(o)) & st) for o in options]


def gold_is_top(record: dict) -> bool:
    options = record["options"]
    ids = [option_id(o) for o in options]
    if record["label"] not in ids:
        return False
    scores = overlap_scores(str(record["state"]), options)
    if max(scores, default=0) == 0:
        return False
    return scores[ids.index(record["label"])] == max(scores)


def deoverlap(state: str, gold_text: str, rng: random.Random) -> Tuple[str, int]:
    """Rewrite premise words that the gold option echoes, preserving meaning."""
    gold_tokens = content_tokens(gold_text)
    swaps = 0

    def replace(match: re.Match) -> str:
        nonlocal swaps
        word = match.group(0)
        low = word.lower()
        if low in PROTECTED or low not in gold_tokens:
            return word
        synonym = SYNONYMS.get(low)
        if not synonym:
            return word
        swaps += 1
        if word[0].isupper():
            return synonym[0].upper() + synonym[1:]
        return synonym

    return WORD_RE.sub(replace, state), swaps


def harden(records: List[dict], target: float, seed: int = 0) -> Tuple[List[dict], dict]:
    rng = random.Random(seed)
    scored = [(i, gold_is_top(r)) for i, r in enumerate(records)]
    top_idx = [i for i, is_top in scored if is_top]
    measurable = [i for i, r in enumerate(records)
                  if max(overlap_scores(str(r["state"]), r["options"]), default=0) > 0
                  and r["label"] in [option_id(o) for o in r["options"]]]
    before = len(top_idx) / len(measurable) if measurable else 0.0

    # Only rewrite enough records to reach the target; leave the rest intact so
    # the corpus keeps genuine cases where the obvious answer is the right one.
    keep = int(round(target * len(measurable)))
    to_fix = max(0, len(top_idx) - keep)
    rng.shuffle(top_idx)

    out = [dict(r) for r in records]
    rewritten = attempted = 0
    for i in top_idx[:to_fix]:
        record = out[i]
        ids = [option_id(o) for o in record["options"]]
        gold_text = option_text(record["options"][ids.index(record["label"])])
        attempted += 1
        new_state, swaps = deoverlap(str(record["state"]), gold_text, rng)
        if swaps:
            record["state"] = new_state
            record["hardened"] = True
            rewritten += 1

    after_top = sum(1 for i in measurable if gold_is_top(out[i]))
    stats = {
        "records": len(records),
        "measurable": len(measurable),
        "gold_top_before": before,
        "gold_top_after": after_top / len(measurable) if measurable else 0.0,
        "attempted": attempted,
        "rewritten": rewritten,
    }
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", dest="dst", required=True)
    parser.add_argument("--target", type=float, default=0.32,
                        help="desired share of items where gold is the highest-overlap option")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        with open(args.src, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {args.src}: {exc}") from exc

    out, stats = harden(records, args.target, args.seed)

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
    print(f"rewrote {stats['rewritten']}/{stats['attempted']} targeted records "
          f"of {stats['records']} total ({stats['measurable']} measurable)")
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
