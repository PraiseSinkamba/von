"""Generate two-hop decision scenarios with analytically derived posteriors.

Motivation
----------
Von fails specifically on compositional families (multi_hop, long_policy,
tradeoff, judge_hard) where it scores *below* the chance baseline, which means
it is being pulled toward distractors rather than merely guessing. It also
trains on one-hot labels, so it can only ever learn certainty.

This generator addresses both without ingesting anybody's model outputs. Each
scenario is built from two evidence clauses whose reliabilities are chosen
first; the answer distribution is then *computed* from them in log-odds space.
The label is therefore ground truth by construction rather than an estimate,
which is a stronger signal than a distilled soft label: JevBench's probability
items are scored against true Bayesian posteriors, not against another model's
opinion.

Design constraints, each one earned from a previous failure
-----------------------------------------------------------
1. Composition is mandatory. Clause weights are drawn so that the two clauses
   disagree often enough that either clause alone points at the wrong answer in
   a large share of items. `--audit` measures this directly; a corpus that can
   be solved from one clause is not teaching two-hop reasoning.

2. The shortcut must not pay. Option wording is drawn from a vocabulary that is
   deliberately disjoint from evidence wording, so the correct option is not
   systematically the one sharing the most words with the premise. Our previous
   synthetic generator failed this and plausibly trained distractor attraction.

3. No single surface template. The distilled corpus we mined was 100% one
   template ("...; additionally ... Context: ..."), which teaches the template.
   Templates, connectives and clause order are all randomised here.

4. Uncertainty must be real, not decorative. Posteriors come from clause
   reliability, so a scenario with weak or conflicting evidence genuinely
   lands near the prior and the target says so.

Usage:
    uv run python -m training.generate_synthetic_decisions --n 120000 \
        --out data_synth/synthetic.jsonl --audit
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

# --- Surface variation -------------------------------------------------------
# Several templates, so no single connective becomes a learnable cue.
TEMPLATES = [
    "{a}; additionally {b}. Context: {ctx}.",
    "{ctx}. Two things are known: {a}, and {b}.",
    "{a}. Separately, {b}. This is happening {ctx}.",
    "Given {ctx} — {b}, while {a}.",
    "Reviewing {ctx}: {b}. Also noted: {a}.",
    "{b}. On top of that, {a}. Setting: {ctx}.",
]

CONTEXTS = [
    "during a staged rollout", "in the final approval window", "under an active incident",
    "in a routine weekly review", "ahead of a customer deadline", "during a vendor migration",
    "in an out-of-hours on-call shift", "under a regulatory audit", "at end of quarter",
    "while a partial outage is open", "in a first-time onboarding", "during a freeze period",
]

# --- Evidence vocabulary -----------------------------------------------------
# Deliberately disjoint from OPTION_SETS wording so that lexical overlap between
# premise and option carries no information about which option is correct.
# Each clause is (text, log-odds contribution toward the positive hypothesis).
EVIDENCE: Dict[str, List[Tuple[str, float]]] = {
    "strong_for": [
        ("the originating system re-confirmed the record twice", 2.2),
        ("an independent audit trail corroborates the entry", 2.4),
        ("two separate operators signed off on the same finding", 2.0),
        ("the upstream ledger and the mirror agree exactly", 2.3),
        ("a deterministic checksum matched on replay", 2.5),
    ],
    "weak_for": [
        ("a single unreviewed note points that way", 0.7),
        ("one operator recalled it happening", 0.6),
        ("an informal summary mentions it in passing", 0.5),
        ("a stale dashboard still shows the earlier reading", 0.6),
        ("a partial sample leaned in that direction", 0.8),
    ],
    "weak_against": [
        ("the timestamps drift by several minutes", -0.7),
        ("one field was filled in by hand afterwards", -0.6),
        ("the sample size behind it was very small", -0.8),
        ("a duplicate entry was found nearby", -0.6),
        ("the reporting window closed early", -0.5),
    ],
    "strong_against": [
        ("the source system has no matching record at all", -2.4),
        ("the referenced document was withdrawn by its author", -2.2),
        ("replay produced a contradictory checksum", -2.5),
        ("the only witness later retracted the account", -2.1),
        ("an independent reconciliation found the opposite", -2.3),
    ],
}

BUCKETS = list(EVIDENCE.keys())

# Both constants are fitted, not guessed. Sweeping them against the distilled
# corpus's measured normalised target entropy (0.617) lands at opposing=0.42,
# scale=1.6 -> 0.631, while the single-clause ablation rate stays flat at ~50%
# because it is structural rather than a function of clause magnitude. Softer
# settings produced targets around 0.74, which would have trained
# underconfidence -- the opposite of the calibration problem being solved.
OPPOSING_PROB = 0.42
WEIGHT_SCALE = 1.6

# --- Option vocabularies -----------------------------------------------------
# Wording shares as little as possible with EVIDENCE above.
OPTION_SETS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("Handling for this situation.", [
        ("proceed", "Go ahead now without further checks."),
        ("proceed_with_notice", "Go ahead, but flag it for someone to review later."),
        ("ask_first", "Pause and get explicit sign-off before moving."),
        ("refuse", "Decline to act on this at all."),
    ]),
    ("Verdict for this situation.", [
        ("supported", "The claim stands up to what is available."),
        ("partially_supported", "Only part of the claim stands up."),
        ("unsupported", "Nothing available backs the claim."),
        ("contradicted", "What is available points the other way."),
    ]),
    ("Routing for this situation.", [
        ("resolve_locally", "Close it out at this level."),
        ("escalate", "Hand it to a higher authority."),
        ("defer", "Hold it until more is available."),
        ("reassign", "Move it to a different owner entirely."),
    ]),
]

NOUL_QUESTIONS = [
    "Is this a case where the finding should be treated as established?",
    "Is this a case where the record can be relied on as it stands?",
    "Is this a case where the conclusion holds up?",
]
NOUL_OPTIONS = [
    ("true", "Yes, it holds up."),
    ("false", "No, it does not hold up."),
]

SCORE_QUESTION = "Rate how well the available material supports the conclusion, 0-5."


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def pick_clauses(rng: random.Random) -> Tuple[str, float, str, float]:
    """Two clauses whose combination, not either alone, decides the answer.

    Same-direction pairs are deliberately down-weighted: when both clauses agree
    the item is solvable from one clause and teaches nothing about composition.
    """
    b1 = rng.choice(BUCKETS)
    if rng.random() < OPPOSING_PROB:
        # Opposing or mismatched-strength evidence: the two clauses must be
        # weighed against each other.
        opposing = [b for b in BUCKETS if ("for" in b) != ("for" in b1)]
        b2 = rng.choice(opposing)
    else:
        b2 = rng.choice(BUCKETS)
    t1, w1 = rng.choice(EVIDENCE[b1])
    t2, w2 = rng.choice(EVIDENCE[b2])
    w1 *= WEIGHT_SCALE
    w2 *= WEIGHT_SCALE
    return t1, w1, t2, w2


# Drawn independently of the answer, so these cannot correlate with the label.
# They exist for two reasons: they multiply scenario diversity (the clause and
# template combinatorics alone cap out near 29k distinct states), and text that
# is genuinely irrelevant teaches the model to ignore irrelevant text. That is
# only safe because the draw is independent -- our earlier long-context
# generator attached detail that tracked the answer, which is what plausibly
# taught distractor attraction.
NEUTRAL_DETAILS = [
    "The item was filed under the usual reference scheme.",
    "Two other unrelated tickets were open at the same time.",
    "The handling team rotates on a fortnightly basis.",
    "This category sees roughly forty cases a month.",
    "The originating request arrived through the standard queue.",
    "A routine backup completed earlier the same day.",
    "The relevant policy was last reformatted in the spring.",
    "Several similar items are pending elsewhere in the backlog.",
    "The workspace had been migrated to new hardware recently.",
    "An unrelated maintenance notice was posted that morning.",
    "The reference number falls in the middle of the current block.",
    "Staffing for the week matched the usual baseline.",
]


def render(rng: random.Random, a: str, b: str) -> str:
    text = rng.choice(TEMPLATES).format(a=a, b=b, ctx=rng.choice(CONTEXTS))
    if rng.random() < 0.75:
        detail = rng.choice(NEUTRAL_DETAILS)
        text = f"{text} {detail}" if rng.random() < 0.5 else f"{detail} {text}"
    return text


def make_noul(rng: random.Random) -> dict:
    t1, w1, t2, w2 = pick_clauses(rng)
    prior = rng.uniform(-0.4, 0.4)
    p_true = sigmoid(prior + w1 + w2)
    opts = list(NOUL_OPTIONS)
    rng.shuffle(opts)
    probs = {"true": p_true, "false": 1.0 - p_true}
    target = [probs[i] for i, _ in opts]
    return {
        "state": render(rng, t1, t2),
        "question": rng.choice(NOUL_QUESTIONS),
        "options": [{"id": i, "description": d} for i, d in opts],
        "label": opts[max(range(len(opts)), key=lambda k: target[k])][0],
        "target": target,
        "source": {"kind": "synth_twohop_noul"},
        "_single": [sigmoid(prior + w1), sigmoid(prior + w2)],
    }


def make_choice(rng: random.Random) -> dict:
    t1, w1, t2, w2 = pick_clauses(rng)
    question, options = rng.choice(OPTION_SETS)
    combined = w1 + w2
    # Map the combined evidence onto an ordered option scale, then soften by how
    # much the two clauses disagree: conflicting evidence means a flatter target.
    conflict = abs(w1 - w2) / 5.0
    centre = (len(options) - 1) * (1.0 - sigmoid(combined))
    sharp = max(0.6, 2.2 - conflict)
    scores = [-sharp * abs(i - centre) for i in range(len(options))]
    mx = max(scores)
    exps = [math.exp(s - mx) for s in scores]
    total = sum(exps)
    target = [e / total for e in exps]
    opts = list(options)
    order = list(range(len(opts)))
    rng.shuffle(order)
    opts = [opts[i] for i in order]
    target = [target[i] for i in order]
    return {
        "state": render(rng, t1, t2),
        "question": question,
        "options": [{"id": i, "description": d} for i, d in opts],
        "label": opts[max(range(len(target)), key=lambda k: target[k])][0],
        "target": target,
        "source": {"kind": "synth_twohop_choice"},
        "_single": None,
    }


def make_score(rng: random.Random) -> dict:
    t1, w1, t2, w2 = pick_clauses(rng)
    combined = w1 + w2
    centre = 5.0 * sigmoid(combined)
    conflict = abs(w1 - w2) / 5.0
    spread = max(0.5, 0.7 + conflict)
    scores = [-((i - centre) ** 2) / (2 * spread ** 2) for i in range(6)]
    mx = max(scores)
    exps = [math.exp(s - mx) for s in scores]
    total = sum(exps)
    target = [e / total for e in exps]
    return {
        "state": render(rng, t1, t2),
        "question": SCORE_QUESTION,
        "options": [{"id": str(i), "description": f"Rating level {i} on a 0-5 scale"}
                    for i in range(6)],
        "label": str(max(range(6), key=lambda k: target[k])),
        "target": target,
        "source": {"kind": "synth_twohop_score"},
        "_single": None,
    }


# --- Numeric/rule composition (targets temporal_numeric, a distinct skill) ----
# Unlike the evidence-weighing generators above, ground truth here is exact
# arithmetic: sum stated component quantities (deliberately requiring the
# model to combine 2-3 numbers stated in different sentences), apply a
# rounding rule, then compare against a stated threshold. This mirrors
# JevBench's temporal_numeric family (e.g. "gross weight = net cargo + tare +
# restraints, rounded up, must not exceed 2,500 kg") without copying any of
# its wording or numbers -- domain flavour, unit, and quantities are all
# randomised per record.
NUMERIC_DOMAINS = [
    {
        "item": "shipment", "unit": "kg", "cap_word": "weight limit",
        "components": ["the base cargo weighs {v} {unit}",
                        "the packaging adds {v} {unit}",
                        "the restraint hardware adds {v} {unit}"],
        "threshold_phrase": "the position's rule caps total {cap_word} at {t} {unit}",
    },
    {
        "item": "roster", "unit": "hours", "cap_word": "duty-hour limit",
        "components": ["the scheduled shift runs {v} {unit}",
                        "a mandatory briefing adds {v} {unit}",
                        "carried-over overtime adds {v} {unit}"],
        "threshold_phrase": "policy sets the {cap_word} at {t} {unit}",
    },
    {
        "item": "budget line", "unit": "thousand dollars", "cap_word": "spending cap",
        "components": ["the base allocation is {v} {unit}",
                        "an approved change order adds {v} {unit}",
                        "a contingency draw adds {v} {unit}"],
        "threshold_phrase": "the approved {cap_word} is {t} {unit}",
    },
    {
        "item": "headcount", "unit": "people", "cap_word": "occupancy limit",
        "components": ["the confirmed attendee count is {v} {unit}",
                        "walk-up registrations add {v} {unit}",
                        "staff on site add {v} {unit}"],
        "threshold_phrase": "the venue's {cap_word} is {t} {unit}",
    },
]

NUMERIC_VERDICT_OPTIONS = [
    ("complies", "The total is at or under the limit."),
    ("marginal_over", "The total exceeds the limit, but by a small margin."),
    ("major_over", "The total exceeds the limit by a wide margin."),
]


def _numeric_scenario(rng: random.Random) -> Tuple[dict, float, float, List[str]]:
    """Draws a domain, renders 2-3 component sentences, and computes the exact total.

    Returns (domain, total, threshold, component_sentences). Ground truth from
    here on is plain arithmetic -- no sigmoid, no estimation.
    """
    domain = rng.choice(NUMERIC_DOMAINS)
    n_components = rng.choice([2, 2, 3])  # mostly 2-hop, sometimes 3-hop
    chosen = rng.sample(domain["components"], n_components)
    values = [round(rng.uniform(1, 40), 1) for _ in chosen]
    total = round(sum(values), 1)
    # Threshold drawn relative to the total so both compliant and violating
    # cases, including close-margin ones, occur with real frequency.
    margin_frac = rng.uniform(-0.35, 0.35)
    threshold = round(total * (1.0 - margin_frac), 1)
    threshold = max(threshold, 1.0)
    sentences = [c.format(v=v, unit=domain["unit"]) for c, v in zip(chosen, values)]
    rng.shuffle(sentences)
    return domain, total, threshold, sentences


def _numeric_render(rng: random.Random, domain: dict, threshold: float, sentences: List[str]) -> str:
    thresh_sentence = domain["threshold_phrase"].format(cap_word=domain["cap_word"], t=threshold, unit=domain["unit"])
    parts = sentences + [thresh_sentence]
    rng.shuffle(parts)
    text = f"For this {domain['item']}: " + "; ".join(parts) + "."
    if rng.random() < 0.6:
        detail = rng.choice(NEUTRAL_DETAILS)
        text = f"{text} {detail}" if rng.random() < 0.5 else f"{detail} {text}"
    return text


def make_numeric_noul(rng: random.Random) -> dict:
    domain, total, threshold, sentences = _numeric_scenario(rng)
    margin = total - threshold
    over = margin > 0
    # Sharp but not one-hot: a record whose margin is tiny still carries a
    # touch of real uncertainty about rounding/reading precision.
    steep = 2.0
    p_over = sigmoid(steep * margin / max(threshold, 1.0) * 5.0)
    opts = [("over", "The total exceeds the stated limit."),
            ("within", "The total is at or under the stated limit.")]
    rng.shuffle(opts)
    probs = {"over": p_over, "within": 1.0 - p_over}
    target = [probs[i] for i, _ in opts]
    return {
        "state": _numeric_render(rng, domain, threshold, sentences),
        "question": f"Does this {domain['item']}'s total exceed its {domain['cap_word']}?",
        "options": [{"id": i, "description": d} for i, d in opts],
        "label": "over" if over else "within",
        "target": target,
        "source": {"kind": "synth_numeric_noul"},
        "_single": None,
    }


def make_numeric_choice(rng: random.Random) -> dict:
    domain, total, threshold, sentences = _numeric_scenario(rng)
    margin = total - threshold
    frac = margin / max(threshold, 1.0)
    if frac <= 0:
        idx = 0
    elif frac < 0.15:
        idx = 1
    else:
        idx = 2
    spread = 0.35
    scores = [-((i - idx) ** 2) / (2 * spread ** 2) for i in range(3)]
    mx = max(scores)
    exps = [math.exp(s - mx) for s in scores]
    tot = sum(exps)
    target = [e / tot for e in exps]
    opts = list(NUMERIC_VERDICT_OPTIONS)
    order = list(range(len(opts)))
    rng.shuffle(order)
    opts = [opts[i] for i in order]
    target = [target[i] for i in order]
    return {
        "state": _numeric_render(rng, domain, threshold, sentences),
        "question": f"What is the compliance verdict for this {domain['item']}?",
        "options": [{"id": i, "description": d} for i, d in opts],
        "label": opts[max(range(len(target)), key=lambda k: target[k])][0],
        "target": target,
        "source": {"kind": "synth_numeric_choice"},
        "_single": None,
    }


def generate(n: int, seed: int, dedupe: bool = True) -> List[dict]:
    """Generate n records, refusing to pad the corpus with repeats.

    Repeated scenarios are memorisable rather than learnable, so generation
    stops early and says so rather than silently emitting the same state many
    times over.
    """
    rng = random.Random(seed)
    makers = [make_noul, make_choice, make_score, make_numeric_noul, make_numeric_choice]
    weights = [0.30, 0.30, 0.15, 0.15, 0.10]
    out: List[dict] = []
    seen = set()
    stalled = 0
    while len(out) < n and stalled < 20000:
        record = rng.choices(makers, weights)[0](rng)
        if dedupe:
            key = (record["state"], record["question"])
            if key in seen:
                stalled += 1
                continue
            seen.add(key)
        stalled = 0
        out.append(record)
    if len(out) < n:
        print(f"NOTE: generated {len(out):,} distinct records, short of the "
              f"{n:,} requested; the clause/template/detail combinatorics are "
              f"exhausted. Widen EVIDENCE or NEUTRAL_DETAILS to go higher.")
    return out


def audit(records: List[dict]) -> None:
    """Prove the corpus teaches what it claims to teach."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from training.harden_corpus import overlap_scores, option_id

    rew = neu = pun = meas = 0
    ents: List[float] = []
    single_wrong = single_total = 0
    kinds: Counter = Counter()

    for r in records:
        kinds[r["source"]["kind"]] += 1
        t = r["target"]
        ents.append(-sum(p * math.log(p + 1e-12) for p in t) / math.log(len(t)))

        ids = [option_id(o) for o in r["options"]]
        sc = overlap_scores(str(r["state"]), r["options"])
        best = max(sc, default=0)
        if best > 0 and r["label"] in ids:
            meas += 1
            if sc.count(best) > 1:
                neu += 1
            elif sc[ids.index(r["label"])] == best:
                rew += 1
            else:
                pun += 1

        if r.get("_single"):
            p_full = t[ids.index(r["label"])]
            for p_one in r["_single"]:
                single_total += 1
                # Would one clause alone have pointed at the other answer?
                if (p_one > 0.5) != (p_full > 0.5 if len(t) == 2 else True):
                    single_wrong += 1

    ents.sort()
    print(f"records {len(records):,}  kinds={dict(kinds)}")
    print(f"normalised target entropy: mean {sum(ents)/len(ents):.3f}  "
          f"p25 {ents[len(ents)//4]:.3f}  p50 {ents[len(ents)//2]:.3f}  "
          f"p75 {ents[3*len(ents)//4]:.3f}  near-deterministic(<0.2) "
          f"{sum(1 for e in ents if e < 0.2)/len(ents):.1%}")
    if meas:
        print(f"overlap shortcut on {meas:,} measurable: rewards {rew/meas:.1%}  "
              f"neutral {neu/meas:.1%}  punishes {pun/meas:.1%}")
    if single_total:
        print(f"single-clause ablation: one clause alone points at the WRONG answer "
              f"{single_wrong/single_total:.1%} of the time "
              f"(high = composition genuinely required)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=120000)
    parser.add_argument("--out", default="data_synth/synthetic.jsonl")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()

    records = generate(args.n, args.seed)
    if args.audit:
        audit(records)

    tmp = args.out + ".tmp"
    try:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                r = {k: v for k, v in r.items() if not k.startswith("_")}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, args.out)
    except OSError as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"cannot write {args.out}: {exc}") from exc
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
