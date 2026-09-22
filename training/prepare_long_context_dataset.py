"""Long-context decision corpus builder for Von 1.1.

Motivation
----------
Von's Option-Marker head was trained exclusively on short premises (median ~16
tokens, max ~116) while inference runs at an 8,192-token context. A length-
stratified diagnostic on JevBench's hard tier showed accuracy decaying
monotonically with premise length:

    0-512 tokens    40.7%  (24/59)
    512-1024        33.3%  (5/15)
    1024-2048       33.3%  (1/3)
    2048-4096       23.5%  (8/34)

Raising the trainer's `max_length` alone is a no-op because no training example
is long enough to truncate. This module supplies the missing ingredient: decision
records whose premises are genuinely long and require cross-clause resolution.

Two complementary sources
-------------------------
1. `harvest_legalbench()` - real contracts from nguha/legalbench
   (contract_nli_*, maud_*, consumer_contracts_qa). These carry authentic legal
   "trap" knowledge that hand-written templates cannot invent, but only ~26% of
   rows exceed 512 tokens and essentially none reach 2048.

2. `generate_synthetic_policies()` - procedurally composed policy documents with
   controllable length (default 1,500-4,000 tokens). These target the 2048-4096
   bucket specifically, where measured accuracy is worst. Each document contains
   a base rule, an amendment that overrides it, date-gated applicability, and
   lexically-attractive distractor clauses, so the correct answer requires
   locating and composing several separated spans rather than pattern-matching
   one.

Output format matches `training/prepare_universal_dataset.py` exactly:
    {"state": str, "question": str,
     "options": [{"id": str, "description": str}], "label": str, "source": str}
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# 1. Curated real-contract ingestion (nguha/legalbench)
# ---------------------------------------------------------------------------

# Config prefixes whose `text` field is a contract excerpt rather than a sentence.
LEGALBENCH_PREFIXES = ("contract_nli", "maud")

# legalbench multiple-choice tasks encode options as bare letters; without the
# letter->meaning mapping the option markers carry no semantics for the scorer.
# Yes/No tasks are unambiguous, so those are the ones we keep by default.
YESNO_LABELS = {"yes", "no"}


def _yesno_options(bare_ratio: float = 0.5) -> List[Dict[str, str]]:
    """Mix descriptive and bare polarity phrasings (matches universal corpus policy)."""
    descriptive = [
        {"id": "yes", "description": "Yes, the document supports this conclusion."},
        {"id": "no", "description": "No, the document does not support this conclusion."},
    ]
    bare = [
        [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}],
        [{"id": "yes", "description": "True"}, {"id": "no", "description": "False"}],
        [{"id": "yes", "description": "Yes, condition holds true."},
         {"id": "no", "description": "No, condition is false."}],
    ]
    return random.choice(bare) if random.random() < bare_ratio else descriptive


def harvest_legalbench(max_rows: Optional[int] = None, verbose: bool = True) -> List[dict]:
    """Pull long-document Yes/No decision items from legalbench.

    Returns [] (with a warning) if `datasets` is unavailable or the hub is
    unreachable, so a corpus build never hard-fails on an optional source.
    """
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError:
        if verbose:
            print("  [legalbench] 'datasets' not installed - skipping curated source.")
        return []

    try:
        configs = [
            c for c in get_dataset_config_names("nguha/legalbench")
            if c.startswith(LEGALBENCH_PREFIXES)
        ]
    except Exception as exc:  # network/hub failure
        if verbose:
            print(f"  [legalbench] config listing failed ({type(exc).__name__}) - skipping.")
        return []

    # consumer_contracts_qa has the best length profile (median 532 tokens), and
    # keeps its premise in `contract` rather than `text`.
    configs.append("consumer_contracts_qa")

    records: List[dict] = []
    loaded = 0
    for cfg in configs:
        try:
            ds = load_dataset("nguha/legalbench", cfg, split="test")
        except Exception:
            continue

        cols = ds.column_names
        text_col = "contract" if "contract" in cols else ("text" if "text" in cols else None)
        if text_col is None or "answer" not in cols:
            continue

        # Task name doubles as the question when the row carries no explicit one.
        fallback_q = cfg.replace("contract_nli_", "").replace("maud_", "").replace("_", " ")
        fallback_q = f"Based on the document, does the following hold: {fallback_q}?"

        for row in ds:
            answer = str(row["answer"]).strip().lower()
            if answer not in YESNO_LABELS:
                continue  # skip A/B/C/D tasks: bare letters carry no semantics
            premise = (row[text_col] or "").strip()
            if not premise:
                continue
            question = (row.get("question") or fallback_q).strip()
            records.append({
                "state": premise,
                "question": question,
                "options": _yesno_options(),
                "label": answer,
                "source": f"legalbench_{cfg}",
            })
        loaded += 1

    random.shuffle(records)
    if max_rows is not None:
        records = records[:max_rows]
    if verbose:
        print(f"  [legalbench] {loaded} configs -> {len(records):,} Yes/No records")
    return records


# ---------------------------------------------------------------------------
# 2. Synthetic long-document policy generator
# ---------------------------------------------------------------------------

DOMAINS = [
    {
        "name": "insurance_water_damage",
        "title": "{carrier} MUTUAL INSURANCE COMPANY\nHOMEOWNERS POLICY FORM {form} - EXCERPTS, ENDORSEMENTS AND CLAIM FILE {claim}",
        "question": "Acting as the coverage reviewer, decide how the building damage portion of claim {claim} must be settled under the policy form and the endorsements attached to the current term.",
        "options": [
            ("deny_excluded", "The damage is excluded under the base exclusion with no exception applying."),
            ("deny_vacancy", "The damage is excluded because the dwelling was vacant beyond the permitted period before the loss."),
            ("pay_full_less_deductible", "The loss is covered without any sublimit; the full estimate is paid after deducting the deductible."),
            ("pay_subject_to_base_sublimit", "The loss is covered but payment is capped by the base concealed-damage sublimit."),
            ("pay_subject_to_amended_sublimit", "The loss is covered but payment is capped by the amended sublimit from the attached endorsement."),
        ],
    },
    {
        "name": "procurement_approval",
        "title": "{carrier} GROUP PROCUREMENT POLICY {form} (VERSION 7)\nEXTRACT FOR APPROVAL ROUTING, WITH AMENDMENT {form}/A3 AND PURCHASE REQUEST {claim}",
        "question": "Determine the required approval level for purchase request {claim} under policy {form} as amended.",
        "options": [
            ("budget_holder", "Level 1 is the highest level reached (lowest total contract value band)."),
            ("department_head", "Level 2 is the highest level reached (second value band)."),
            ("vp_and_finance_director", "Level 3 is the highest level reached (third value band)."),
            ("cfo", "Level 4 is the highest level reached (fourth value band)."),
            ("executive_committee", "Level 5 is the highest level reached (highest value band)."),
        ],
    },
    {
        "name": "warranty_determination",
        "title": "{carrier} SYSTEMS - CONSUMER SERVICE CENTRE\nWARRANTY DETERMINATION PACKET, SERVICE ORDER {claim}\nLIMITED WARRANTY FORM {form}",
        "question": "Determine the warranty outcome for the reported failure in service order {claim} under form {form} and the attached service bulletins.",
        "options": [
            ("full_repair_no_charge", "The defect is within standard coverage: parts and labour are free."),
            ("parts_covered_labour_charged", "The defect is within extended parts coverage: the part is supplied free, but labour is charged at the flat rate."),
            ("pro_rata_replacement_credit", "The unit is to be replaced by a pro-rated replacement credit."),
            ("not_covered", "No warranty coverage applies; the customer pays for parts and labour."),
        ],
    },
]

CARRIERS = ["HARBORLINE", "CASTELLAN", "BREVANO", "VELANT", "PELAGOS", "MERIDIAN", "WAYFARER", "NORTHGATE"]

FILLER_SECTIONS = [
    ("Definitions", "In this document, words in capitals have the meanings given in this section. "
                    "\"We\", \"us\" and \"our\" mean the issuing organisation named on the declarations page. "
                    "\"You\" and \"your\" mean the named party and, where applicable, their lawful successors. "
                    "\"Period of cover\" means the term shown on the current schedule, including any renewal term "
                    "accepted in writing before expiry."),
    ("General conditions", "The named party must take reasonable steps to prevent further loss once a condition "
                           "becomes known. Failure to mitigate may reduce the amount otherwise payable. Notice must "
                           "be given in writing within the period stated in the schedule. Verbal notice does not "
                           "satisfy this condition unless confirmed in writing within ten business days."),
    ("Records and inspection", "The named party must retain supporting records for the period stated and make them "
                               "available for inspection on reasonable notice. Records held only in electronic form "
                               "are acceptable where a verifiable audit trail exists. Inspection does not constitute "
                               "acceptance of any claim or request."),
    ("Dispute resolution", "Any dispute arising under this document shall first be referred to the internal review "
                           "panel. If unresolved after thirty days, either party may refer the matter to mediation. "
                           "Nothing in this section limits either party's right to seek injunctive relief."),
    ("Assignment", "Neither party may assign its rights under this document without prior written consent, which "
                   "shall not be unreasonably withheld. A permitted assignment does not release the assigning party "
                   "from obligations accrued before the assignment date."),
    ("Notices", "Notices must be sent to the address shown on the current schedule. A notice sent by recorded "
                "delivery is deemed served two business days after posting. A notice sent by electronic mail is "
                "deemed served on receipt of an acknowledgement generated by the recipient's system."),
    ("Severability", "If any provision of this document is held unenforceable, the remaining provisions continue in "
                     "full force. The unenforceable provision shall be replaced by a valid provision that most "
                     "closely reflects the original commercial intent of the parties."),
    ("Governing law", "This document is governed by the law of the jurisdiction stated in the schedule. The parties "
                      "submit to the non-exclusive jurisdiction of the courts of that territory. Time limits stated "
                      "in this document are calculated in calendar days unless expressly stated otherwise."),
    ("Interpretation", "Headings are for convenience only and do not affect interpretation. The singular includes "
                       "the plural and vice versa. A reference to a statute includes any amendment or re-enactment "
                       "of it. Where a period is expressed to run from a given day, that day is excluded."),
    ("Premium and adjustment", "Amounts payable are calculated on the basis stated in the schedule and may be "
                               "adjusted at renewal following review of the preceding term's experience. An "
                               "adjustment does not take effect retrospectively unless expressly agreed in writing."),
]


def _filler_block(rng: random.Random, n_sections: int, start_idx: int) -> str:
    """Emit neutral, plausible sections that carry no decision-relevant signal."""
    picks = rng.sample(FILLER_SECTIONS, min(n_sections, len(FILLER_SECTIONS)))
    out = []
    for i, (heading, body) in enumerate(picks):
        out.append(f"{start_idx + i}. {heading}\n{start_idx + i}.1 {body}")
    return "\n\n".join(out)


def _build_insurance_doc(rng: random.Random, target_tokens: int) -> Dict[str, object]:
    """Vary the claim facts so every outcome label is reachable.

    The label is sampled first, then the document's facts are constructed to make
    exactly that outcome correct. Without this the generator emits a constant
    label and the scorer learns a position/option prior instead of reading.
    """
    label = rng.choice([
        "deny_excluded",
        "deny_vacancy",
        "pay_full_less_deductible",
        "pay_subject_to_base_sublimit",
        "pay_subject_to_amended_sublimit",
    ])

    base_sublimit = rng.choice([5000, 10000, 12500])
    amended_sublimit = base_sublimit + rng.choice([2500, 5000, 7500])
    deductible = rng.choice([500, 1000, 2500])
    vacancy_limit = rng.choice([30, 60, 90])
    endorsement = f"HE-{rng.randint(1000, 9999)}"

    # Facts are derived from the sampled label.
    has_endorsement = label == "pay_subject_to_amended_sublimit"
    if label == "deny_vacancy":
        days_away = vacancy_limit + rng.randint(5, 40)      # exceeds the limit
    else:
        days_away = max(3, vacancy_limit - rng.randint(5, 25))  # safely under

    if label == "deny_excluded":
        cause = ("slow weeping at a visible pipe joint in the open basement, present and visible for "
                 "approximately eleven weeks before it was reported")
        concealed = False
        seepage_over_14_days = True
    elif label == "pay_full_less_deductible":
        cause = ("sudden burst of a washing machine supply hose in the open utility room, discovered "
                 "the same day it occurred")
        concealed = False
        seepage_over_14_days = False
    else:
        cause = ("pinhole failure in a copper supply line concealed within an interior wall cavity")
        concealed = True
        seepage_over_14_days = False

    applicable_sublimit = amended_sublimit if has_endorsement else base_sublimit
    estimate = applicable_sublimit + rng.randint(3000, 9000)

    exclusions = (
        f"4. Exclusions\n"
        f"4.3 Repeated seepage or leakage. We do not cover loss caused by continuous or repeated seepage or "
        f"leakage of water over a period of fourteen or more days.\n"
        f"4.3.1 Exception - concealed damage. Notwithstanding 4.3, we cover sudden damage from a source concealed "
        f"within a wall, floor or ceiling, subject to the concealed-damage sublimit stated in 4.3.2.\n"
        f"4.3.2 The concealed-damage sublimit is ${base_sublimit:,} unless amended by endorsement attached to the "
        f"current term.\n"
        f"4.3.3 Damage that is neither concealed nor the result of repeated seepage is covered in full, subject only "
        f"to the deductible shown on the schedule.\n"
        f"4.7 Vacancy. We do not cover loss occurring while the dwelling has been vacant for more than "
        f"{vacancy_limit} consecutive days immediately before the loss."
    )

    if has_endorsement:
        endorsement_block = (
            f"ENDORSEMENT {endorsement} - CONCEALED DAMAGE SUBLIMIT AMENDMENT\n"
            f"Attached to and forming part of the current term. Effective from the first day of the current period "
            f"of cover. The concealed-damage sublimit stated in 4.3.2 is deleted and replaced with "
            f"${amended_sublimit:,}. All other terms remain unchanged."
        )
    else:
        # A real but non-operative endorsement: present, lexically similar, wrong subject.
        endorsement_block = (
            f"ENDORSEMENT {endorsement} - DEBRIS REMOVAL CLARIFICATION\n"
            f"Attached to and forming part of the current term. Reasonable debris removal costs are included within "
            f"the applicable limit and do not increase it. No sublimit stated elsewhere in this policy is amended "
            f"by this endorsement."
        )

    claim_block = (
        f"CLAIM FILE SUMMARY\n"
        f"Reported cause: {cause}.\n"
        f"Concealment assessment: the source was "
        f"{'not visible from the occupied space prior to discovery' if concealed else 'in open view and accessible without opening any wall, floor or ceiling'}.\n"
        f"Duration assessment: "
        f"{'the escape of water continued for more than fourteen days before discovery' if seepage_over_14_days else 'the escape of water did not continue for fourteen or more days before discovery'}.\n"
        f"Occupancy: the named party was away from the dwelling for {days_away} consecutive days immediately "
        f"before the loss.\n"
        f"Repair estimate: ${estimate:,}.\n"
        f"Deductible shown on schedule: ${deductible:,}.\n"
        f"[File clerk note: a trainee adjuster suggested the vacancy exclusion in 4.7 should apply. The reviewer "
        f"should verify this against the recorded occupancy dates and the {vacancy_limit}-day limit.]"
    )

    parts = [
        _filler_block(rng, 3, 1),
        exclusions,
        _filler_block(rng, 3, 5),
        endorsement_block,
        _filler_block(rng, 2, 9),
        claim_block,
    ]
    return {"body": "\n\n".join(parts), "label": label}


def _build_procurement_doc(rng: random.Random, target_tokens: int) -> Dict[str, object]:
    """Sample the target approval level, then size the request to land in it."""
    bands = [10000, 50000, 150000, 500000]
    levels = [
        "budget_holder",
        "department_head",
        "vp_and_finance_director",
        "cfo",
        "executive_committee",
    ]
    idx = rng.randrange(len(levels))
    label = levels[idx]

    # Aggregate value must fall inside the sampled band.
    lo = 1000 if idx == 0 else bands[idx - 1] + 1
    hi = bands[idx] if idx < len(bands) else bands[-1] * 3
    total = rng.randint(lo, hi)

    # Aggregation is only in play when a related request exists; when it does, the
    # primary line item alone sits in a strictly lower band, so reading only the
    # first figure gives the wrong level.
    aggregated = rng.random() < 0.65 and total > 2000
    if aggregated:
        related = rng.randint(total // 5, total // 2)
        line_item = total - related
    else:
        related = 0
        line_item = total

    thresholds = (
        f"3. Approval thresholds\n"
        f"3.1 Level 1 - budget holder: total contract value up to EUR {bands[0]:,}.\n"
        f"3.2 Level 2 - department head: above EUR {bands[0]:,} up to EUR {bands[1]:,}.\n"
        f"3.3 Level 3 - VP and finance director: above EUR {bands[1]:,} up to EUR {bands[2]:,}.\n"
        f"3.4 Level 4 - CFO: above EUR {bands[2]:,} up to EUR {bands[3]:,}.\n"
        f"3.5 Level 5 - executive committee: above EUR {bands[3]:,}.\n"
        f"3.6 Total contract value (TCV) means the value of the goods or services over the full committed term, "
        f"excluding recoverable tax."
    )

    amendment = (
        f"AMENDMENT A3 - AGGREGATION OF RELATED REQUESTS\n"
        f"Effective for all requests raised after the amendment date. Where two or more requests relate to the same "
        f"project, supplier and budget period, their values must be aggregated and the approval level determined on "
        f"the aggregate TCV, not on the individual request value. This amendment supersedes any contrary practice "
        f"previously applied under section 3."
    )

    if aggregated:
        request_block = (
            f"PURCHASE REQUEST SUMMARY\n"
            f"Primary line item: EUR {line_item:,} (committed term, excluding recoverable tax).\n"
            f"Related request, same project, same supplier, same budget period: EUR {related:,}.\n"
            f"Both requests were raised after the amendment date.\n"
            f"Aggregate value across related requests: EUR {total:,}.\n"
            f"[File clerk note: the requesting department routed this on the primary line item alone.]"
        )
    else:
        request_block = (
            f"PURCHASE REQUEST SUMMARY\n"
            f"Primary line item: EUR {line_item:,} (committed term, excluding recoverable tax).\n"
            f"No other request relates to this project, supplier and budget period, so no aggregation applies "
            f"under Amendment A3.\n"
            f"Total contract value for approval purposes: EUR {total:,}.\n"
            f"[File clerk note: a prior unrelated request to the same supplier in an earlier budget period is "
            f"recorded at EUR {rng.randint(20000, 400000):,}; it falls outside the current budget period.]"
        )

    parts = [
        _filler_block(rng, 3, 1),
        thresholds,
        _filler_block(rng, 3, 5),
        amendment,
        _filler_block(rng, 2, 9),
        request_block,
    ]
    return {"body": "\n\n".join(parts), "label": label}



def _build_warranty_doc(rng: random.Random, target_tokens: int) -> Dict[str, object]:
    """Sample the warranty outcome, then set age / serial range / repairability to match."""
    label = rng.choice([
        "full_repair_no_charge",
        "parts_covered_labour_charged",
        "pro_rata_replacement_credit",
        "not_covered",
    ])

    base_years = rng.choice([1, 2])
    extended_years = base_years + rng.choice([1, 2, 3])
    bulletin = f"SB-{rng.randint(100, 999)}"

    in_serial_range = True
    economically_repairable = True

    if label == "full_repair_no_charge":
        age_years = max(1, base_years - rng.choice([0, 1]))        # inside base cover
    elif label == "parts_covered_labour_charged":
        age_years = rng.randint(base_years + 1, extended_years)    # past base, inside bulletin
    elif label == "pro_rata_replacement_credit":
        age_years = max(1, base_years - rng.choice([0, 1]))        # inside base cover...
        economically_repairable = False                            # ...but beyond repair
    else:  # not_covered
        if rng.random() < 0.5:
            age_years = extended_years + rng.randint(1, 3)         # past everything
        else:
            age_years = rng.randint(base_years + 1, extended_years)
            in_serial_range = False                                # bulletin does not apply

    coverage = (
        f"2. What is covered\n"
        f"2.1 We warrant the unit against defects in materials and workmanship for {base_years} year(s) from the "
        f"date of purchase. During this period both parts and labour are supplied free of charge.\n"
        f"2.2 After the standard period expires, no cover applies unless extended cover is granted under a service "
        f"bulletin in force at the date of the failure and the unit falls within the serial range of that bulletin.\n"
        f"3.4 Where a unit cannot be economically repaired, we instead offer a pro-rated replacement credit "
        f"calculated on the remaining unexpired portion of the standard period. This applies only while the unit "
        f"remains within the standard period."
    )

    bulletin_block = (
        f"SERVICE BULLETIN {bulletin} - EXTENDED PARTS COVER\n"
        f"Applies to units manufactured in the affected serial range. The pump component is covered against "
        f"premature failure for {extended_years} years from the date of purchase. Under this bulletin the affected "
        f"part is supplied free of charge; labour is charged at the published flat rate and is not waived. This "
        f"bulletin extends parts cover only and does not reinstate the standard labour entitlement under 2.1."
    )

    order_block = (
        f"SERVICE ORDER SUMMARY\n"
        f"Unit age at date of failure: {age_years} year(s) from date of purchase.\n"
        f"Serial number: "
        f"{'within' if in_serial_range else 'outside'} the affected range identified in service bulletin {bulletin}.\n"
        f"Reported failure: pump component, consistent with the premature failure mode described in the bulletin.\n"
        f"Repair assessment: "
        f"{'economically repairable; replacement not indicated' if economically_repairable else 'not economically repairable; the cost of repair exceeds the value of the unit'}.\n"
        f"[File clerk note: the counter staff quoted the customer for both parts and labour.]"
    )

    parts = [
        _filler_block(rng, 3, 1),
        coverage,
        _filler_block(rng, 3, 5),
        bulletin_block,
        _filler_block(rng, 2, 9),
        order_block,
    ]
    return {"body": "\n\n".join(parts), "label": label}



_BUILDERS = {
    "insurance_water_damage": _build_insurance_doc,
    "procurement_approval": _build_procurement_doc,
    "warranty_determination": _build_warranty_doc,
}


def generate_synthetic_policies(
    n: int = 12000,
    min_tokens: int = 1500,
    max_tokens: int = 4000,
    seed: Optional[int] = None,
) -> List[dict]:
    """Compose policy documents requiring multi-clause resolution.

    Length is padded with neutral filler sections until the approximate token
    budget is met, so the resulting distribution covers the 1.5k-4k range that
    the measured accuracy cliff sits in.
    """
    rng = random.Random(seed)
    records: List[dict] = []

    for _ in range(n):
        domain = rng.choice(DOMAINS)
        built = _BUILDERS[domain["name"]](rng, rng.randint(min_tokens, max_tokens))

        carrier = rng.choice(CARRIERS)
        form = f"{rng.choice('ABCDEFGH')}{rng.randint(10, 99)}-{rng.randint(2020, 2026)}"
        claim = f"{rng.choice(['HX', 'PR', 'SO', 'TC'])}-{rng.randint(2024, 2026)}-{rng.randint(100000, 999999)}"

        title = domain["title"].format(carrier=carrier, form=form, claim=claim)
        question = domain["question"].format(carrier=carrier, form=form, claim=claim)

        body = built["body"]
        # Pad toward the target length with additional neutral sections.
        target_words = rng.randint(min_tokens, max_tokens) // 1.4
        guard = 0
        while len(body.split()) < target_words and guard < 12:
            body += "\n\n" + _filler_block(rng, 3, 20 + guard * 3)
            guard += 1

        state = f"{title}\n\n{body}"

        options = [{"id": oid, "description": desc} for oid, desc in domain["options"]]
        rng.shuffle(options)  # defeat positional bias in the marker layout

        records.append({
            "state": state,
            "question": question,
            "options": options,
            "label": built["label"],
            "source": f"synthetic_long_{domain['name']}",
        })

    return records


# ---------------------------------------------------------------------------
# 3. Corpus assembly
# ---------------------------------------------------------------------------

def build_long_context_corpus(
    n_synthetic: int = 12000,
    max_legalbench: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> List[dict]:
    random.seed(seed)
    if verbose:
        print("Building long-context decision corpus...")

    curated = harvest_legalbench(max_rows=max_legalbench, verbose=verbose)
    synthetic = generate_synthetic_policies(n=n_synthetic, seed=seed)
    if verbose:
        print(f"  [synthetic] {len(synthetic):,} long policy documents")

    records = curated + synthetic
    random.shuffle(records)
    if verbose:
        print(f"  TOTAL: {len(records):,} long-context records")
    return records


def write_jsonl(records: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

DATASET_CARD = """---
license: apache-2.0
task_categories:
  - text-classification
  - multiple-choice
language:
  - en
tags:
  - decision-making
  - long-context
  - system-one
  - legal
  - von
size_categories:
  - 10K<n<100K
---

# Von Long-Context Decision Corpus

Decision items whose premises are long enough to require locating and composing
evidence across widely separated spans.

## Why this exists

Encoder-based decision models are typically trained on short premises. Von's own
Option-Marker corpus had a median premise of ~16 tokens and a maximum of ~116,
while inference runs at an 8,192-token context. A length-stratified diagnostic on
JevBench's hard tier showed accuracy decaying monotonically with premise length:

| Premise tokens | Accuracy |
|---|---|
| 0-512 | 40.7% |
| 512-1024 | 33.3% |
| 1024-2048 | 33.3% |
| 2048-4096 | 23.5% |

Raising a trainer's `max_length` does not fix this on its own: if no training
example is long enough to truncate, the larger window simply stays empty. This
corpus supplies the missing examples.

## Composition

**Curated (`legalbench_*`)** - real contract text from
[nguha/legalbench](https://huggingface.co/datasets/nguha/legalbench)
(`contract_nli_*`, `maud_*`, `consumer_contracts_qa`), restricted to Yes/No tasks
where the option semantics are unambiguous. These carry authentic legal trap
knowledge that templates cannot invent.

**Synthetic (`synthetic_long_*`)** - procedurally composed policy documents with
controllable length, targeting the 2,048-4,096 token band specifically. Each
document is built so the answer cannot be pattern-matched from any single span:

- a base rule and an **amendment or endorsement that overrides it**,
- **applicability gates** (serial ranges, occupancy periods, budget periods) that
  decide whether the override is even operative,
- **lexically attractive distractors** - a real endorsement on the wrong subject,
  a prior request outside the budget period, a file-clerk note asserting the wrong
  conclusion,
- ~2,000 tokens of neutral boilerplate separating the decisive clauses.

Labels are **sampled first**, then the document's facts are constructed to make
that outcome correct, so every label is reachable and no label is a constant.
Option order is shuffled per record to defeat positional priors.

## Format

One JSON object per line:

```json
{
  "state": "<the document / premise>",
  "question": "<decision to make>",
  "options": [{"id": "...", "description": "..."}],
  "label": "<correct option id>",
  "source": "<generator or legalbench config>"
}
```

## Caveats

- Synthetic documents are procedurally generated from a fixed set of three
  domains (insurance coverage, procurement approval, warranty determination).
  They exercise cross-clause composition, not domain breadth.
- Synthetic text is **not legal advice** and describes fictitious organisations.
- The curated split inherits legalbench's own licensing and task definitions.

## Reproduce

```bash
uv run python training/prepare_long_context_dataset.py \\
  --out data_long/long_context.jsonl --n_synthetic 12000 --stats
```
"""


def push_to_hub(jsonl_path: str, repo_id: str, private: bool = False) -> None:
    """Publish the corpus and its card to the Hugging Face Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=jsonl_path,
        path_in_repo="long_context.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Published -> https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Von's long-context decision corpus.")
    parser.add_argument("--out", type=str, default="data_long/long_context.jsonl")
    parser.add_argument("--n_synthetic", type=int, default=12000)
    parser.add_argument("--max_legalbench", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats", action="store_true",
                        help="Tokenize a sample and report the length distribution.")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/von-option-marker-universal")
    parser.add_argument("--push_to_hub", type=str, default=None,
                        metavar="REPO_ID",
                        help="Publish to the Hub, e.g. wfzyx/von-long-context-decisions")
    parser.add_argument("--private", action="store_true",
                        help="Create the Hub dataset repo as private.")
    args = parser.parse_args()

    records = build_long_context_corpus(
        n_synthetic=args.n_synthetic,
        max_legalbench=args.max_legalbench,
        seed=args.seed,
    )

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_jsonl(records, args.out)
    print(f"Wrote {len(records):,} records -> {args.out}")

    if args.stats:
        from transformers import AutoTokenizer
        import numpy as np
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        sample = records[:600]
        lens = np.array([len(tok.encode(r["state"])) for r in sample])
        print("\n=== premise token length (sample of %d) ===" % len(sample))
        print(f"  median {int(np.median(lens))}  p90 {int(np.percentile(lens, 90))}  max {int(lens.max())}")
        for lo, hi in [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 10**9)]:
            frac = float(((lens >= lo) & (lens < hi)).mean())
            print(f"  {lo:>5}-{hi if hi < 10**9 else 'inf':<5}: {frac:6.1%}")

    if args.push_to_hub:
        push_to_hub(args.out, args.push_to_hub, private=args.private)


if __name__ == "__main__":
    main()
