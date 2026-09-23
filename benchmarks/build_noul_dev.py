"""Build a zero-shot noul dev set for fitting the zero-shot debias prior (B.1).

Every row here must be clean of jabr's and JevBench's actual test items, and NOT drawn
from any of Von's own hand-written synthetic generators (training/prepare_*.py) --
those generators' output premises were found (this session) to literally overlap
jabr's test-case text across most of its categories, so reusing them here would make
the "held-out" dev set circular with the benchmark we verify against.

Two independent sources:
  1. google/boolq validation split -- external, real, zero-shot yes/no reading
     comprehension; nothing to do with jabr or Von's generators.
  2. A small hand-written set (fresh wording, written for this dev set only, never
     copied from jabr, JevBench, or any existing Von generator) covering generic
     business-decision-style yes/no scenarios, for domain diversity closer to what
     the zero-shot noul path actually sees in practice.

Output: {"state": str, "instructions": str, "gold": "yes"|"no"}
"""

from __future__ import annotations

import argparse
import json
import os
import random

OUT_DEFAULT = "benchmarks/data/noul_zeroshot_dev.jsonl"

# Hand-written, fresh for this dev set. Deliberately NOT overlapping with any string
# in training/prepare_universal_dataset.py, training/prepare_decision_dataset.py,
# jabr's cases/v1.toml or cases/v2.toml, or any JevBench public item.
HANDWRITTEN = [
    ("A customer emailed asking whether the extended warranty on their laptop, "
     "purchased fourteen months ago, still applies after a two-year coverage period.",
     "Does the warranty coverage still apply?", "no"),
    ("The wire transfer arrived from an account the sender has used for every "
     "monthly payment over the past three years, for the usual invoiced amount.",
     "Does this transaction look suspicious?", "no"),
    ("A first-time sender moved the entire account balance to a newly created "
     "recipient in a country the account has never transacted with before.",
     "Does this transaction look suspicious?", "yes"),
    ("The tenant application lists a stable two-year employment history and a "
     "credit score comfortably above the property's stated minimum.",
     "Should this rental application be approved on financial grounds?", "yes"),
    ("The landlord's posting states a preference for tenants without young "
     "children, citing noise concerns from previous renters.",
     "Does this listing raise a fair-housing concern?", "yes"),
    ("The shipment manifest lists bottled water, paperback books, and folded "
     "cotton towels bound for a residential address.",
     "Does this shipment require hazardous-materials handling?", "no"),
    ("The package contains two sealed canisters of compressed aerosol propellant "
     "flagged under the carrier's dangerous-goods schedule.",
     "Does this shipment require hazardous-materials handling?", "yes"),
    ("The recipe substitutes coconut cream for dairy cream and uses only "
     "vegetable-based broth throughout.",
     "Is this recipe suitable for a strict vegan diet?", "yes"),
    ("The recipe calls for a stick of butter and two eggs folded into the batter.",
     "Is this recipe suitable for a strict vegan diet?", "no"),
    ("The login attempt came from the same device fingerprint and city the "
     "account has used for its last forty successful logins.",
     "Does this login attempt look risky?", "no"),
    ("The login attempt used a password-spray pattern against twelve different "
     "accounts within ninety seconds from one IP address.",
     "Does this login attempt look risky?", "yes"),
    ("The email asks the recipient to confirm their password by replying "
     "directly, sent from a domain one character off from the real company name.",
     "Is this email a phishing attempt?", "yes"),
    ("The email is a routine calendar invite from a colleague's verified "
     "internal address confirming next week's stand-up time.",
     "Is this email a phishing attempt?", "no"),
    ("The patient reports a papercut on one finger with no bleeding after a "
     "minute of pressure.",
     "Does this patient need urgent medical attention?", "no"),
    ("The patient reports sudden chest tightness radiating to the left arm "
     "and shortness of breath that started ten minutes ago.",
     "Does this patient need urgent medical attention?", "yes"),
    ("The contract clause states the vendor may terminate for convenience with "
     "ninety days written notice and no penalty to either party.",
     "Does this clause impose an early-termination penalty?", "no"),
    ("The contract clause requires the customer to pay the remaining full "
     "contract value if they cancel before the end of the term.",
     "Does this clause impose an early-termination penalty?", "yes"),
    ("The two support tickets both describe the mobile app crashing on launch "
     "after the latest update, filed four minutes apart by different users.",
     "Are these two tickets describing the same underlying issue?", "yes"),
    ("One ticket describes a billing double-charge and the other describes a "
     "forgotten password, filed by different users.",
     "Are these two tickets describing the same underlying issue?", "no"),
    ("The claim form lists a windshield chip repair with a timestamped photo "
     "from the day of the reported incident and a matching police report number.",
     "Does this insurance claim look legitimate?", "yes"),
    ("The claim form was submitted eight months after the stated incident date "
     "with no supporting documentation and a policy that had already lapsed.",
     "Does this insurance claim look legitimate?", "no"),
    ("The delivery tracking shows the package scanned as delivered to the "
     "correct address at 2pm, matching the recipient's doorbell camera footage.",
     "Does this delivery look like it was misdelivered?", "no"),
    ("The delivery tracking shows the package scanned as delivered in a zip "
     "code three states away from the shipping address on the order.",
     "Does this delivery look like it was misdelivered?", "yes"),
    ("The product review focuses entirely on how the packaging looked festive "
     "for a holiday gift with no comment on the product's function.",
     "Should this review be flagged as off-topic?", "yes"),
    ("The product review describes exactly how the blender performed on ice "
     "and how loud it was during use.",
     "Should this review be flagged as off-topic?", "no"),
]


def load_boolq(n: int, seed: int) -> list:
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split="validation")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        row = ds[i]
        out.append({
            "state": row["passage"],
            "instructions": row["question"].strip().rstrip("?") + "?",
            "gold": "yes" if row["answer"] else "no",
            "source": "boolq_validation",
        })
    return out


def build(n_boolq: int, seed: int, out_path: str) -> None:
    rows = load_boolq(n_boolq, seed)
    for state, instructions, gold in HANDWRITTEN:
        rows.append({"state": state, "instructions": instructions, "gold": gold,
                      "source": "handwritten_dev"})
    random.Random(seed).shuffle(rows)
    yes = sum(1 for r in rows if r["gold"] == "yes")
    print(f"dev set: {len(rows)} rows ({n_boolq} boolq + {len(HANDWRITTEN)} handwritten), "
          f"yes-rate {yes/len(rows):.1%}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_boolq", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--out", default=OUT_DEFAULT)
    args = parser.parse_args()
    build(args.n_boolq, args.seed, args.out)
