"""Read the permutation sweep: does averaging N passes recover accuracy?

Answers three things:

  single      - accuracy in the canonical option order, the shipped behaviour
  ensemble    - accuracy after averaging probabilities across permutations
  best/worst  - accuracy if every item got its luckiest / unluckiest order

The spread between best and worst is the honest size of the order effect. If it
is large, Von is reading option position rather than the premise, and the
ensemble should claw back real accuracy. If single ~ ensemble ~ best ~ worst,
order is irrelevant and the errors are genuine reasoning failures that no amount
of extra passes will fix.

`flip rate` is the share of items whose predicted answer changes under some
permutation. An answer that moves when only the option order moves was never
grounded in the evidence.

Usage:
    uv run python benchmarks/analyze_permutations.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List


def pick(probs: Dict[str, float]) -> str:
    return max(probs, key=lambda k: probs[k])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="benchmarks/data/permutation_hard.json")
    args = parser.parse_args()

    try:
        with open(args.data, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {args.data}: {exc}") from exc

    n = len(records)
    single = ens = best = worst = flips = 0
    by_family: Dict[str, List[bool]] = defaultdict(list)
    ens_by_family: Dict[str, List[bool]] = defaultdict(list)

    for rec in records:
        expected = rec["expected"].strip().lower()
        orders = rec["per_order"]

        ok_single = pick(orders[0]).strip().lower() == expected
        single += ok_single

        avg = {lab: sum(o.get(lab, 0.0) for o in orders) / len(orders) for lab in rec["labels"]}
        ok_ens = pick(avg).strip().lower() == expected
        ens += ok_ens

        hits = [pick(o).strip().lower() == expected for o in orders]
        best += any(hits)
        worst += all(hits)
        if len(set(pick(o) for o in orders)) > 1:
            flips += 1

        fam = rec.get("family") or "?"
        by_family[fam].append(ok_single)
        ens_by_family[fam].append(ok_ens)

    print(f"hard tier, {n} items x {len(records[0]['per_order'])} option orders\n")
    print(f"  single pass (shipped)     {single/n:>6.1%}")
    print(f"  ensemble over orders      {ens/n:>6.1%}   ({ens-single:+d} items)")
    print(f"  best possible order       {best/n:>6.1%}   <- ceiling if order were chosen perfectly")
    print(f"  correct in EVERY order    {worst/n:>6.1%}   <- genuinely robust answers")
    print(f"  answer flips on reorder   {flips/n:>6.1%}   <- ungrounded in the premise")

    print(f"\n{'family':<18} {'n':>3} {'single':>7} {'ensemble':>9} {'delta':>7}")
    for fam in sorted(by_family, key=lambda f: -len(by_family[f])):
        s = sum(by_family[fam]) / len(by_family[fam])
        e = sum(ens_by_family[fam]) / len(ens_by_family[fam])
        print(f"{fam:<18} {len(by_family[fam]):>3} {s:>6.1%} {e:>8.1%} {e-s:>+7.1%}")


if __name__ == "__main__":
    main()
