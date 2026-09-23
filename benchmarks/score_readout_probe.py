"""Compare score readout methods (argmax-over-levels vs round(expectation)) on JevBench
hard score items, using cached probabilities from a single small targeted run.

jabr's own harness (bench/backends/base.py:level_from_probabilities) already uses argmax
over Von's returned probability dict for its score tasks -- it ignores answer.score
entirely -- so this comparison only matters for Von's own API/JevBench harness path,
where eval_hard_fast.py currently reads round(float(ans.score)).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

HARD_PATH = "/tmp/jevbench/datasets/public/hard.jsonl"


def main() -> None:
    import torch

    torch.set_num_threads(4)
    from von.backends.option_marker_backend import OptionMarkerBackend
    from von.types import Score

    rows = [json.loads(l) for l in open(HARD_PATH)]
    score_rows = [r for r in rows if r["question"].get("type") == "score"]
    print(f"{len(score_rows)} score items in JevBench hard tier")

    backend = OptionMarkerBackend(checkpoint_dir="checkpoints/von-option-marker-universal", device="cpu")
    backend._get_model()

    n_argmax_correct = n_expect_correct = 0
    for r in score_rows:
        q = r["question"]
        crit = q["criteria"]  # ordered list of level descriptions
        instructions = q["instructions"]
        score_q = Score(instructions=instructions, criteria=crit)
        ans = backend.evaluate_score("q", r["state"], score_q)

        probs = ans.probabilities
        argmax_level = max(probs, key=lambda k: probs[k])
        expect_level = str(int(round(float(ans.score))))
        expected = str(r["expected"])

        hit_argmax = argmax_level == expected
        hit_expect = expect_level == expected
        n_argmax_correct += hit_argmax
        n_expect_correct += hit_expect
        marker = "" if hit_argmax == hit_expect else "  <-- DIFFERS"
        print(f"  {r['id']:<28} expected={expected}  argmax={argmax_level}({hit_argmax})  "
              f"expectation={expect_level}({hit_expect})  raw_score={ans.score:.3f}{marker}")

    n = len(score_rows)
    print(f"\nargmax-over-levels:      {n_argmax_correct}/{n} = {n_argmax_correct/n:.3f}")
    print(f"round(expectation):      {n_expect_correct}/{n} = {n_expect_correct/n:.3f}")


if __name__ == "__main__":
    main()
