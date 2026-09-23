"""Cache raw zero-shot noul logits + fit a replacement for the 0.7*bias debias correction.

evaluate_noul's zero-shot path today does:
    bias = null_logits[0] - null_logits[1]
    logits = stack([logits[0] - 0.7*bias, logits[1]])

Measured (this session, jabr v2 noul tasks): the context-free bias is positive on
nearly every task (+6..+7.5), so subtracting 0.7*bias shoves everything toward "no" --
ad_policy_violation predicted "yes" on 0/16 cases. This script fits a linear
replacement `(l_yes - l_no) - (a*bias + b)` on a held-out dev set (never jabr,
never JevBench -- see build_noul_dev.py) and reports before/after accuracy and ECE.

Usage:
    uv run python benchmarks/noul_prior_fit.py --cache   # run the model once, cache raw logits
    uv run python benchmarks/noul_prior_fit.py --fit     # fit a,b from the cache, report metrics
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DEV_PATH = "benchmarks/data/noul_zeroshot_dev.jsonl"
CACHE_PATH = "/tmp/noul_dev_cache.json"
CKPT = "checkpoints/von-option-marker-universal"


def cache_logits(dev_path: str, ckpt: str, out_path: str) -> None:
    import torch

    torch.set_num_threads(4)
    from von.backends.option_marker_backend import OptionMarkerBackend

    backend = OptionMarkerBackend(checkpoint_dir=ckpt, device="cpu")
    model = backend._get_model()
    tok = model.tokenizer

    def raw_pass(state_text: str, instructions: str, descriptions):
        packed = model.pack_sequence(state_text, instructions, descriptions)
        inputs = tok(packed, return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        pos_list = (input_ids == model.mask_token_id).nonzero(as_tuple=True)[0].tolist()
        with torch.no_grad():
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                mask_positions=[pos_list],
                independent_options=backend._independent_options,
            )[0]
        return logits.cpu().tolist()

    rows = [json.loads(l) for l in open(dev_path)]
    out = []
    descriptions = ["Yes, condition holds true.", "No, condition is false."]
    for i, r in enumerate(rows):
        l_yes, l_no = raw_pass(r["state"], r["instructions"], descriptions)
        null_yes, null_no = raw_pass("", r["instructions"], descriptions)
        out.append({
            "gold": r["gold"], "source": r.get("source", "?"),
            "l_yes": l_yes, "l_no": l_no,
            "null_yes": null_yes, "null_no": null_no,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"cached {len(out)} rows -> {out_path}")


def sigmoid(x: float) -> float:
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def metrics(cached, a: float, b: float, coef_today: float = 0.7):
    """Returns (acc_today, acc_fit, ece_today, ece_fit, yes_rate_today, yes_rate_fit)."""
    def eval_with(score_fn):
        correct = 0
        yes_n = 0
        bins = [[] for _ in range(10)]  # (confidence, correct) per bin
        for r in cached:
            bias = r["null_yes"] - r["null_no"]
            raw = r["l_yes"] - r["l_no"]
            score = score_fn(raw, bias)
            p_yes = sigmoid(score)
            pred = "yes" if p_yes >= 0.5 else "no"
            correct += int(pred == r["gold"])
            yes_n += int(pred == "yes")
            conf = p_yes if pred == "yes" else (1 - p_yes)
            b_idx = min(int(conf * 10), 9)
            bins[b_idx].append((conf, int(pred == r["gold"])))
        n = len(cached)
        ece = 0.0
        for bucket in bins:
            if not bucket:
                continue
            avg_conf = sum(c for c, _ in bucket) / len(bucket)
            avg_acc = sum(ok for _, ok in bucket) / len(bucket)
            ece += (len(bucket) / n) * abs(avg_conf - avg_acc)
        return correct / n, ece, yes_n / n

    acc_today, ece_today, yes_today = eval_with(lambda raw, bias: raw - coef_today * bias)
    acc_fit, ece_fit, yes_fit = eval_with(lambda raw, bias: raw - (a * bias + b))
    return acc_today, acc_fit, ece_today, ece_fit, yes_today, yes_fit


def fit(cached):
    """Grid search a,b minimising log-loss (fine enough for a 175-row dev set)."""
    best = (None, None, float("inf"))
    for a10 in range(-20, 21):
        a = a10 / 10.0
        for b10 in range(-100, 101):
            b = b10 / 10.0
            loss = 0.0
            for r in cached:
                bias = r["null_yes"] - r["null_no"]
                raw = r["l_yes"] - r["l_no"]
                score = raw - (a * bias + b)
                p = sigmoid(score)
                p = min(max(p, 1e-6), 1 - 1e-6)
                y = 1.0 if r["gold"] == "yes" else 0.0
                loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
            if loss < best[2]:
                best = (a, b, loss)
    return best[0], best[1], best[2] / len(cached)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--dev", default=DEV_PATH)
    parser.add_argument("--cache_path", default=CACHE_PATH)
    parser.add_argument("--ckpt", default=CKPT)
    args = parser.parse_args()

    if args.cache:
        cache_logits(args.dev, args.ckpt, args.cache_path)

    if args.fit:
        cached = json.load(open(args.cache_path))
        yes_gold = sum(1 for r in cached if r["gold"] == "yes") / len(cached)
        biases = [r["null_yes"] - r["null_no"] for r in cached]
        print(f"n={len(cached)}  gold yes-rate={yes_gold:.3f}  "
              f"mean null bias={sum(biases)/len(biases):.3f}")

        a, b, mean_logloss = fit(cached)
        acc_today, acc_fit, ece_today, ece_fit, yes_today, yes_fit = metrics(cached, a, b)
        print(f"fitted a={a:.2f} b={b:.2f}  mean logloss={mean_logloss:.4f}")
        print(f"today  (0.7*bias, no offset): acc={acc_today:.3f}  ece={ece_today:.3f}  yes-rate={yes_today:.3f}")
        print(f"fitted (a*bias + b):          acc={acc_fit:.3f}  ece={ece_fit:.3f}  yes-rate={yes_fit:.3f}")
        print(f"gold yes-rate: {yes_gold:.3f}")

        out = {"a": round(a, 4), "b": round(b, 4)}
        with open("/tmp/noul_prior_fit_result.json", "w") as f:
            json.dump(out, f)
        print(f"wrote /tmp/noul_prior_fit_result.json: {out}")


if __name__ == "__main__":
    main()
