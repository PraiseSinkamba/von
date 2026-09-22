#!/usr/bin/env python3
"""Launch a short GPU timing probe for length-bucketed training.

Long batches carry ~7,500 padded tokens against a short batch's ~400, and hold
2-3 rows instead of 8. That makes the full-run cost of `--long_ratio` a guess
until it is measured on real hardware. This launcher runs a few hundred steps
and prints the measured long/short cost ratio plus a projected run cost, then
terminates the instance.

Deliberately cheap: a small corpus, no checkpoint upload, hard shutdown
watchdog. Expect ~15-25 minutes of instance time.

Usage:
    uv run python training/launch_timing_probe.py --steps 200 --long-ratio 0.30
"""

import argparse
import json
import time

from launch_universal_training import (  # reuse the proven infra constants
    AMI_ID,
    IAM_PROFILE,
    SUBNETS,
    run_aws,
)

# Single-GPU is enough to time a batch and is a quarter the price of the 4x box.
CANDIDATE_TYPES = [
    ("g5.4xlarge", "1x A10G 24GB, 16 vCPU"),
    ("g5.2xlarge", "1x A10G 24GB, 8 vCPU"),
    ("g4dn.2xlarge", "1x T4 16GB, 8 vCPU"),
    ("g4dn.4xlarge", "1x T4 16GB, 16 vCPU"),
]

USER_DATA_TEMPLATE = """#!/bin/bash
set -x
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

# Hard watchdog: this box must never outlive the probe.
shutdown -c 2>/dev/null || true
shutdown -h +{watchdog} &

echo "=== [VON TIMING PROBE START] ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y awscli curl git

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

mkdir -p /opt/von
aws s3 cp s3://model-weight/von-marker-src.tar.gz /tmp/von-marker-src.tar.gz
tar -xzf /tmp/von-marker-src.tar.gz -C /opt/von
cd /opt/von

/root/.local/bin/uv venv --clear /opt/von/.venv
# PyPI ships CUDA-enabled Linux torch wheels; the old cu121 index now 404s and
# breaks the solve. One install so torch and its dependents resolve together.
/root/.local/bin/uv pip install --python /opt/von/.venv torch torchvision transformers datasets scipy sentencepiece tiktoken accelerate pydantic awscli
export PYTHONPATH="/opt/von/src:$PYTHONPATH"

# Small corpus: per-batch cost depends on batch composition, not corpus size.
echo "=== Building probe corpus ==="
/opt/von/.venv/bin/python -m training.prepare_universal_dataset \\
    --max_train {max_train} --val_samples 500 \\
    --long_context {long_context} --output_dir data_probe

nvidia-smi -L

echo "=== Running {steps}-step timing probe (long_ratio={long_ratio}) ==="
/opt/von/.venv/bin/python -m training.train_option_marker \\
    --train_data data_probe/train.jsonl \\
    --val_data data_probe/val.jsonl \\
    --base_model_id wfzyx/von \\
    --epochs 3 \\
    --batch_size 8 \\
    --grad_accum_steps 2 \\
    --max_position_embeddings 8192 \\
    --long_ratio {long_ratio} \\
    --max_steps {steps} \\
    --output_dir /tmp/probe_out 2>&1 | tail -80

echo "=== [PROBE COMPLETE] ==="
aws s3 cp /var/log/user-data.log s3://model-weight/probe/timing-probe.log || true
shutdown -h now
"""


def launch(steps: int, long_ratio: float, max_train: int, long_context: int, watchdog: int):
    print("=" * 64)
    print("  VON LENGTH-BUCKETING TIMING PROBE")
    print(f"  Steps: {steps}   long_ratio: {long_ratio}")
    print(f"  Probe corpus: {max_train:,} train rows ({long_context:,} synthetic long)")
    print(f"  Watchdog: {watchdog} min")
    print("=" * 64 + "\n")

    user_data = USER_DATA_TEMPLATE.format(
        steps=steps,
        long_ratio=long_ratio,
        max_train=max_train,
        long_context=long_context,
        watchdog=watchdog,
    )
    path = "/tmp/user_data_probe.sh"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(user_data)
    except OSError as exc:
        raise RuntimeError(f"Cannot stage user-data script at {path!r}: {exc}") from exc

    for itype, desc in CANDIDATE_TYPES:
        print(f"Evaluating {itype} [{desc}]...")
        for subnet_id, az in SUBNETS:
            print(f"  -> trying {az} ({subnet_id})...")
            try:
                res = run_aws([
                    "ec2", "run-instances",
                    "--image-id", AMI_ID,
                    "--instance-type", itype,
                    "--subnet-id", subnet_id,
                    "--iam-instance-profile", f"Name={IAM_PROFILE}",
                    "--user-data", f"file://{path}",
                    "--count", "1",
                    "--instance-initiated-shutdown-behavior", "terminate",
                    "--tag-specifications", json.dumps([{
                        "ResourceType": "instance",
                        "Tags": [{"Key": "Name", "Value": "von-timing-probe"}],
                    }]),
                ])
                iid = res["Instances"][0]["InstanceId"]
                print(f"\n-> LAUNCHED {itype} in {az}: {iid}")
                print(f"\nWatch with:\n  uv run python training/launch_timing_probe.py --watch {iid}\n")
                return iid
            except Exception as e:
                err = str(e)
                if any(k in err for k in ("InsufficientInstanceCapacity", "Unsupported", "SpotMaxPriceTooLow")):
                    print("     capacity unavailable.")
                    continue
                print(f"     failed: {err}")
    print("\nAll candidate pools exhausted.")
    return None


def watch(instance_id: str, poll: int = 60):
    """Poll console output until the profile block appears or the box dies."""
    print(f"Watching {instance_id} (Ctrl-C to stop; instance self-terminates)...\n")
    while True:
        try:
            state = run_aws(["ec2", "describe-instances", "--instance-ids", instance_id])
            st = state["Reservations"][0]["Instances"][0]["State"]["Name"]
        except Exception as e:
            print(f"describe failed: {e}")
            st = "unknown"

        try:
            out = run_aws(["ec2", "get-console-output", "--instance-id", instance_id])
            text = out.get("Output", "") or ""
        except Exception:
            text = ""

        if "STEP TIMING PROFILE" in text:
            idx = text.index("STEP TIMING PROFILE")
            print(text[max(0, idx - 200):])
            print("\n-> profile captured.")
            return
        tail = [l for l in text.strip().splitlines() if l.strip()][-3:]
        print(f"[{time.strftime('%H:%M:%S')}] state={st} | " + " | ".join(t[:90] for t in tail))
        if st in ("terminated", "stopped"):
            print("\nInstance gone. Full log (if uploaded): s3://model-weight/probe/timing-probe.log")
            return
        time.sleep(poll)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--long-ratio", type=float, default=0.30)
    p.add_argument("--max-train", type=int, default=30000)
    p.add_argument("--long-context", type=int, default=12000)
    p.add_argument("--watchdog", type=int, default=45)
    p.add_argument("--watch", type=str, default=None, metavar="INSTANCE_ID")
    args = p.parse_args()

    if args.watch:
        watch(args.watch)
    else:
        launch(args.steps, args.long_ratio, args.max_train, args.long_context, args.watchdog)
