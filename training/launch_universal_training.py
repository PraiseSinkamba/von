#!/usr/bin/env python3
"""Launch Phase 4 Universal Decision Corpus Training on AWS Spot GPUs.

Targets 4x NVIDIA T4 (g4dn.12xlarge) running 200,000 samples across all 49
operational decision domains.
"""

import json
import os
import shutil
import subprocess
import time

VENV_AWS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "bin", "aws"))
AWS_CLI = VENV_AWS if os.path.exists(VENV_AWS) else (shutil.which("aws") or "/mnt/c/Program Files/Amazon/AWSCLIV2/aws.exe")
REGION = "us-west-2"
SUBNETS = [
    ("subnet-083ba040", "us-west-2a"),
    ("subnet-070e7461", "us-west-2b"),
    ("subnet-b75369ec", "us-west-2c"),
    ("subnet-a663228e", "us-west-2d"),
]
AMI_ID = "ami-0e24e0019a12c5b13"

CANDIDATE_TYPES = [
    # A10G first: ~2x the throughput of a T4 for this workload and 24GB per GPU,
    # which makes it both faster and cheaper per epoch despite the higher rate.
    ("g5.12xlarge", "4x NVIDIA A10G 96GB, 48 vCPU (On-Demand ~$5.67/hr)"),
    ("g4dn.12xlarge", "4x NVIDIA T4 64GB, 48 vCPU (On-Demand ~$3.91/hr)"),
    ("g5.4xlarge", "1x NVIDIA A10G 24GB, 16 vCPU (Spot ~$0.69/hr)"),
    ("g5.2xlarge", "1x NVIDIA A10G 24GB, 8 vCPU (Spot ~$0.54/hr)"),
]
IAM_PROFILE = "AmazonSSMRoleForInstancesQuickSetup"
S3_TARGET = "s3://model-weight/von-option-marker-universal"

USER_DATA_TEMPLATE = """#!/bin/bash
set -e
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

# Any failure must stop billing. Without this, `set -e` exits before the
# shutdown line below and the instance idles until the watchdog fires.
cleanup() {{
  rc=$?
  echo "=== [EXIT rc=$rc] uploading log and shutting down ==="
  aws s3 cp /var/log/user-data.log {s3_target}/run.log || true
  shutdown -h now
}}
trap cleanup EXIT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cancel any lingering shutdown timer and set a rock-solid 240-minute watchdog
shutdown -c 2>/dev/null || true
shutdown -h +240 &

echo "=== [VON UNIVERSAL DECISION TRAINING START] ==="
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

# Build the Universal Decision Corpus, including the long-context core
echo "=== Building {max_train}-sample Universal Decision Corpus (long_context={long_context}) ==="
/opt/von/.venv/bin/python -m training.prepare_universal_dataset \\
    --max_train {max_train} --val_samples 5000 \\
    --long_context {long_context} --output_dir data_universal

# Detect GPUs and train with DDP
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs. Starting PyTorch DDP training with 8,192 Context Window..."

/opt/von/.venv/bin/torchrun --nproc_per_node=$NUM_GPUS training/train_option_marker.py \\
    --train_data data_universal/train.jsonl \\
    --val_data data_universal/val.jsonl \\
    --base_model_id wfzyx/von \\
    --epochs {epochs} \\
    --batch_size 8 \\
    --grad_accum_steps 2 \\
    --max_position_embeddings 8192 \\
    --long_ratio {long_ratio} \\
    --s3_target {s3_target} \\
    --output_dir checkpoints/von-long-context

aws s3 cp /var/log/user-data.log {s3_target}/run.log || true

echo "=== [UNIVERSAL TRAINING COMPLETE - TERMINATING] ==="
shutdown -h now
"""


def run_aws(cmd: list) -> dict:
    full_cmd = [AWS_CLI] + cmd + ["--region", REGION, "--output", "json"]
    res = subprocess.run(full_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"AWS CLI error: {res.stderr.strip()}")
    if not res.stdout.strip():
        return {}
    return json.loads(res.stdout)


def launch(
    on_demand: bool = False,
    epochs: int = 3,
    s3_target: str = S3_TARGET,
    max_train: int = 290000,
    long_context: int = 40000,
    long_ratio: float = 0.30,
):
    market_str = "On-Demand (Guaranteed)" if on_demand else "Spot"
    print("================================================================")
    print(f"  VON TRAINING LAUNCHER [{market_str}]")
    print(f"  Corpus:           {max_train:,} samples (+{long_context:,} long-context)")
    print(f"  Epochs:           {epochs}")
    print(f"  Long batch ratio: {long_ratio:.0%}")
    print("  Cluster Target:   4x GPU (g4dn.12xlarge / g5.12xlarge)")
    print("  Region:           us-west-2")
    print(f"  Target S3 Prefix: {s3_target}")
    if s3_target == S3_TARGET:
        print("  !! WARNING: writing to the SHIPPED weights prefix.")
    print("================================================================\n")

    user_data_path = "/tmp/user_data_universal.sh"
    with open(user_data_path, "w") as f:
        f.write(USER_DATA_TEMPLATE.format(
            epochs=epochs,
            s3_target=s3_target,
            max_train=max_train,
            long_context=long_context,
            long_ratio=long_ratio,
        ))

    instance_id = None
    selected_type = None

    for itype, desc in CANDIDATE_TYPES:
        print(f"\nEvaluating instance type: {itype} [{desc}]...")
        for subnet_id, az in SUBNETS:
            print(f"  -> Trying {itype} in {az} ({subnet_id})...")
            try:
                run_args = [
                    "ec2", "run-instances",
                    "--image-id", AMI_ID,
                    "--instance-type", itype,
                    "--subnet-id", subnet_id,
                    "--iam-instance-profile", f"Name={IAM_PROFILE}",
                    "--user-data", f"file://{user_data_path}",
                    "--count", "1",
                    "--tag-specifications", json.dumps([{
                        "ResourceType": "instance",
                        "Tags": [{"Key": "Name", "Value": f"von-universal-phase4-{'ondemand' if on_demand else 'spot'}"}]
                    }]),
                ]
                if not on_demand:
                    run_args.extend(["--instance-market-options", json.dumps({"MarketType": "spot"})])

                res = run_aws(run_args)
                instance_id = res["Instances"][0]["InstanceId"]
                selected_type = itype
                print(f"\n-> SUCCESS! Launched {itype} {market_str} instance in {az}: {instance_id}")
                break
            except Exception as e:
                err = str(e)
                if "InsufficientInstanceCapacity" in err or "Unsupported" in err or "SpotMaxPriceTooLow" in err:
                    print(f"     Capacity unavailable in {az}.")
                    continue
                print(f"     Failed: {err}")
        if instance_id:
            break

    if not instance_id:
        print(f"\nAll {market_str} candidate pools exhausted.")
        return

    print("\nWaiting for instance to enter 'running' state...")
    while True:
        desc = run_aws(["ec2", "describe-instances", "--instance-ids", instance_id])
        state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]
        print(f"  -> Instance {instance_id} status: {state}")
        if state == "running":
            break
        time.sleep(10)

    print(f"\nUniversal Phase 4 {market_str} training instance is RUNNING!")
    print(f"Artifacts will automatically upload to {s3_target} upon completion.")
    return instance_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch Von training on AWS GPUs.")
    parser.add_argument("--on-demand", action="store_true",
                        help="Use guaranteed On-Demand capacity instead of Spot.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=290000)
    parser.add_argument("--long-context", type=int, default=40000)
    parser.add_argument("--long-ratio", type=float, default=0.30)
    parser.add_argument("--s3-target", type=str, default=S3_TARGET,
                        help="S3 prefix for checkpoints. Defaults to the SHIPPED weights "
                             "prefix, so point experiments somewhere else.")
    args = parser.parse_args()

    launch(
        on_demand=args.on_demand,
        epochs=args.epochs,
        s3_target=args.s3_target,
        max_train=args.max_train,
        long_context=args.long_context,
        long_ratio=args.long_ratio,
    )
