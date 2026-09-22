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
    ("g4dn.12xlarge", "4x NVIDIA T4 64GB, 48 vCPU (Spot ~$1.53/hr)"),
    ("g5.12xlarge", "4x NVIDIA A10G 96GB, 48 vCPU (Spot ~$3.67/hr)"),
    ("g5.4xlarge", "1x NVIDIA A10G 24GB, 16 vCPU (Spot ~$0.69/hr)"),
    ("g5.2xlarge", "1x NVIDIA A10G 24GB, 8 vCPU (Spot ~$0.54/hr)"),
]
IAM_PROFILE = "AmazonSSMRoleForInstancesQuickSetup"
S3_TARGET = "s3://model-weight/von-option-marker-universal"

USER_DATA_SCRIPT = """#!/bin/bash
set -e
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

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
/root/.local/bin/uv pip install --python /opt/von/.venv torch torchvision --index-url https://download.pytorch.org/whl/cu121
/root/.local/bin/uv pip install --python /opt/von/.venv transformers datasets scipy sentencepiece tiktoken accelerate pydantic awscli
export PYTHONPATH="/opt/von/src:$PYTHONPATH"

# Build Phase 4 Universal Decision Corpus (290k samples)
echo "=== Building 290,000-sample Universal Decision Corpus ==="
/opt/von/.venv/bin/python -m training.prepare_universal_dataset --max_train 290000 --val_samples 5000 --output_dir data_universal

# Detect GPUs and train with DDP
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs. Starting PyTorch DDP training with 8,192 Context Window..."

/opt/von/.venv/bin/torchrun --nproc_per_node=$NUM_GPUS training/train_option_marker.py \\
    --train_data data_universal/train.jsonl \\
    --val_data data_universal/val.jsonl \\
    --base_model_id wfzyx/von-1.0 \\
    --epochs 3 \\
    --batch_size 8 \\
    --grad_accum_steps 2 \\
    --max_position_embeddings 8192 \\
    --s3_target s3://model-weight/von-option-marker-universal \\
    --output_dir checkpoints/von-option-marker-universal

aws s3 cp /var/log/user-data.log s3://model-weight/von-option-marker-universal/run.log || true

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


def launch(on_demand: bool = False):
    market_str = "On-Demand (Guaranteed)" if on_demand else "Spot"
    print("================================================================")
    print(f"  VON UNIVERSAL (PHASE 4) TRAINING LAUNCHER [{market_str}]")
    print("  Corpus:           290,000 samples across 49 domains")
    print("  Cluster Target:   4x GPU (g4dn.12xlarge / g5.12xlarge)")
    print("  Region:           us-west-2")
    print("  Target S3 Prefix: s3://model-weight/von-option-marker-universal")
    print("================================================================\n")

    user_data_path = "/tmp/user_data_universal.sh"
    with open(user_data_path, "w") as f:
        f.write(USER_DATA_SCRIPT)

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
    print(f"Artifacts will automatically upload to {S3_TARGET} upon completion.")
    return instance_id


if __name__ == "__main__":
    import sys
    on_demand_flag = "--on-demand" in sys.argv
    launch(on_demand=on_demand_flag)


if __name__ == "__main__":
    launch()
