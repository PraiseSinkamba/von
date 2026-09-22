#!/usr/bin/env python3
"""Background monitor for Von Universal Phase 4 training.

Polls AWS EC2 console output and S3 checkpoints every 5 minutes.
Logs timestamped progress to /tmp/von_training_monitor.log.
Exits automatically when training completes or terminates.
"""

import datetime
import os
import subprocess
import sys
import time

INSTANCE_ID = "i-022e1791bd034a66b"
S3_CHECKPOINT = "s3://model-weight/von-option-marker-universal/"
LOG_FILE = "/tmp/von_training_monitor.log"

AWS_CLI = "/home/wfzyx/Code/personal/von/.venv/bin/aws"


def log(msg: str):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{now}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def check_status():
    log(f"--- Checking status for {INSTANCE_ID} ---")
    
    # 1. EC2 state
    res = subprocess.run([
        AWS_CLI, "ec2", "describe-instances",
        "--instance-ids", INSTANCE_ID,
        "--query", "Reservations[0].Instances[0].State.Name",
        "--output", "text"
    ], capture_output=True, text=True)
    state = res.stdout.strip()
    log(f"EC2 Instance State: {state}")
    
    if state in ["shutting-down", "terminated", "stopped"]:
        log("Instance has stopped or terminated. Ending monitor.")
        return False

    # 2. Console Output / Training progress
    res = subprocess.run([
        AWS_CLI, "ec2", "get-console-output",
        "--instance-id", INSTANCE_ID,
        "--query", "Output",
        "--output", "text"
    ], capture_output=True, text=True)
    output = res.stdout.strip()
    if output:
        progress_lines = [
            l for l in output.split("\n")
            if any(k in l for k in ["Epoch [", "Validation:", "Syncing", "Complete"])
        ]
        if progress_lines:
            log(f"Latest Progress: {progress_lines[-1]}")
            if len(progress_lines) > 1:
                log(f"Prior Progress:  {progress_lines[-2]}")

    # 3. Check S3 timestamp on marker_calibration.json
    res = subprocess.run([
        AWS_CLI, "s3", "ls", f"{S3_CHECKPOINT}marker_calibration.json"
    ], capture_output=True, text=True)
    if res.stdout.strip():
        log(f"S3 Checkpoint File: {res.stdout.strip()}")

    return True


def main():
    log("Starting background training monitor (interval: 300s / 5m)...")
    while True:
        try:
            active = check_status()
            if not active:
                break
        except Exception as e:
            log(f"Monitor error: {e}")
        time.sleep(300)


if __name__ == "__main__":
    main()
