"""Trains Von's Option-Marker Joint Attention Decision Head with RLCD calibration.

Enables single-pass non-autoregressive decision evaluation:
- Pack state and all options into one sequence marked by [MASK] tokens
- Evaluates joint relative competition across all candidate choices in 1 forward pass
- Calibrated with composite Cross-Entropy + Brier score loss
"""

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from von.models.option_marker import OptionMarkerModel


class OptionMarkerDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def estimate_packed_tokens(item: dict) -> int:
    """Cheap character-based estimate of an item's packed sequence length.

    Tokenizing a 290k-row corpus just to bucket it costs more than the training
    step it feeds, so approximate at ~3.6 chars/token. Only the relative ordering
    and the long/short split matter here, not exactness.
    """
    n = len(item.get("state", "")) + len(item.get("question", ""))
    for opt in item.get("options", []):
        n += len(opt.get("description", "")) + 8  # +8 for the mask/sep scaffolding
    return max(8, int(n / 3.6))


class LengthBucketedBatchSampler(Sampler):
    """Batches by length so long-context examples actually reach the model.

    Two problems are solved together:

    1. *Exposure.* Long examples are a small minority of the corpus, so uniform
       shuffling means the model almost never sees a full-length window. Long
       examples are oversampled until they account for `long_ratio` of batches.

    2. *Memory.* The collator pads to the longest item in the batch, so a single
       3k-token document in a batch of 8 inflates that batch to ~24k tokens and
       OOMs a 16GB card. Batches are therefore built against a token budget:
       long batches automatically get fewer rows.

    DDP safety: every rank derives the identical global batch list from
    (seed, epoch), then takes a strided shard truncated to a common length. Ranks
    that disagree on batch count deadlock at the gradient all-reduce, so the
    truncation is load-bearing, not tidiness.
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        max_tokens: int = 8192,
        long_threshold: int = 2048,
        long_ratio: float = 0.30,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.long_threshold = long_threshold
        self.long_ratio = long_ratio
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.long_idx = [i for i, L in enumerate(lengths) if L >= long_threshold]
        self.short_idx = [i for i, L in enumerate(lengths) if L < long_threshold]
        self._cached_len = len(self._build_batches())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _pack(self, indices: List[int]) -> List[List[int]]:
        """Group pre-sorted indices into batches respecting the token budget."""
        batches: List[List[int]] = []
        cur: List[int] = []
        cur_max = 0
        for i in indices:
            cand_max = max(cur_max, self.lengths[i])
            # Padded cost is (rows * longest row), which is what actually allocates.
            if cur and ((len(cur) + 1) * cand_max > self.max_tokens or len(cur) >= self.batch_size):
                batches.append(cur)
                cur, cur_max = [i], self.lengths[i]
            else:
                cur.append(i)
                cur_max = cand_max
        if cur and not self.drop_last:
            batches.append(cur)
        elif cur and len(cur) == self.batch_size:
            batches.append(cur)
        return batches

    def _build_batches(self) -> List[List[int]]:
        rng = random.Random(self.seed + self.epoch)

        short = list(self.short_idx)
        rng.shuffle(short)
        short_batches = self._pack(short)

        long_batches: List[List[int]] = []
        if self.long_idx:
            # Target count so long batches are `long_ratio` of the final mix.
            n_short = len(short_batches)
            target_long = int(round(n_short * self.long_ratio / max(1e-6, 1 - self.long_ratio)))

            pool: List[int] = []
            while True:
                chunk = list(self.long_idx)
                rng.shuffle(chunk)
                pool.extend(chunk)
                # Sort within the pool so similar lengths batch together (less padding).
                probe = sorted(pool, key=lambda i: self.lengths[i])
                if len(self._pack(probe)) >= target_long or not target_long:
                    pool = probe
                    break
            long_batches = self._pack(pool)[:target_long] if target_long else []

        if not long_batches:
            batches = short_batches
            rng.shuffle(batches)
        else:
            # Interleave evenly rather than shuffling, so long batches are spread
            # across the epoch instead of clumping into a late memory spike.
            batches = list(short_batches)
            stride = max(1, len(batches) // max(1, len(long_batches)))
            for k, lb in enumerate(long_batches):
                pos = min(len(batches), k * (stride + 1))
                batches.insert(pos, lb)

        # Equal batch count per rank: unequal counts deadlock DDP all-reduce.
        per_rank = len(batches) // self.num_replicas
        if per_rank == 0:
            return batches[self.rank:self.rank + 1]
        return batches[self.rank:per_rank * self.num_replicas:self.num_replicas]

    def __iter__(self):
        return iter(self._build_batches())

    def __len__(self) -> int:
        return self._cached_len


def collate_marker_fn(batch: List[dict], tokenizer, max_length: int = 8192):
    packed_texts = []
    labels = []
    mask = tokenizer.mask_token
    sep = tokenizer.sep_token

    for item in batch:
        state = item["state"].strip()
        q = item["question"].strip()
        opts = item["options"]
        target = item["label"]

        opt_ids = [opt["id"] for opt in opts]
        target_idx = opt_ids.index(target) if target in opt_ids else 0
        labels.append(target_idx)

        prefix = f"{q} {state}".strip() if q else state
        opts_packed = " ".join(f"{mask} {opt['description'].strip()}" for opt in opts)
        packed_texts.append(f"{prefix} {sep} {opts_packed}")

    encodings = tokenizer(
        packed_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    batch_mask_positions = []
    mask_id = tokenizer.mask_token_id
    for b in range(len(batch)):
        pos = (encodings["input_ids"][b] == mask_id).nonzero(as_tuple=True)[0].tolist()
        batch_mask_positions.append(pos)

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
        "mask_positions": batch_mask_positions,
    }


def compute_marker_rlcd_loss(
    batch_logits: List[torch.Tensor],
    labels: torch.Tensor,
    brier_weight: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    ce_losses = []
    brier_losses = []
    correct = 0
    total = len(labels)

    for i, logits in enumerate(batch_logits):
        target_idx = labels[i].item()
        probs = torch.softmax(logits, dim=-1)

        safe_target = min(target_idx, probs.size(0) - 1)
        ce = -torch.log(probs[safe_target] + 1e-8)
        ce_losses.append(ce)

        one_hot = torch.zeros_like(probs)
        one_hot[safe_target] = 1.0
        brier = torch.sum((probs - one_hot) ** 2)
        brier_losses.append(brier)

        if torch.argmax(probs).item() == safe_target:
            correct += 1

    mean_ce = torch.stack(ce_losses).mean()
    mean_brier = torch.stack(brier_losses).mean()
    total_loss = mean_ce + brier_weight * mean_brier
    accuracy = correct / max(total, 1)

    return total_loss, mean_ce, accuracy


def train(
    train_path: str,
    val_path: str,
    base_model_id: str,
    output_dir: str,
    s3_target: Optional[str] = None,
    epochs: int = 1,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    lr: float = 3e-5,
    brier_weight: float = 0.5,
    max_position_embeddings: int = 8192,
    max_length: int = 8192,
    length_bucketing: bool = True,
    max_tokens_per_batch: int = 8192,
    long_threshold: int = 2048,
    long_ratio: float = 0.30,
):
    is_ddp = "RANK" in os.environ
    if is_ddp:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_main = (rank == 0)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
        rank = 0
        world_size = 1

    if is_main:
        print(f"Device: {device} (World Size: {world_size}, DDP: {is_ddp})")
        print(f"Loading OptionMarkerModel with base: {base_model_id}...")

    model = OptionMarkerModel(
        base_model_id=base_model_id,
        max_position_embeddings=max_position_embeddings,
    ).to(device)
    tokenizer = model.tokenizer

    train_ds = OptionMarkerDataset(train_path)
    val_ds = OptionMarkerDataset(val_path)

    train_sampler = None
    batch_sampler = None
    if length_bucketing:
        # Bucket by estimated length so long documents are both seen often enough
        # and batched small enough to fit. Falls back to plain shuffling if the
        # corpus turns out to have no long examples at all.
        lengths = [estimate_packed_tokens(r) for r in train_ds.rows]
        n_long = sum(1 for L in lengths if L >= long_threshold)
        if n_long == 0:
            if is_main:
                print(f"  !! No examples >= {long_threshold} tokens; length bucketing disabled.")
            length_bucketing = False
        else:
            batch_sampler = LengthBucketedBatchSampler(
                lengths=lengths,
                batch_size=batch_size,
                max_tokens=max_tokens_per_batch,
                long_threshold=long_threshold,
                long_ratio=long_ratio,
                num_replicas=world_size,
                rank=rank,
                seed=42,
            )
            if is_main:
                print(f"  -> Length bucketing: {n_long:,}/{len(lengths):,} rows >= {long_threshold} tok "
                      f"({n_long / len(lengths):.1%}); target {long_ratio:.0%} of batches, "
                      f"budget {max_tokens_per_batch:,} tok/batch")

    if batch_sampler is not None:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            collate_fn=lambda b: collate_marker_fn(b, tokenizer, max_length=max_length),
        )
    else:
        train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=lambda b: collate_marker_fn(b, tokenizer, max_length=max_length),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_marker_fn(b, tokenizer, max_length=max_length),
    )

    total_steps = math.ceil(len(train_loader) / grad_accum_steps) * epochs
    if is_main:
        print(f"\nStarting Option-Marker training:")
        print(f"  -> Train Samples:   {len(train_ds):,}")
        print(f"  -> Val Samples:     {len(val_ds):,}")
        print(f"  -> Batch Size:      {batch_size}")
        print(f"  -> Grad Accum:      {grad_accum_steps} (Effective: {batch_size * grad_accum_steps * world_size})")
        print(f"  -> Total Steps:     {total_steps:,}\n")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": lr * 0.5},
            {"params": model.scorer.parameters(), "lr": lr * 2.5},
        ],
        weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.08),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        elif is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            mask_positions = batch["mask_positions"]

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                batch_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    mask_positions=mask_positions,
                )
                loss, ce_loss, acc = compute_marker_rlcd_loss(
                    batch_logits, labels, brier_weight=brier_weight
                )
                accum_loss = loss / grad_accum_steps

            scaler.scale(accum_loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item()

            if is_main and ((step + 1) % 100 == 0 or (step + 1) == len(train_loader)):
                elapsed = time.time() - t0
                print(
                    f"Epoch [{epoch}/{epochs}] Step [{step+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}) Acc: {acc*100:.1f}% "
                    f"Elapsed: {elapsed:.1f}s"
                )

        # Validation
        if is_main:
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_scores_list = []
            val_labels_list = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    mask_positions = batch["mask_positions"]

                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        batch_logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            mask_positions=mask_positions,
                        )
                        loss, _, acc = compute_marker_rlcd_loss(
                            batch_logits, labels, brier_weight=brier_weight
                        )

                    val_loss += loss.item()
                    val_acc += acc

                    for i, logits in enumerate(batch_logits):
                        val_scores_list.append(logits.cpu())
                        val_labels_list.append(labels[i].item())

            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_acc / len(val_loader)
            print(f"\n--- Epoch {epoch} Validation: Loss = {avg_val_loss:.4f}, Accuracy = {avg_val_acc*100:.2f}% ---\n")

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                os.makedirs(output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(output_dir, "option_marker.pt"))
                model.encoder.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                calib_config = {
                    "model_type": "option_marker",
                    "base_model": base_model_id,
                    "best_val_accuracy": round(best_val_acc, 4),
                    "epoch": epoch,
                    "timestamp": time.time(),
                }
                with open(os.path.join(output_dir, "marker_calibration.json"), "w") as f:
                    json.dump(calib_config, f, indent=2)

                if s3_target:
                    print(f"Syncing Epoch {epoch} checkpoint to S3: {s3_target} ...")
                    os.system(f"/usr/bin/aws s3 cp --recursive {output_dir}/ {s3_target}/ || aws s3 cp --recursive {output_dir}/ {s3_target}/")

    if is_main:
        calib_config = {
            "model_type": "option_marker",
            "base_model": base_model_id,
            "best_val_accuracy": round(best_val_acc, 4),
            "timestamp": time.time(),
        }
        with open(os.path.join(output_dir, "marker_calibration.json"), "w") as f:
            json.dump(calib_config, f, indent=2)

        if s3_target:
            print(f"Uploading artifacts to S3: {s3_target} ...")
            os.system(f"/usr/bin/aws s3 cp --recursive {output_dir}/ {s3_target}/ || aws s3 cp --recursive {output_dir}/ {s3_target}/")
            print("=== [OPTION-MARKER TRAINING COMPLETE] ===")

    if is_ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Option-Marker Joint Attention Model")
    parser.add_argument("--train_data", type=str, default="data_decision/train.jsonl")
    parser.add_argument("--val_data", type=str, default="data_decision/val.jsonl")
    parser.add_argument("--base_model_id", type=str, default="checkpoints/von-modernbert-rlcd")
    parser.add_argument("--output_dir", type=str, default="checkpoints/von-option-marker")
    parser.add_argument("--s3_target", type=str, default="s3://model-weight/von-option-marker")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_position_embeddings", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--brier_weight", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=8192,
                        help="Tokenizer truncation length during training. Must match inference-time "
                             "context or the scorer head never learns long-premise aggregation.")
    parser.add_argument("--no_length_bucketing", action="store_true",
                        help="Disable length-bucketed batching (uniform shuffling instead).")
    parser.add_argument("--max_tokens_per_batch", type=int, default=8192,
                        help="Padded token budget per batch. Long batches get fewer rows so a "
                             "single long document cannot OOM the card.")
    parser.add_argument("--long_threshold", type=int, default=2048,
                        help="Token count at or above which an example counts as long.")
    parser.add_argument("--long_ratio", type=float, default=0.30,
                        help="Target fraction of batches drawn from long examples.")
    args = parser.parse_args()

    train(
        train_path=args.train_data,
        val_path=args.val_data,
        base_model_id=args.base_model_id,
        output_dir=args.output_dir,
        s3_target=args.s3_target,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        brier_weight=args.brier_weight,
        max_position_embeddings=args.max_position_embeddings,
        max_length=args.max_length,
        length_bucketing=not args.no_length_bucketing,
        max_tokens_per_batch=args.max_tokens_per_batch,
        long_threshold=args.long_threshold,
        long_ratio=args.long_ratio,
    )
