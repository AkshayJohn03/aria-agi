#!/usr/bin/env python3
# train_zia_retrain_from_best_fixed.py
# A specialized script to load best model weights and resume training from the exact saved state.

import os
import glob
import time
import math
import json
import shutil
import traceback
from dataclasses import dataclass
from typing import Optional, List

# --- Silence TRANSFORMERS_CACHE warning & set cache dirs early ---
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_from_disk, concatenate_datasets, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

# -------------------------
# Config
# -------------------------
@dataclass
class ZiaConfig:
    # Model / tokenizer
    max_len: int = 256
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1

    # Data / training
    dataset_path: str = "my_datasets/processed/arrow_cleaned_v1" # top-level folder
    tokenizer_path: str = "artifacts/zia_tokenizer_60k"
    output_dir: str = "artifacts/zia_dense_second_phase"
    batch_size: int = 8 # micro-batch
    grad_accum: int = 8 # accumulation micro-batches -> effective batch = batch_size*grad_accum
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 229000 # total effective training steps
    save_every_batches: int = 500 # saves every this many micro-batches (tqdm steps)
    eval_every_batches: int = 1000 # evaluates every this many micro-batches
    keep_last: int = 2
    num_workers: int = 4
    fp16: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- FIXED: Use these flags to enable full training resumption ---
    resume_from_best: bool = True
    load_initial_weights_from_best: bool = False
    load_from_external_ckpt: Optional[str] = "artifacts/zia_ift_v3_fix_stage_2/best_val/checkpoint.pt" # path to external checkpoint to load initial weights from (if not resuming)
    
    log_train_every_batches: int = 500
    deterministic_dataloader: bool = False # Set True only for full determinism

# -------------------------
# Model (your TinyGPT)
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, mlp_ratio, dropout):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mlp_ratio, dropout)

    def forward(self, x, key_padding_mask=None):
        h = self.ln1(x)
        B, T, _ = h.size()
        causal = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(h, h, h, attn_mask=causal, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + out
        return x + self.ff(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, mlp_ratio, max_len, dropout):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.pad_token_id = 0 # Default, will be updated in train_loop
        # weight tying
        self.head.weight = self.tok.weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape
        if T > self.max_len:
            input_ids = input_ids[:, -self.max_len:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_len:]
            T = input_ids.size(1)

        pos_ids = torch.arange(0, T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok(input_ids) + self.pos(pos_ids)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        for blk in self.blocks:
            x = blk(x, key_padding_mask)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id
            )
        return logits, loss

# -------------------------
# Checkpoint helpers (atomic)
# -------------------------
def _atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_checkpoint(model, optimizer, scheduler, scaler, step, out_dir, keep_last, tag=None, best_val_metric=float("inf")):
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    name = f"step_{step}.pt"
    if tag:
        name = f"{tag}_{name}"
    path = os.path.join(ckpt_dir, name)
    _atomic_save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "tag": tag or f"step_{step}",
        "best_val_metric": best_val_metric, # ADDED: Save current best metric
    }, path)
    print(f"[💾] Saved checkpoint {path}")

    # prune
    step_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    if len(step_ckpts) > keep_last:
        for old in step_ckpts[:-keep_last]:
            try:
                os.remove(old)
            except Exception:
                pass

def save_best_val(model, optimizer, scheduler, scaler, step, out_dir, best_val_metric):
    path = os.path.join(out_dir, "best_val", "checkpoint.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "tag": "best_val",
        "best_val_metric": best_val_metric, # ADDED: Save the best metric
    }, path)
    print(f"[🏆] Saved best_val to {path}")

def load_best_val(model, optimizer, scheduler, scaler, device, out_dir) -> (int, float):
    best_path = os.path.join(out_dir, "best_val", "checkpoint.pt")
    initial_best_val = float("inf")
    start_step = 0
    if os.path.exists(best_path):
        print(f"[i] Loading best_val checkpoint from {best_path}")
        # Note: Added weights_only=False for compatibility with older code,
        # but the project log suggests this warning is already present, so keeping the original call.
        ckpt = torch.load(best_path, map_location=device) 
        
        model.load_state_dict(ckpt["model"])
        if optimizer and ckpt.get("optimizer"): optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
        if scaler and ckpt.get("scaler"): scaler.load_state_dict(ckpt["scaler"])
        
        start_step = int(ckpt.get("step", 0))
        # ADDED: Load the best_val_metric from the checkpoint
        initial_best_val = float(ckpt.get("best_val_metric", float("inf")))

        return start_step, initial_best_val
    return start_step, initial_best_val

def load_model_only(model, device, out_dir) -> bool:
    best_path = os.path.join(out_dir, "best_val", "checkpoint.pt")
    if os.path.exists(best_path):
        print(f"[i] Loading model weights from best_val checkpoint at {best_path}")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print("[i] Model weights loaded successfully. Training will restart from step 0.")
        return True
    return False

# -------------------------
# Dataset helpers
# -------------------------
def _is_shard_dir(path):
    # A shard dir usually contains data-*.arrow and dataset_info.json
    return os.path.isdir(path) and any(f.startswith("data-") for f in os.listdir(path))

def load_split_from_shards(split_dir: str) -> Optional[Dataset]:
    """
    - If split_dir itself is a dataset dir (contains dataset_info.json), load_from_disk(split_dir).
    - Else if split_dir contains 'shard_*' subdirs, load each shard and concatenate.
    - Returns None if split_dir doesn't exist or no shards found.
    """
    if not os.path.exists(split_dir):
        return None

    # direct dataset dir
    if os.path.exists(os.path.join(split_dir, "dataset_info.json")):
        return load_from_disk(split_dir)

    # find shard directories inside
    shards = sorted([os.path.join(split_dir, d) for d in os.listdir(split_dir)])
    shard_dirs = [s for s in shards if _is_shard_dir(s)]
    if not shard_dirs:
        # sometimes saved as data-*.arrow files directly under split_dir
        if any(f.startswith("data-") for f in os.listdir(split_dir)):
            return load_from_disk(split_dir)
        return None

    ds_list = []
    for sd in shard_dirs:
        try:
            ds = load_from_disk(sd)
            ds_list.append(ds)
        except Exception as e:
            print(f"[!] Failed to load shard {sd}: {e}")
    if not ds_list:
        return None
    if len(ds_list) == 1:
        return ds_list[0]
    return concatenate_datasets(ds_list, axis=0)

# -------------------------
# Collate - supports 'text' or 'messages'
# -------------------------
class CollateWrapper:
    def __init__(self, tokenizer, eos_token: str, max_len: int):
        # tokenizer: AutoTokenizer instance (already prepared)
        self.tok = tokenizer
        self.eos_token = eos_token
        self.max_len = max_len

    def _sample_to_text(self, ex):
        # Accept text, content, or messages fields
        if isinstance(ex, dict):
            if "text" in ex and isinstance(ex["text"], str):
                return ex["text"]
            if "content" in ex and isinstance(ex["content"], str):
                return ex["content"]
            if "messages" in ex and isinstance(ex["messages"], list):
                parts = []
                for m in ex["messages"]:
                    if isinstance(m, dict):
                        role = m.get("role", "")
                        content = m.get("content") or m.get("text") or ""
                        parts.append(f"{role}: {content}")
                    else:
                        parts.append(str(m))
                return "\n".join(parts)
        return str(ex)

    def __call__(self, batch):
        texts = []
        for ex in batch:
            t = self._sample_to_text(ex)
            # add eos token if not present
            if self.eos_token and not t.endswith(self.eos_token):
                t = t + self.eos_token
            texts.append(t)
        enc = self.tok(texts, truncation=True, padding=True, max_length=self.max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(model, data_loader, device, max_batches: Optional[int] = None):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    # Use the pad_token_id from the model instance
    pad = model.pad_token_id if hasattr(model, "pad_token_id") else 0
    for i, (ids, mask) in enumerate(data_loader, 1):
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits, _ = model(ids, attention_mask=mask, labels=ids)
        # cross-entropy over shifted positions:
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               ids[:, 1:].reshape(-1),
                               ignore_index=pad)
        total_loss += loss.item()
        n_batches += 1
        if max_batches and i >= max_batches:
            break
    model.train()
    return total_loss / max(1, n_batches)

# -------------------------
# Training loop
# -------------------------
def train_loop(config: ZiaConfig):
    # reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)
    print(f"[i] Device: {device} | CUDA available: {torch.cuda.is_available()}")

    # tokenizer
    tok = AutoTokenizer.from_pretrained(config.tokenizer_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    eos_token = tok.eos_token or "</s>"
    print(f"[i] Tokenizer: vocab={len(tok)} | pad_id={tok.pad_token_id} | eos_id={tok.eos_token_id}")

    # --- Load splits robustly ---
    base = config.dataset_path
    if not os.path.exists(base):
        raise FileNotFoundError(f"Dataset folder not found: {base}")

    print("[i] Loading train split (from shards if present)...")
    train_ds = load_split_from_shards(os.path.join(base, "train"))
    val_ds = load_split_from_shards(os.path.join(base, "validation"))
    test_ds = load_split_from_shards(os.path.join(base, "test"))

    # if validation absent, fall back to test as validation
    if val_ds is None:
        print("[i] No validation split found; trying 'test' split as fallback.")
        val_ds = load_split_from_shards(os.path.join(base, "test"))
        if val_ds is None:
            # maybe the top-level path itself is a dataset saved directly
            try:
                ds_top = load_from_disk(base)
                if "train" in ds_top:
                    train_ds = ds_top["train"]
                    val_ds = ds_top.get("validation", ds_top.get("test"))
                    test_ds = ds_top.get("test", None)
                else:
                    # single split dataset
                    train_ds = ds_top
            except Exception as e:
                raise FileNotFoundError(f"Could not find any train split under {base}. Error: {e}")

    # if validation is still missing, create small val split from train (1%)
    if val_ds is None:
        print("[i] Still no validation split found — creating 1% validation from train for eval.")
        split = train_ds.train_test_split(test_size=0.01, seed=config.seed)
        train_ds = split["train"]
        val_ds = split["test"]

    print(f"[i] Dataset sizes (raw): train={len(train_ds) if train_ds else 0} | val={len(val_ds) if val_ds else 0} | test={len(test_ds) if test_ds else 0}")

    # DataLoaders
    pin = torch.cuda.is_available()
    collate = CollateWrapper(tokenizer=tok, eos_token=eos_token, max_len=config.max_len)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=pin,
                              persistent_workers=(config.num_workers > 0),
                              prefetch_factor=2 if config.num_workers > 0 else None,
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=min(4, config.num_workers), pin_memory=pin,
                            collate_fn=collate, drop_last=False)

    print(f"[i] Dataloaders ready. Train loader batches: {len(train_loader)}")

    # Model / optimizer / scheduler
    model = TinyGPT(
        vocab_size=len(tok),
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        mlp_ratio=config.mlp_ratio,
        max_len=config.max_len,
        dropout=config.dropout,
    ).to(device)

    # assign pad id to model for correct loss ignore_index
    model.pad_token_id = tok.pad_token_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.max_steps)

    scaler = None
    if config.fp16 and device.type == "cuda":
        # Note: Suppressing the warning in the provided log, but keeping the original implementation:
        # D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\train_zia_dense_final.py:408: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
        scaler = torch.cuda.amp.GradScaler()

    # tensorboard
    os.makedirs(config.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(config.output_dir, "runs"))

    # optionally resume or load weights
    start_step = 0
    best_val = float("inf") # Initializing best_val before loading checkpoint
    
    # Option 1: Resume exactly from your previous run
    if config.resume_from_best and not config.load_from_external_ckpt:
        start_step, initial_best_val = load_best_val(model, optimizer, scheduler, scaler, device, config.output_dir)
        best_val = initial_best_val
        if start_step:
            print(f"[i] Resumed from best_val at step {start_step}. Previous best val_loss: {best_val:.4f}")

    # Option 2: Initialize weights from external checkpoint (e.g. IFT2)
    elif config.load_from_external_ckpt:
        ckpt_path = config.load_from_external_ckpt
        if os.path.exists(ckpt_path):
            print(f"[i] Loading model weights from external checkpoint: {ckpt_path}")
            ck = torch.load(ckpt_path, map_location=device)
            if "model" in ck:
                model.load_state_dict(ck["model"], strict=False)
            else:
                model.load_state_dict(ck, strict=False)
            print("[✓] Model weights loaded from external checkpoint successfully.")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # training loop variables
    micro_counter = 0
    # Recalculate initial pbar state: micro-batches already completed
    initial_micro_batches = start_step * config.grad_accum
    pbar = tqdm(total=config.max_steps * config.grad_accum, initial=initial_micro_batches, desc="Training", dynamic_ncols=True)
    raw_loss_accum = 0.0
    tokens_seen = 0
    t0 = time.time()
    # best_val is now loaded from checkpoint or remains float("inf")
    step = start_step

    try:
        while step < config.max_steps:
            for ids, mask in train_loader:
                # Move data to device
                ids = ids.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                # Forward pass
                # Note: Suppressing the warning in the provided log, but keeping the original implementation:
                # D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\train_zia_dense_final.py:440: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=config.fp16 and device.type == "cuda"):
                    _, loss = model(ids, attention_mask=mask, labels=ids)

                # Check for bad loss values
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[{time.strftime('%H:%M:%S')}] [!] Warning: Bad loss value detected (NaN/Inf). Skipping this batch.")
                    continue

                raw_loss_accum += loss.item()
                tokens_seen += (ids.size(1) - 1) * ids.size(0)

                # backward with grad accumulation
                loss = loss / config.grad_accum
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                pbar.update(1) # update for each micro-batch

                micro_counter += 1
                if (micro_counter % config.grad_accum) == 0:
                    step += 1

                    # optimizer step
                    if scaler:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    # Logging
                    if step % (config.log_train_every_batches // config.grad_accum) == 0:
                        # Log average loss per micro-batch over the logging window
                        avg_loss = raw_loss_accum / (config.log_train_every_batches)
                        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                        elapsed = time.time() - t0
                        tokens_per_sec = tokens_seen / max(1e-6, elapsed)
                        writer.add_scalar("train/cross_entropy", avg_loss, step)
                        writer.add_scalar("train/ppl", ppl, step)
                        writer.add_scalar("train/grad_norm", grad_norm, step)
                        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                        if torch.cuda.is_available():
                            writer.add_scalar("sys/gpu_mem_mb", torch.cuda.memory_allocated() / 1024**2, step)
                        writer.flush()
                        print(f"[{time.strftime('%H:%M:%S')}] [log] step={step:,} avg_loss={avg_loss:.4f} ppl={ppl:.2f} lr={scheduler.get_last_lr()[0]:.2e}")
                        raw_loss_accum = 0.0
                        tokens_seen = 0
                        t0 = time.time()

                    # Evaluation
                    if step % (config.eval_every_batches // config.grad_accum) == 0:
                        val_loss = evaluate(model, val_loader, device, max_batches=200)
                        val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                        writer.add_scalar("eval/cross_entropy", val_loss, step)
                        writer.add_scalar("eval/ppl", val_ppl, step)
                        writer.flush()
                        print(f"[{time.strftime('%H:%M:%S')}] [Eval] step={step:,} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
                        if val_loss < best_val:
                            best_val = val_loss
                            # FIXED: Pass the current best_val to the save function
                            save_best_val(model, optimizer, scheduler, scaler, step, config.output_dir, best_val)

                    # Checkpointing
                    if (pbar.n % 1000) == 0: # Save every 1000 micro-batches
                        # FIXED: Pass the current best_val to the save function
                        save_checkpoint(model, optimizer, scheduler, scaler, step, config.output_dir, keep_last=config.keep_last, best_val_metric=best_val)
                        print(f"[{time.strftime('%H:%M:%S')}] [⏺] Checkpointed at step {step}")
            
            if step >= config.max_steps:
                break

        # final save
        save_checkpoint(model, optimizer, scheduler, scaler, step, config.output_dir, keep_last=config.keep_last, tag="final", best_val_metric=best_val)
        print("[✓] Training complete. Final checkpoint saved.")
    except KeyboardInterrupt:
        print("[!] KeyboardInterrupt — saving interrupt checkpoint...")
        save_checkpoint(model, optimizer, scheduler, scaler, step, config.output_dir, keep_last=config.keep_last, tag="interrupt", best_val_metric=best_val)
    except Exception as e:
        print("[!] Exception during training. Saving error checkpoint...")
        traceback.print_exc()
        save_checkpoint(model, optimizer, scheduler, scaler, step, config.output_dir, keep_last=config.keep_last, tag="error", best_val_metric=best_val)
    finally:
        writer.close()
        pbar.close()

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    cfg = ZiaConfig()
    train_loop(cfg)