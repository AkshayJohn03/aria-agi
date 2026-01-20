#!/usr/bin/env python3
"""
train_zia_ift_safe.py

Safe Instruction Fine-Tuning (IFT) script for ZIA Dense model.
Features:
 - Loads tokenizer from provided path.
 - Loads base weights-only from a "recovered_best.pt" (optional).
 - Prepares IFT datasets (HF Alpaca / OpenAssistant / Dolly) or uses local arrow dataset.
 - Conservative hyperparams for safe IFT smoke-test.
 - Robust atomic checkpoints and timestamped best copies.
 - Windows-safe DataLoader settings (low num_workers, disable mmap).
 - TensorBoard logging + console heartbeats.
"""

import os
import sys
import argparse
import time
import math
import glob
import shutil
import json
import traceback
from dataclasses import dataclass
from typing import Optional, List

# ensure HF env vars set early (avoid TRANSFORMERS_CACHE warnings)
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
# allow fallback allocation behavior
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

# -------------------------
# Config
# -------------------------
@dataclass
class ZiaConfig:
    # Model/tokenizer
    tokenizer_path: str = "artifacts/zia_tokenizer_60k"
    max_len: int = 256
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1

    # Data / training
    dataset_path: str = "my_datasets/processed/arrow_cleaned_v1"  # optional local dataset
    output_dir: str = "artifacts/zia_ift_runs"
    batch_size: int = 4           # micro-batch for IFT smoke-test
    grad_accum: int = 8
    lr: float = 5e-05             # conservative start; lower if unstable
    weight_decay: float = 0.0
    warmup_steps: int = 100
    max_steps: int = 20000        # short first run
    save_every_steps: int = 500
    eval_every_steps: int = 500
    keep_last: int = 5
    num_workers: int = 4
    fp16: bool = True
    seed: int = 42

    # loading behavior
    load_weights_only_from: Optional[str] = "artifacts/zia_dense_base/recovered_best.pt"
    resume_optimizer: bool = False  # DO NOT resume optimizer by default (safe)
    timestamp_best_copies: bool = True

    # dataset choices (HF keys)
    use_alpaca: bool = True
    use_oasst: bool = False
    use_dolly: bool = False
    hf_alpaca_name: str = "yahma/alpaca-cleaned"  # common cleaned alpaca
    hf_oasst_name: str = "OpenAssistant/oasst1"   # large; use subsets
    hf_dolly_name: str = "databricks/dolly_v2_12b"  # example; you'll likely not load huge models via dataset

    # logging
    log_every: int = 100
    tb_comment: str = "zia_ift"

# -------------------------
# Model (TinyGPT identical to your current architecture)
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
        self.pad_token_id = 0
        # weight tie
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
# Checkpoint helpers (atomic + timestamped best)
# -------------------------
def _atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_checkpoint_atomic(model, optimizer, scheduler, scaler, step, out_dir, keep_last=5, tag=None, best_metric=float("inf")):
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    name = f"step_{step:06d}.pt" if tag is None else f"{tag}_step_{step:06d}.pt"
    path = os.path.join(ckpt_dir, name)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "tag": tag or f"step_{step}",
        "best_metric": best_metric
    }
    _atomic_save(payload, path)
    print(f"[💾] Saved checkpoint {path}")

    # prune old step_* checkpoints (keep last keep_last)
    step_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    if len(step_ckpts) > keep_last:
        for old in step_ckpts[:-keep_last]:
            try:
                os.remove(old)
            except Exception:
                pass

def save_best_checkpoint(model, optimizer, scheduler, scaler, step, out_dir, best_metric, timestamped=True):
    best_dir = os.path.join(out_dir, "best_val")
    os.makedirs(best_dir, exist_ok=True)
    best_path = os.path.join(best_dir, "checkpoint.pt")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "tag": "best_val",
        "best_metric": best_metric
    }
    _atomic_save(payload, best_path)
    print(f"[🏆] Saved best_val to: {best_path}")

    if timestamped:
        ts = time.strftime("%Y%m%d-%H%M%S")
        copy_path = os.path.join(best_dir, f"best_{best_metric:.6f}_step{step}_{ts}.pt")
        _atomic_save({
            "model": model.state_dict(),
            "step": step,
            "best_metric": best_metric
        }, copy_path)
        print(f"[🏆] Also saved timestamped best copy: {copy_path}")

def load_weights_only_if_exists(model, path, map_location="cpu"):
    if path and os.path.exists(path):
        print(f"[i] Loading model weights-only from: {path}")
        try:
            # Some torch versions may support weights_only flag; try gracefully
            ckpt = torch.load(path, map_location=map_location)
        except TypeError:
            # fallback to standard load
            ckpt = torch.load(path, map_location=map_location)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            # assume this file is a raw state_dict
            model.load_state_dict(ckpt)
        print("[i] Weights loaded (model only).")
        return True
    return False

# -------------------------
# Dataset helpers
# -------------------------
def make_ift_text_from_alpaca(example):
    # alpaca-cleaned fields: instruction, input (sometimes), output
    inst = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output") or example.get("response") or example.get("output_text") or ""
    prompt = inst
    if inp and len(inp.strip()) > 0:
        prompt = f"{inst}\n{inp}"
    text = f"### Instruction:\n{prompt}\n\n### Response:\n{out}"
    return {"text": text}

def make_ift_text_from_oasst(example):
    # OASST samples vary; many have 'response' or 'text' fields. We'll attempt basic formatting.
    prompt = example.get("input", example.get("prompt", ""))
    out = example.get("response", example.get("text", ""))
    text = f"### Instruction:\n{prompt}\n\n### Response:\n{out}"
    return {"text": text}

def prepare_ift_dataset(cfg: ZiaConfig, max_samples:int=10000):
    """Prepare an IFT dataset by optionally pulling HF datasets and returning a Dataset object
       with 'text' field ready for collator. This is intentionally conservative (small subset)."""
    ds_list = []

    if cfg.use_alpaca:
        try:
            print("[i] Loading Alpaca dataset (subset)...")
            d = load_dataset(cfg.hf_alpaca_name, split=f"train[:{max_samples}]")
            d = d.map(lambda ex: make_ift_text_from_alpaca(ex))
            ds_list.append(d)
        except Exception as e:
            print("[!] Failed to load Alpaca:", e)

    if cfg.use_oasst:
        try:
            print("[i] Loading OpenAssistant (subset)...")
            # use a small slice by default
            d = load_dataset(cfg.hf_oasst_name, split=f"train[:{max_samples}]")
            d = d.map(lambda ex: make_ift_text_from_oasst(ex))
            ds_list.append(d)
        except Exception as e:
            print("[!] Failed to load OASST:", e)

    if cfg.use_dolly:
        try:
            print("[i] Loading Dolly (subset)...")
            d = load_dataset(cfg.hf_dolly_name, split=f"train[:{max_samples}]")
            # Dolly fields can vary; attempt to combine fields
            d = d.map(lambda ex: {"text": (ex.get("instruction","") or "") + "\n\n" + (ex.get("output","") or "")})
            ds_list.append(d)
        except Exception as e:
            print("[!] Failed to load Dolly:", e)

    # If there's a local arrow dataset, prefer that (more domain-specific)
    if cfg.dataset_path and os.path.exists(cfg.dataset_path):
        try:
            print("[i] Loading local arrow dataset (if present) as extra data...")
            ds_local = load_from_disk(cfg.dataset_path)
            # if train split exists, grab a small piece
            if isinstance(ds_local, dict) and "train" in ds_local:
                small = ds_local["train"].select(range(min(len(ds_local["train"]), max_samples)))
            else:
                small = ds_local.select(range(min(len(ds_local), max_samples)))
            # normalize to 'text' if not already
            def _norm(ex):
                if "text" in ex:
                    return {"text": ex["text"]}
                if "messages" in ex:
                    # flatten messages
                    parts = []
                    for m in ex["messages"]:
                        if isinstance(m, dict):
                            role = m.get("role","")
                            content = m.get("content") or m.get("text") or ""
                            parts.append(f"{role}: {content}")
                        else:
                            parts.append(str(m))
                    return {"text": "\n".join(parts)}
                # fallback
                return {"text": str(ex)}
            small = small.map(_norm)
            ds_list.append(small)
        except Exception as e:
            print("[!] Failed to load local dataset:", e)

    if not ds_list:
        raise RuntimeError("No datasets available for IFT. Enable at least one HF dataset or provide local dataset_path.")

    # concat all small datasets
    if len(ds_list) > 1:
        merged = concatenate_datasets(ds_list)
    else:
        merged = ds_list[0]

    # dedupe & filter very short items
    def _filter_short(ex):
        t = ex.get("text","") or ""
        return len(t.strip().split()) >= 3
    merged = merged.filter(_filter_short)
    print(f"[i] Prepared IFT dataset size: {len(merged)}")
    return merged

# -------------------------
# Collate
# -------------------------
class Collate:
    def __init__(self, tokenizer:AutoTokenizer, eos_token:str, max_len:int):
        self.tok = tokenizer
        self.eos = eos_token or "</s>"
        self.max_len = max_len

    def __call__(self, batch):
        texts = []
        for ex in batch:
            t = ex.get("text") if isinstance(ex, dict) else str(ex)
            if not t.endswith(self.eos):
                t = t + self.eos
            texts.append(t)
        enc = self.tok(texts, truncation=True, padding=True, max_length=self.max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

# -------------------------
# Eval routine
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device, max_batches:Optional[int]=None):
    model.eval()
    total = 0.0
    n = 0
    pad = model.pad_token_id if hasattr(model, "pad_token_id") else 0
    for i, (ids, mask) in enumerate(loader, start=1):
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits, _ = model(ids, attention_mask=mask, labels=ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               ids[:, 1:].reshape(-1),
                               ignore_index=pad)
        total += loss.item()
        n += 1
        if max_batches and i >= max_batches:
            break
    model.train()
    return total / max(1, n)

# -------------------------
# Training loop
# -------------------------
def train_loop(cfg:ZiaConfig):
    # detect windows and adjust
    is_windows = (os.name == "nt")
    if is_windows:
        print("[i] Windows detected -> lowering num_workers and disabling dataset mmap to avoid arrow multiprocessing issues.")
        cfg.num_workers = min(2, cfg.num_workers)
        os.environ["HF_DATASETS_DISABLE_MMAP"] = "1"

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Device: {device} | CUDA available: {torch.cuda.is_available()}")

    # tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token":"<pad>"})
    eos_token = tok.eos_token or "</s>"
    print(f"[i] Tokenizer loaded: vocab={len(tok)} pad={tok.pad_token_id} eos={tok.eos_token_id}")

    # dataset (prepare IFT dataset)
    ds = prepare_ift_dataset(cfg, max_samples=10000)
    # split to train/val small
    splits = ds.train_test_split(test_size=0.02, seed=cfg.seed)
    train_ds = splits["train"]
    val_ds = splits["test"]
    print(f"[i] IFT dataset splits: train={len(train_ds)} val={len(val_ds)}")

    # dataloaders
    collate = Collate(tokenizer=tok, eos_token=eos_token, max_len=cfg.max_len)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=pin,
                              persistent_workers=(cfg.num_workers > 0),
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=min(2, cfg.num_workers), pin_memory=pin,
                            collate_fn=collate, drop_last=False)

    print(f"[i] Dataloaders ready. Train batches: {len(train_loader)}")

    # model
    model = TinyGPT(
        vocab_size=len(tok),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        mlp_ratio=cfg.mlp_ratio,
        max_len=cfg.max_len,
        dropout=cfg.dropout
    ).to(device)
    model.pad_token_id = tok.pad_token_id

    # load weights-only if exists
    if cfg.load_weights_only_from:
        loaded = load_weights_only_if_exists(model, cfg.load_weights_only_from, map_location=device)
        if loaded:
            print("[i] Base weights loaded; starting fine-tune from weights-only (optimizer state NOT restored).")

    # optimizer/scheduler (fresh by default)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.max_steps)

    scaler = None
    if cfg.fp16 and device.type == "cuda":
        scaler = torch.amp.GradScaler()

    # tensorboard
    os.makedirs(cfg.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg.output_dir, "runs"), comment=cfg.tb_comment)

    # training state
    step = 0
    raw_loss_accum = 0.0
    micro_counter = 0
    tokens_seen = 0
    t0 = time.time()
    best_val = float("inf")

    pbar = tqdm(total=cfg.max_steps * cfg.grad_accum, desc="IFT Training", dynamic_ncols=True)

    try:
        while step < cfg.max_steps:
            for ids, mask in train_loader:
                ids = ids.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=(cfg.fp16 and device.type=="cuda")):
                    _, loss = model(ids, attention_mask=mask, labels=ids)

                # guard
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[{time.strftime('%H:%M:%S')}] [!] NaN/Inf loss detected; skipping batch.")
                    continue

                raw_loss_accum += loss.item()
                tokens_seen += (ids.size(1) - 1) * ids.size(0)

                micro_loss = loss / cfg.grad_accum
                if scaler:
                    scaler.scale(micro_loss).backward()
                else:
                    micro_loss.backward()

                micro_counter += 1
                pbar.update(1)

                if micro_counter % cfg.grad_accum == 0:
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

                    # logging
                    if step % max(1, cfg.log_every) == 0:
                        avg_loss = raw_loss_accum / (cfg.log_every if cfg.log_every>0 else 1)
                        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                        elapsed = time.time() - t0
                        tps = tokens_seen / max(1e-6, elapsed)
                        writer.add_scalar("train/cross_entropy", avg_loss, step)
                        writer.add_scalar("train/ppl", ppl, step)
                        writer.add_scalar("train/grad_norm", grad_norm, step)
                        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                        if torch.cuda.is_available():
                            writer.add_scalar("sys/gpu_mem_mb", torch.cuda.memory_allocated()/1024**2, step)
                        print(f"[{time.strftime('%H:%M:%S')}] step={step:,} avg_loss={avg_loss:.4f} ppl={ppl:.2f} lr={scheduler.get_last_lr()[0]:.2e}")
                        raw_loss_accum = 0.0
                        tokens_seen = 0
                        t0 = time.time()

                    # eval
                    if step % cfg.eval_every_steps == 0:
                        val_loss = evaluate(model, val_loader, device, max_batches=200)
                        val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                        writer.add_scalar("eval/cross_entropy", val_loss, step)
                        writer.add_scalar("eval/ppl", val_ppl, step)
                        writer.flush()
                        print(f"[{time.strftime('%H:%M:%S')}] [Eval] step={step:,} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
                        if val_loss < best_val:
                            best_val = val_loss
                            save_best_checkpoint(model, optimizer, scheduler, scaler, step, cfg.output_dir, best_val, timestamped=cfg.timestamp_best_copies)

                    # checkpointing
                    if step % cfg.save_every_steps == 0:
                        save_checkpoint_atomic(model, optimizer, scheduler, scaler, step, cfg.output_dir, keep_last=cfg.keep_last, best_metric=best_val)
                        print(f"[{time.strftime('%H:%M:%S')}] [⏺] Checkpoint saved at step {step}")

                if step >= cfg.max_steps:
                    break
            if step >= cfg.max_steps:
                break

        # final save
        save_checkpoint_atomic(model, optimizer, scheduler, scaler, step, cfg.output_dir, keep_last=cfg.keep_last, tag="final", best_metric=best_val)
        print("[✓] IFT training complete. Final checkpoint saved.")
    except KeyboardInterrupt:
        print("[!] KeyboardInterrupt — saving interrupt checkpoint...")
        save_checkpoint_atomic(model, optimizer, scheduler, scaler, step, cfg.output_dir, keep_last=cfg.keep_last, tag="interrupt", best_metric=best_val)
        raise
    except Exception as e:
        print("[!] Exception during training — saving error checkpoint...")
        traceback.print_exc()
        save_checkpoint_atomic(model, optimizer, scheduler, scaler, step, cfg.output_dir, keep_last=cfg.keep_last, tag="error", best_metric=best_val)
        raise
    finally:
        writer.close()
        pbar.close()

# -------------------------
# CLI Entrypoint
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--dataset_path", default=None)
    p.add_argument("--load_weights_only_from", default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--use_alpaca", action="store_true")
    p.add_argument("--use_oasst", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = ZiaConfig()
    # override with CLI if provided
    if args.tokenizer_path:
        cfg.tokenizer_path = args.tokenizer_path
    if args.dataset_path:
        cfg.dataset_path = args.dataset_path
    if args.load_weights_only_from:
        cfg.load_weights_only_from = args.load_weights_only_from
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.use_alpaca:
        cfg.use_alpaca = True
    if args.use_oasst:
        cfg.use_oasst = True

    print("[i] Starting safe IFT with config:")
    print(json.dumps(cfg.__dict__, indent=2, default=str))
    train_loop(cfg)
