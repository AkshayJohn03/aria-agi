#!/usr/bin/env python3
# train_zia_ift.py
# Instruction Fine-Tuning of Zia on Alpaca-52k

import os, time, math, sys
sys.path.append(os.path.abspath("."))  # add repo root to path

from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from train_zia_dense_final import (
    TinyGPT, evaluate,
    save_checkpoint, save_best_val, load_best_val
)

# --- Collator with EOS enforcement ---
class CollateWrapper:
    def __init__(self, tokenizer, eos_token: str, max_len: int):
        self.tok = tokenizer
        self.eos_token = eos_token
        self.max_len = max_len

    def __call__(self, batch):
        texts = []
        for ex in batch:
            t = ex["text"]
            if not t.endswith(self.eos_token):
                t += self.eos_token
            texts.append(t)
        enc = self.tok(
            texts, truncation=True, padding=True,
            max_length=self.max_len, return_tensors="pt"
        )
        return enc["input_ids"], enc["attention_mask"]

@dataclass
class IFTConfig:
    # Model
    max_len: int = 256
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1

    # Data / training
    dataset_path: str = "my_datasets/processed/alpaca_ift"
    tokenizer_path: str = "artifacts/zia_tokenizer_60k"
    output_dir: str = "artifacts/zia_dense_ift"
    batch_size: int = 8
    grad_accum: int = 8
    lr: float = 5e-5
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_epochs: int = 4
    save_every_batches: int = 500
    eval_every_batches: int = 500
    keep_last: int = 2
    num_workers: int = 2
    fp16: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    resume_from_best: bool = False
    load_initial_weights_from_best: bool = True
    base_ckpt_dir: str = "artifacts/zia_dense_runs/best_val"

def train_loop(cfg: IFTConfig):
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device)

    print(f"[i] Device: {device}")
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    eos_token = tok.eos_token or "</s>"

    # Load dataset
    ds = load_from_disk(cfg.dataset_path)
    train_ds, val_ds = ds["train"], ds["validation"]
    print(f"[i] Dataset: train={len(train_ds)} | val={len(val_ds)}")

    collate = CollateWrapper(tok, eos_token, cfg.max_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate, drop_last=False)

    # Debug check first batch
    ids, mask = next(iter(train_loader))
    print("[debug] Example batch:")
    print("input_ids:", ids[0][:50].tolist())
    print("decoded:", tok.decode(ids[0]))

    # Model
    model = TinyGPT(
        vocab_size=len(tok),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        mlp_ratio=cfg.mlp_ratio,
        max_len=cfg.max_len,
        dropout=cfg.dropout,
    ).to(device)
    model.pad_token_id = tok.pad_token_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    max_steps = len(train_loader) * cfg.max_epochs // cfg.grad_accum
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg.warmup_steps, max_steps)
    scaler = torch.cuda.amp.GradScaler() if cfg.fp16 and device.type == "cuda" else None

    os.makedirs(cfg.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg.output_dir, "runs"))

    # Load weights from base best_val
    if cfg.load_initial_weights_from_best:
        ckpt = torch.load(os.path.join(cfg.base_ckpt_dir, "checkpoint.pt"), map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[i] Loaded weights from {cfg.base_ckpt_dir}/checkpoint.pt")

    best_val = float("inf")
    step = 0
    raw_loss_accum = 0
    micro_counter = 0
    tokens_seen = 0
    t0 = time.time()
    pbar = tqdm(total=max_steps * cfg.grad_accum, desc="Training IFT", dynamic_ncols=True)

    try:
        for epoch in range(cfg.max_epochs):
            print(f"[i] Starting epoch {epoch+1}/{cfg.max_epochs}")
            for ids, mask in train_loader:
                ids, mask = ids.to(device), mask.to(device)
                with torch.cuda.amp.autocast(enabled=cfg.fp16 and device.type=="cuda"):
                    _, loss = model(ids, attention_mask=mask, labels=ids)
                raw_loss_accum += loss.item()
                tokens_seen += (ids.size(1)-1) * ids.size(0)

                loss = loss / cfg.grad_accum
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                pbar.update(1)
                micro_counter += 1

                if micro_counter % cfg.grad_accum == 0:
                    step += 1
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    # log
                    if step % 50 == 0:
                        avg_loss = raw_loss_accum / 50
                        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                        elapsed = time.time()-t0
                        print(f"[log] step={step} loss={avg_loss:.4f} ppl={ppl:.2f}")
                        writer.add_scalar("train/loss", avg_loss, step)
                        writer.add_scalar("train/ppl", ppl, step)
                        raw_loss_accum = 0
                        tokens_seen = 0
                        t0 = time.time()

                    # eval
                    if step % cfg.eval_every_batches == 0:
                        val_loss = evaluate(model, val_loader, device, max_batches=200)
                        val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                        print(f"[Eval] step={step} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
                        writer.add_scalar("val/loss", val_loss, step)
                        writer.add_scalar("val/ppl", val_ppl, step)
                        if val_loss < best_val:
                            best_val = val_loss
                            save_best_val(model, optimizer, scheduler, scaler, step, cfg.output_dir, val_loss)

                    if step % cfg.save_every_batches == 0:
                        save_checkpoint(model, optimizer, scheduler, scaler, step, cfg.output_dir, cfg.keep_last)
            # end epoch
    except KeyboardInterrupt:
        print("[!] Interrupted")
        save_checkpoint(model, optimizer, scheduler, scaler, step, cfg.output_dir, cfg.keep_last, tag="interrupt")
    finally:
        pbar.close()
        writer.close()

if __name__ == "__main__":
    cfg = IFTConfig()
    train_loop(cfg)
