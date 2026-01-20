#!/usr/bin/env python3
# ZIA LM Trainer v18 — Speed Optimized, DataLoader, Safe Config

import os, time, math, gc
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# ---------------- Environment ----------------
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------- CONFIG ----------------
# Optimize memory allocation for GTX 1050 Ti (Pascal)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hardware Speedups
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True # Optimizes for fixed input sizes

TRAIN_DIR = "datasets/processed/wikitext_60k_4096"
VAL_SHARD = "datasets/processed/wikitext_60k_4096/val_shard_0000"
RESUME_CKPT = "artifacts/zia_mixed_v1/checkpoint_mixed_step10703.pt"
OUTPUT_DIR = "artifacts/zia_lm_v18"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VOCAB_SIZE = 60011
D_MODEL = 384
LAYERS = 8
HEADS = 6
SEQ_LEN = 4096

# 1050 Ti Safe Settings
BATCH_SIZE = 1          # Keep 1 to prevent OOM at 4k context
GRAD_ACCUM = 80         # Higher accumulation to compensate for Batch=1
LR = 5e-5
WEIGHT_DECAY = 1e-2
CLIP_NORM = 0.5

SAVE_EVERY = 500       # steps
VAL_EVERY = 1000       # steps
USE_FP16 = True
# ----------------------------------------

# -------- Data Loader --------
class DiskDataset(Dataset):
    def __init__(self, shard_paths):
        self.files = sorted(shard_paths)
        # We load one shard into memory to avoid disk seek latency
        # If you have many shards, we can implement rotating buffers, 
        # but for WikiText, loading the current shard is fastest.
        print(f"[i] Loading shard: {self.files[0].name} into RAM...")
        self.x = torch.load(self.files[0] / "input_ids.pt", map_location="cpu")
        self.y = torch.load(self.files[0] / "labels.pt", map_location="cpu")
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Slicing here is fast
        x = self.x[idx][:SEQ_LEN].long()
        y = self.y[idx][:SEQ_LEN].long()
        return x, y

# -------- Model --------
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=HEADS, dim_feedforward=D_MODEL*4,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True
            )
            for _ in range(LAYERS)
        ])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, x):
        B, T = x.size()
        pos_ids = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos_ids)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.head(h)

# -------- Validation --------
@torch.no_grad()
def quick_val(model, samples=8):
    p = Path(VAL_SHARD)
    if not p.exists(): return None
    x = torch.load(p/"input_ids.pt", map_location="cpu")[:samples, :SEQ_LEN].long().to(DEVICE)
    y = torch.load(p/"labels.pt", map_location="cpu")[:samples, :SEQ_LEN].long().to(DEVICE)
    
    with torch.amp.autocast("cuda", enabled=USE_FP16):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=-100)
    return loss.item(), math.exp(min(loss.item(), 50))

# ---------------- MAIN TRAINER ----------------
def main():
    torch.manual_seed(1337)

    # 1. Model Init
    model = TinyGPT().to(DEVICE)
    model.head.weight = model.tok.weight

    # 2. Strict Resume
    if Path(RESUME_CKPT).exists():
        print(f"[i] Resuming from {RESUME_CKPT}...")
        ck = torch.load(RESUME_CKPT, map_location=DEVICE)
        st = ck.get("model", ck)
        
        # Check size mismatch
        if st["pos.weight"].shape[0] != SEQ_LEN:
            print(f"[!] Resizing Position Embeddings: {st['pos.weight'].shape[0]} -> {SEQ_LEN}")
            # Safe resize logic
            old_pos = st["pos.weight"]
            new_pos = model.pos.weight.data.clone()
            min_len = min(len(old_pos), SEQ_LEN)
            new_pos[:min_len] = old_pos[:min_len]
            st["pos.weight"] = new_pos
            
        model.load_state_dict(st, strict=True)
    else:
        print("[!] Warning: Starting from scratch (No checkpoint found).")

    # 3. Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_FP16)

    # 4. Data Loader (The Speedup)
    shard_paths = list(Path(TRAIN_DIR).glob("train_shard_*"))
    dataset = DiskDataset(shard_paths)
    
    # num_workers=2 allows CPU to prepare batch N+1 while GPU computes batch N
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,      # Background loading
        pin_memory=True,    # Faster RAM->VRAM transfer
        persistent_workers=True
    )

    # 5. Training Loop
    step = 0
    accum_steps = 0
    last_val_step = 0
    
    model.train()
    print(f"[i] Training started on {DEVICE}. Context: {SEQ_LEN}. Workers: 2")

    pbar = tqdm(dataloader, desc="Training")
    
    for x, y in pbar:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=USE_FP16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE), 
                y.view(-1), 
                ignore_index=-100
            )
            # Scale loss immediately
            loss = loss / GRAD_ACCUM

        scaler.scale(loss).backward()
        accum_steps += 1

        if accum_steps >= GRAD_ACCUM:
            # Unscale & Clip
            scaler.unscale_(optim)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)

            # Safety Check
            if math.isfinite(grad_norm):
                scaler.step(optim)
                scaler.update()
                
                # Zero grad (set_to_none is faster)
                optim.zero_grad(set_to_none=True)
                step += 1
                
                # Update Bar
                pbar.set_description(f"Step {step} | Loss {loss.item()*GRAD_ACCUM:.4f}")
            else:
                print(f"[!] Skipped Step {step} (GradNorm: {grad_norm})")
                scaler.update()
                optim.zero_grad(set_to_none=True)

            accum_steps = 0

            # --- Periodic Tasks ---
            if step % SAVE_EVERY == 0 and step > 0:
                s_path = Path(OUTPUT_DIR) / f"v18_step{step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "step": step
                }, s_path)
                # Keep pbar clean
                pbar.write(f"[SAVE] Saved {s_path}")

            if step % VAL_EVERY == 0 and step > last_val_step:
                vloss, vppl = quick_val(model)
                pbar.write(f"[VAL] Step {step} | Loss: {vloss:.4f} | PPL: {vppl:.1f}")
                last_val_step = step

if __name__ == "__main__":
    main()