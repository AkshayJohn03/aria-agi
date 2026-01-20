#!/usr/bin/env python3
"""
Minimal per-shard trainer for aligned model + clean tokenizer.

Features:
 - loads one shard folder at a time (shard = folder with input_ids.pt + labels.pt)
 - immediate tqdm for shards and inner batches
 - gradient accumulation (GRAD_ACC)
 - time-based autosave every SAVE_EVERY_SECS (30 minutes)
 - resume via resume.pt (resume metadata + model)
 - minimal stdout: only tqdm bars and occasional autosave message
"""
import os, time, gc
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ========== CONFIG ==========
DATA_DIR = "datasets/processed/chatml_60k_4096_stream"
SAVE_DIR = "artifacts/zia_ift_v5"
EXPANDED_CKPT = f"{SAVE_DIR}/zia_v5_aligned.pt"   # optional starting point
RESUME_META = f"{SAVE_DIR}/resume.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB_SIZE = 60011    # golden number
D_MODEL = 384
N_LAYERS = 8
N_HEADS = 6
MAX_LEN = 4096

BATCH_SIZE = 1        # as requested
GRAD_ACC = 80
LR = 5e-4             # higher start LR for fast adaptation
SAVE_EVERY_SECS = 30 * 60
TOTAL_EPOCHS = 1

NUM_WORKERS = 0      # set 0 if Windows spawn issues
WARMUP_BATCHES = 1    # disable checkpointing for first N batches
USE_CHECKPOINTING = True
# ===========================

os.makedirs(SAVE_DIR, exist_ok=True)

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                       dim_feedforward=d_model*4,
                                       dropout=0.1, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_blocks(self, h, use_checkpoint):
        for block in self.blocks:
            if use_checkpoint:
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        return h

    def forward(self, x, use_checkpoint=True):
        B,T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        h = self.forward_blocks(h, use_checkpoint)
        h = self.ln_f(h)
        return self.head(h)

# Shard dataset (loads a shard folder into CPU memory)
class ShardFolderDataset(Dataset):
    def __init__(self, shard_path):
        p = Path(shard_path)
        self.inp = torch.load(p / "input_ids.pt", map_location="cpu")
        self.lbl = torch.load(p / "labels.pt", map_location="cpu")
        assert self.inp.shape == self.lbl.shape
    def __len__(self):
        return self.inp.shape[0]
    def __getitem__(self, i):
        return self.inp[i].long(), self.lbl[i].long()

def find_shards(base_dir):
    p = Path(base_dir)
    shards = sorted([d for d in p.glob("**/*") if d.is_dir() and (d / "input_ids.pt").exists()])
    return shards

def save_resume(state):
    torch.save(state, RESUME_META)
    # and a timestamped backup
    ts = int(time.time())
    torch.save(state, f"{SAVE_DIR}/checkpoint_autosave_step{state.get('step',ts)}.pt")

def main():
    torch.manual_seed(42)
    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, MAX_LEN).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()

    # resume
    start_shard = 0
    start_batch = 0
    global_step = 0
    last_save = time.time()

    if Path(RESUME_META).exists():
        r = torch.load(RESUME_META, map_location="cpu")
        st = r.get("model")
        if st:
            model.load_state_dict(st, strict=False)
        opt = r.get("optimizer")
        if opt:
            try:
                optimizer.load_state_dict(opt)
            except Exception:
                pass
        start_shard = r.get("shard_idx", 0)
        start_batch = r.get("batch_idx", 0)
        global_step = r.get("step", 0)
        last_save = r.get("last_save_time", time.time())
    else:
        # optionally load expanded aligned checkpoint (weights-wise)
        if Path(EXPANDED_CKPT).exists():
            ck = torch.load(EXPANDED_CKPT, map_location="cpu")
            st = ck.get("model", ck)
            model.load_state_dict(st, strict=False)

    # tie weights
    model.head.weight = model.tok.weight
    model.to(DEVICE)

    shards = find_shards(DATA_DIR)
    total_shards = len(shards)
    if total_shards == 0:
        return

    warmup_done = 0
    # outer loop: shard-level tqdm
    for shard_idx in tqdm(range(start_shard, total_shards), desc="Shards", unit="shard", dynamic_ncols=True):
        shard = shards[shard_idx]
        ds = ShardFolderDataset(shard)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        # inner tqdm: batches of this shard
        inner_iter = enumerate(loader)
        inner_pbar = tqdm(inner_iter, desc=f"Shard {shard_idx:03d}", total=len(loader), leave=False, dynamic_ncols=True)
        for bidx, (x,y) in inner_pbar:
            if shard_idx == start_shard and bidx < start_batch:
                continue

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            use_ckpt = False if warmup_done < WARMUP_BATCHES else USE_CHECKPOINTING

            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                logits = model(x, use_checkpoint=use_ckpt)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=-100)
                loss = loss / GRAD_ACC

            scaler.scale(loss).backward()
            warmup_done += 1

            if ((global_step + 1) % GRAD_ACC) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            global_step += 1
            inner_pbar.set_postfix_str(f"loss={loss.item()*GRAD_ACC:.4f}")

            # time-based autosave
            now = time.time()
            if (now - last_save) >= SAVE_EVERY_SECS:
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "shard_idx": shard_idx,
                    "batch_idx": bidx + 1,
                    "last_save_time": now
                }
                save_resume(state)
                last_save = now

            # lightweight GC occasionally
            if (global_step % 1000) == 0:
                gc.collect()

        # end shard: reset start_batch for resume path
        start_batch = 0

    # final save
    final_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": global_step,
        "shard_idx": total_shards - 1,
        "batch_idx": 0,
        "last_save_time": time.time()
    }
    save_resume(final_state)

if __name__ == "__main__":
    main()
