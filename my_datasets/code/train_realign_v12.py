#!/usr/bin/env python3
"""
train_minimal_no_val.py
Minimal fast trainer for GTX 1050 Ti / local machine.
- Per-shard streaming (loads one .pt shard folder at a time)
- Immediate tqdm
- First-batch speed hack (disable gradient checkpointing for WARMUP_BATCHES)
- Time-based checkpoints every SAVE_EVERY_SECS (30 minutes default)
- Minimal logging (loss only)
"""
import os, warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------- Environment ----------------
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------- CONFIG ----------
DATA_DIR = "datasets/processed/chatml_60k_4096_stream"
SAVE_DIR = "artifacts/zia_ift_v5"
RESUME_PATH = f"{SAVE_DIR}/resume.pt"            # resume metadata + model
EXPANDED_CKPT = f"{SAVE_DIR}/zia_v5_ready.pt"    # optional load initial weights from expanded ckpt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D_MODEL = 384
N_LAYERS = 8
N_HEADS = 6
MAX_LEN = 4096
VOCAB_SIZE = 60011   # golden number you computed

BATCH_SIZE = 1
GRAD_ACC = 80
LR = 5e-4            # higher starting LR as requested
STEPS_PER_SAVE = 500
SAVE_EVERY_SECS = 30 * 60   # 30 minutes
TOTAL_EPOCHS = 1

NUM_WORKERS = 2      # adjustable; use 0 on Windows if issues
WARMUP_BATCHES = 1   # disable checkpointing for the first N batches to speed first iterations
USE_CHECKPOINTING = True  # gradient checkpointing (enabled after warmup)
# ----------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

# ---- Model ----
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        # using transformer encoder layer as lightweight block
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model*4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # shape [vocab, d_model]
        # tie after loading state dict in main()

    def forward_blocks(self, h, use_checkpoint):
        for block in self.blocks:
            if use_checkpoint:
                # checkpointing saves VRAM at cost of compute
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        return h

    def forward(self, x, use_checkpoint=True):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        h = self.forward_blocks(h, use_checkpoint)
        h = self.ln_f(h)
        return self.head(h)

# ---- Shard loader: loads ONE shard folder (input_ids.pt + labels.pt) into memory then yields rows ----
class SingleShardDataset(Dataset):
    def __init__(self, shard_folder):
        # shard_folder is a Path to folder containing input_ids.pt and labels.pt
        self.shard_folder = Path(shard_folder)
        in_path = self.shard_folder / "input_ids.pt"
        lbl_path = self.shard_folder / "labels.pt"
        assert in_path.exists() and lbl_path.exists(), f"Missing files in {shard_folder}"
        # load whole shard into CPU memory (shard size 1000 x 4096 typical)
        data = torch.load(str(in_path), map_location="cpu")
        labels = torch.load(str(lbl_path), map_location="cpu")
        assert data.shape == labels.shape
        self.data = data
        self.labels = labels
        self.n = data.shape[0]

    def __len__(self): return self.n
    def __getitem__(self, idx):
        return self.data[idx].long(), self.labels[idx].long()

# ---- Utilities for shard discovery ----
def find_shard_folders(base_dir):
    base = Path(base_dir)
    shards = sorted([p for p in base.glob("**/*") if p.is_dir() and (p / "input_ids.pt").exists()])
    return shards

# ---- Save & Resume helpers ----
def save_checkpoint(state, tag=None):
    ts = int(time.time())
    fname = f"{SAVE_DIR}/checkpoint_{tag or ts}.pt"
    torch.save(state, fname)
    # also keep a resume metadata copy
    torch.save(state, RESUME_PATH)

def load_resume():
    if not Path(RESUME_PATH).exists():
        return None
    return torch.load(RESUME_PATH, map_location="cpu")

# ---- Main training loop ----
def main():
    torch.manual_seed(42)
    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, MAX_LEN).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()

    # Resume logic (load expanded ckpt if present)
    resume_meta = load_resume()
    if resume_meta:
        # expected keys: 'model', 'optimizer', 'step', 'shard_idx', 'batch_idx', ...
        print("[resume] loading resume metadata...", end="", flush=True)
        model_state = resume_meta.get("model")
        if model_state:
            model.load_state_dict(model_state, strict=False)
        opt_state = resume_meta.get("optimizer")
        if opt_state:
            try:
                optimizer.load_state_dict(opt_state)
            except Exception:
                pass
        start_step = resume_meta.get("step", 0)
        start_shard_idx = resume_meta.get("shard_idx", 0)
        start_batch_idx = resume_meta.get("batch_idx", 0)
        last_save_time = resume_meta.get("last_save_time", time.time())
        print(" done.")
    else:
        # if there's an expanded ckpt, load it (it contains 'model')
        if Path(EXPANDED_CKPT).exists():
            ck = torch.load(EXPANDED_CKPT, map_location="cpu")
            st = ck.get("model", ck)
            model.load_state_dict(st, strict=False)
            print("[init] loaded expanded ckpt.")
        start_step = 0
        start_shard_idx = 0
        start_batch_idx = 0
        last_save_time = time.time()

    # Tie weights (head <-> tok) AFTER model loaded
    model.head.weight = model.tok.weight
    model.to(DEVICE)

    # Find shards
    shards = find_shard_folders(DATA_DIR)
    if not shards:
        print("No shard folders found in", DATA_DIR)
        return

    total_shards = len(shards)
    global_step = start_step
    step_in_epoch = 0

    use_checkpointing_flag = False  # we'll enable after warmup
    warmup_batches_done = 0

    # Immediate outer tqdm so user sees progress immediately
    outer_pbar = tqdm(range(start_shard_idx, total_shards), desc="Shards", unit="shard", dynamic_ncols=True)
    for shard_idx in outer_pbar:
        shard_path = shards[shard_idx]
        outer_pbar.set_postfix_str(f"{shard_idx+1}/{total_shards}")
        # load shard dataset
        ds = SingleShardDataset(shard_path)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # If resuming within this shard, skip batches up to start_batch_idx
        batch_start = start_batch_idx if shard_idx == start_shard_idx else 0

        inner_pbar = tqdm(enumerate(loader), desc=f"Shard {shard_idx:03d}", total=len(loader),
                          leave=False, dynamic_ncols=True)
        for bidx, (x, y) in inner_pbar:
            if bidx < batch_start:
                continue

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            # choose whether to use checkpointing for this forward
            use_checkpoint = False if warmup_batches_done < WARMUP_BATCHES else USE_CHECKPOINTING

            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                logits = model(x, use_checkpoint=use_checkpoint)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=-100)
                loss = loss / GRAD_ACC

            scaler.scale(loss).backward()

            warmup_batches_done += 1

            if (global_step + 1) % GRAD_ACC == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            global_step += 1
            step_in_epoch += 1

            # update tqdm with aggregated loss (show scaling back)
            inner_pbar.set_postfix_str(f"loss={loss.item()*GRAD_ACC:.4f}")

            # time-based save
            now = time.time()
            if (now - last_save_time) >= SAVE_EVERY_SECS:
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "shard_idx": shard_idx,
                    "batch_idx": bidx + 1,
                    "last_save_time": now,
                }
                save_checkpoint(state, tag=f"autosave_step{global_step}")
                last_save_time = now

            # step-based save
            if ((global_step // GRAD_ACC) % STEPS_PER_SAVE) == 0 and ((global_step % GRAD_ACC) == 0):
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "shard_idx": shard_idx,
                    "batch_idx": bidx + 1,
                    "last_save_time": now,
                }
                save_checkpoint(state, tag=f"step{global_step}")

        # end of shard: reset start_batch_idx so resume doesn't try to skip next shards
        start_batch_idx = 0

    # Final save
    final_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": global_step,
        "shard_idx": total_shards - 1,
        "batch_idx": 0,
        "last_save_time": time.time(),
    }
    save_checkpoint(final_state, tag="final")
    print("Training completed. Final saved.")

if __name__ == "__main__":
    main()
