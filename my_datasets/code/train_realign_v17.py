#!/usr/bin/env python3
# train_repair_v21.py — Encoder-safe semantic stabilization

import os, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tinygpt_encoder import TinyGPT

# ---------------- CONFIG ----------------
DATA_DIR = "datasets/processed/wiki_clean_1024"
RESUME_CKPT = "artifacts/zia_lm_v20/v20_step977_t1765969109.pt"
OUT_DIR = "artifacts/zia_lm_v21_repair"

SEQ_LEN = 1024
VOCAB_SIZE = 60011
D_MODEL = 384
LAYERS = 8
HEADS = 6

BATCH_SIZE = 4        # safe at 1024 ctx
GRAD_ACCUM = 16       # effective batch = 64
LR = 1e-4             # repair LR (do not increase)
MAX_STEPS = 1500      # enough for stabilization

USE_FP16 = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

class ShardDataset(Dataset):
    def __init__(self, path):
        self.x = torch.load(os.path.join(path, "input_ids.pt"))
        self.y = torch.load(os.path.join(path, "labels.pt"))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i][:SEQ_LEN], self.y[i][:SEQ_LEN]

print("[i] Loading encoder model...")
model = TinyGPT(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_layers=LAYERS,
    n_heads=HEADS,
    max_len=4096,   # keep full capacity
    dropout=0.1
).to(DEVICE)

ck = torch.load(RESUME_CKPT, map_location="cpu")
state = ck["model"] if "model" in ck else ck
model.load_state_dict(state, strict=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scaler = torch.amp.GradScaler(enabled=USE_FP16)

model.train()
step = 0

shards = sorted(os.listdir(DATA_DIR))
print(f"[i] Stabilizing on {len(shards)} shards (using first 8–10 only)")

for shard in shards[:10]:
    ds = ShardDataset(os.path.join(DATA_DIR, shard))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    pbar = tqdm(dl, desc=f"repair:{shard}")
    for x, y in pbar:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.amp.autocast("cuda", enabled=USE_FP16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                y.view(-1),
                ignore_index=0
            )
            loss = loss / GRAD_ACCUM

        scaler.scale(loss).backward()

        if step % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % 300 == 0:
            torch.save(
                {"model": model.state_dict(), "step": step},
                f"{OUT_DIR}/repair_step{step}.pt"
            )

        pbar.set_postfix(loss=f"{loss.item()*GRAD_ACCUM:.4f}")
        step += 1

        if step >= MAX_STEPS:
            break

    if step >= MAX_STEPS:
        break

torch.save(
    {"model": model.state_dict(), "step": step},
    f"{OUT_DIR}/repair_final.pt"
)
print("[✓] v21 semantic stabilization complete.")
