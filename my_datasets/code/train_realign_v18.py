#!/usr/bin/env python3
import os, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tinygpt_encoder import TinyGPT

# ---------------- CONFIG ----------------
DATA_DIR = "datasets/processed/ift_v22"
RESUME_CKPT = "artifacts/zia_lm_v21_repair/repair_step1200.pt"
OUT_DIR = "artifacts/zia_ift_v22"

SEQ_LEN = 1024
BATCH_SIZE = 2
GRAD_ACCUM = 16       # effective batch = 32
LR = 5e-5             # VERY IMPORTANT
EPOCHS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = True
# ---------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

class IFTDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.x = data["input_ids"]
        self.y = data["labels"]

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

print("[i] Loading model...")
model = TinyGPT(
    vocab_size=60011,
    d_model=384,
    n_layers=8,
    n_heads=6,
    max_len=4096,
    dropout=0.1
).to(DEVICE)

ck = torch.load(RESUME_CKPT, map_location="cpu")
state = ck["model"] if "model" in ck else ck
model.load_state_dict(state, strict=True)
model.head.weight = model.tok.weight

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler(enabled=USE_FP16)

train_ds = IFTDataset(f"{DATA_DIR}/train.pt")
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

model.train()
step = 0

print("[i] Starting IFT v22...")

for epoch in range(EPOCHS):
    pbar = tqdm(train_dl, desc=f"epoch {epoch+1}")
    optimizer.zero_grad(set_to_none=True)

    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.amp.autocast("cuda", enabled=USE_FP16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=0
            )
            loss = loss / GRAD_ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % 200 == 0:
            torch.save(
                {"model": model.state_dict(), "step": step},
                f"{OUT_DIR}/ift_step{step}.pt"
            )

        pbar.set_postfix(loss=f"{loss.item()*GRAD_ACCUM:.4f}")
        step += 1

torch.save({"model": model.state_dict()}, f"{OUT_DIR}/ift_final.pt")
print("[✓] IFT v22 complete.")
