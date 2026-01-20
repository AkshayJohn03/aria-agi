#!/usr/bin/env python3
# train_realign_v20.py — ZIA full-training v20 (multi-shard streaming, FP16, time-based save/val)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os, time, math, gc
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
# ---------------- Environment ----------------
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------- CONFIG ----------------
# Paths
TRAIN_DIR = "datasets/processed/wikitext_60k_4096"
VAL_SHARD = "datasets/processed/wikitext_60k_4096/val_shard_0000"
RESUME_CKPT = "artifacts/zia_lm_v18/step6.pt"
OUTPUT_DIR = "artifacts/zia_lm_v20"
RESUME_META = Path(OUTPUT_DIR) / "resume_v20.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model / training hyperparams
SEQ_LEN = 4096
VOCAB_SIZE = 60011
D_MODEL = 384
LAYERS = 8
HEADS = 6

# Resource tuning (you can keep BATCH_SIZE=2 if it's stable)
BATCH_SIZE = 1
GRAD_ACCUM = 80
LR = 5e-5
WEIGHT_DECAY = 1e-2
CLIP_NORM = 0.5

# Time-based tasks (seconds)
SAVE_EVERY_SECS = 1800    # 30 minutes
VAL_EVERY_SECS = 3600     # 1 hour

USE_FP16 = True
NUM_WORKERS = 2           # DataLoader workers (CPU prefetch)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

# ----------------------------------------

# -------- Model --------
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=HEADS, dim_feedforward=D_MODEL * 4,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True
            ) for _ in range(LAYERS)
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

# -------- Dataset (single-shard-in-ram) --------
class ShardDataset(Dataset):
    def __init__(self, shard_path: Path):
        # load shard into CPU memory
        self.x = torch.load(shard_path / "input_ids.pt", map_location="cpu")
        self.y = torch.load(shard_path / "labels.pt", map_location="cpu")
        assert self.x.shape[1] >= 1 and self.y.shape[1] >= 1

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx][:SEQ_LEN].long(), self.y[idx][:SEQ_LEN].long()

# -------- Validation (quick small shard) --------
@torch.no_grad()
def quick_val(model, samples=8):
    p = Path(VAL_SHARD)
    if not p.exists():
        return None
    x = torch.load(p / "input_ids.pt", map_location="cpu")[:samples, :SEQ_LEN].long().to(DEVICE)
    y = torch.load(p / "labels.pt", map_location="cpu")[:samples, :SEQ_LEN].long().to(DEVICE)
    with torch.amp.autocast(enabled=USE_FP16, device_type="cuda" if DEVICE=="cuda" else "cpu"):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=-100)
    return float(loss.item()), float(math.exp(min(loss.item(), 50)))

# -------- Utilities: save / load resume meta --------
def save_checkpoint(model, optim, step, shard_idx, sample_idx):
    ts = int(time.time())
    path = Path(OUTPUT_DIR) / f"v20_step{step}_t{ts}.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "step": step,
        "shard_idx": shard_idx,
        "sample_idx": sample_idx,
        "ts": ts
    }, path)
    # save resume metadata separately (small)
    meta = {"ckpt": str(path), "step": step, "shard_idx": shard_idx, "sample_idx": sample_idx, "ts": ts}
    with open(RESUME_META, "w") as f:
        json.dump(meta, f)
    return path

def load_resume_meta():
    if not RESUME_META.exists():
        return None
    try:
        return json.loads(RESUME_META.read_text())
    except Exception:
        return None

# -------- MAIN TRAINING (shard streaming) --------
def main():
    torch.manual_seed(1337)

    # build model + optimizer + scaler
    model = TinyGPT().to(DEVICE)
    model.head.weight = model.tok.weight

    # strict resume from base checkpoint (no pos resizing)
    if Path(RESUME_CKPT).exists():
        ck = torch.load(RESUME_CKPT, map_location=DEVICE)
        st = ck.get("model", ck)
        if st["pos.weight"].shape[0] != SEQ_LEN:
            raise RuntimeError(f"Checkpoint pos length {st['pos.weight'].shape[0]} != SEQ_LEN {SEQ_LEN}. Aborting (no resizing).")
        model.load_state_dict(st, strict=True)

    # optimizer (fresh)
    params = list(model.parameters())
    optim = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(enabled=USE_FP16)

    # shards list
    shard_paths = sorted(Path(TRAIN_DIR).glob("train_shard_*"))
    if not shard_paths:
        # fallback to single folder
        shard_paths = [Path(TRAIN_DIR)]

    # resume metadata (for streaming)
    resume_meta = load_resume_meta()
    start_shard_idx = 0
    start_sample_idx = 0
    global_step = 0
    if resume_meta:
        start_shard_idx = int(resume_meta.get("shard_idx", 0))
        start_sample_idx = int(resume_meta.get("sample_idx", 0))
        global_step = int(resume_meta.get("step", 0))
        print(f"[i] Resume metadata found -> shard {start_shard_idx}, sample {start_sample_idx}, step {global_step}")

    last_save = time.time()
    last_val = time.time()

    model.train()
    # training loop over epochs — we'll loop shards repeatedly until interrupted
    epoch = 0
    try:
        while True:
            epoch += 1
            # iterate shards deterministically
            for s_idx, shard in enumerate(shard_paths):
                if s_idx < start_shard_idx:
                    continue  # skip shards already done per resume
                # load shard into RAM
                ds = ShardDataset(shard)
                dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                pin_memory=True, persistent_workers=False)
                pbar = tqdm(enumerate(dl), total=len(dl), desc=f"Epoch{epoch} Shard{s_idx}:{shard.name}", ncols=120)
                # if resuming inside this shard, set a local counter to skip until start_sample_idx
                local_idx = 0
                for batch_idx, batch in pbar:
                    # skip until the resume sample if necessary
                    if s_idx == start_shard_idx and local_idx < start_sample_idx:
                        local_idx += 1
                        continue

                    xb, yb = batch
                    xb = xb.to(DEVICE, non_blocking=True)
                    yb = yb.to(DEVICE, non_blocking=True)

                    with torch.amp.autocast(enabled=USE_FP16, device_type="cuda" if DEVICE=="cuda" else "cpu"):
                        logits = model(xb)
                        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1), ignore_index=-100)
                        loss = loss / GRAD_ACCUM

                    scaler.scale(loss).backward()
                    local_idx += 1

                    # accumulate
                    if (local_idx % GRAD_ACCUM) == 0:
                        scaler.unscale_(optim)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                        if not math.isfinite(grad_norm):
                            # skip step but update scaler to avoid deadlock
                            scaler.update()
                            optim.zero_grad(set_to_none=True)
                        else:
                            scaler.step(optim)
                            scaler.update()
                            optim.zero_grad(set_to_none=True)
                            global_step += 1
                            pbar.set_postfix_str(f"step={global_step} loss={loss.item()*GRAD_ACCUM:.4f} gn={grad_norm:.3f}")

                    # time-based save
                    now = time.time()
                    if now - last_save >= SAVE_EVERY_SECS:
                        ckpt_path = save_checkpoint(model, optim, global_step, s_idx, local_idx)
                        pbar.write(f"[SAVE] {ckpt_path}")
                        last_save = now

                    # time-based validation
                    if now - last_val >= VAL_EVERY_SECS:
                        v = quick_val(model)
                        if v:
                            vloss, vppl = v
                            pbar.write(f"[VAL] step={global_step} loss={vloss:.4f} ppl={vppl:.1f}")
                        last_val = now

                # done with shard: free memory then continue
                del dl, ds
                gc.collect()
                torch.cuda.empty_cache()
                # after first resume, start_shard_idx only applies once
                start_sample_idx = 0
                start_shard_idx = 0

            # finished one full pass over all shards -> continue to next epoch
    except KeyboardInterrupt:
        # save small resume metadata (no heavy prints)
        meta = {"step": global_step, "shard_idx": s_idx, "sample_idx": local_idx}
        with open(RESUME_META, "w") as f:
            json.dump(meta, f)
        # also save a lightweight checkpoint
        torch.save({"model": model.state_dict(), "optimizer": optim.state_dict(), "step": global_step},
                   Path(OUTPUT_DIR) / f"interrupt_step{global_step}.pt")
        print(f"\n[i] Interrupted. Saved resume meta and checkpoint at step {global_step}.")

if __name__ == "__main__":
    main()
