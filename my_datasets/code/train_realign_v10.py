#!/usr/bin/env python3
"""
train_realign_v10.py (Rewritten)

Fast, hardcoded trainer for ZIA Realign Phase.
- Tokenizer: SentencePiece (24k)
- Dataset: Sharded .pt files (datasets/processed/test_openorca/train)
- Context: Fixed 4096
- No validation, no CLI args, minimal print.
"""

import os
import glob
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from tqdm import tqdm

# ---------------- Performance Setup ----------------
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

# Disable HF cache conflicts
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.setdefault("HF_HOME", "artifacts/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "artifacts/hf_datasets_cache")

# ---------------- Configuration (Hardcoded) ----------------
TRAIN_DIR = "datasets/processed/test_openorca/train"
RESUME_CKPT = "artifacts/zia_ift_fixed4k/checkpoints/checkpoint_step8163.pt"
SPM_MODEL = "artifacts/zia_tokenizer_v2_clean/zia_spm.model"
CHECKPOINT_DIR = "artifacts/zia_ift_fixed4k/checkpoints"

# Model Hyperparams
D_MODEL = 384
N_LAYERS = 8
N_HEADS = 6
MLP_RATIO = 4
DROPOUT = 0.1
CONTEXT_LEN = 4096  # Fixed

# Training Hyperparams
BATCH_SIZE = 4
GRAD_ACCUM = 80
NUM_WORKERS = 4
LR = 2.5e-4
AUTOSAVE_MINUTES = 30
MAX_SHARDS = 9999  # Train on all available shards
SEED = 42

# ---------------- Utilities ----------------
def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def smart_load(model, ckpt_path):
    """Load weights, ignoring shape mismatches (e.g. embedding size changes)."""
    if not os.path.exists(ckpt_path):
        print(f"[!] Checkpoint not found: {ckpt_path}")
        return 0, 0, 0

    print(f"[i] Loading weights from {ckpt_path}...")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Handle both full checkpoint dict and model-only dict
    state_dict = ck.get("model", ck)
    
    model_dict = model.state_dict()
    load_dict = {}
    mismatch_count = 0
    
    for k, v in state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                load_dict[k] = v
            else:
                mismatch_count += 1
                # print(f"    [!] Shape mismatch for {k}: {v.shape} vs {model_dict[k].shape} (skipped)")
        else:
            pass # print(f"    [!] Key {k} not in model (skipped)")

    model.load_state_dict(load_dict, strict=False)
    print(f"[i] Loaded {len(load_dict)} layers. Skipped {mismatch_count} mismatches.")
    
    return ck.get("step", 0), ck.get("current_shard_idx", 0), ck.get("current_batch_idx", 0)

# ---------------- Model ----------------
class FeedForward(nn.Module):
    def __init__(self, d, m, drop):
        super().__init__()
        hidden = d * m
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
            nn.Dropout(drop)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, d, h, m, drop):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ff = FeedForward(d, m, drop)
    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         attn_mask=mask, need_weights=False)
        return x + a + self.ff(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self, vocab, d, l, h, m, max_len, drop, pad):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, h, m, drop) for _ in range(l)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight # Weight tying
        self.max_len = max_len
        self.pad_token_id = pad

    def forward(self, ids, lbls=None):
        T = ids.size(1)
        # Truncate if needed (though dataset should be fixed)
        if T > self.max_len: 
            ids = ids[:, :self.max_len]
            if lbls is not None: lbls = lbls[:, :self.max_len]
            T = self.max_len
            
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device).unsqueeze(0))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if lbls is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                lbls[:, 1:T].reshape(-1),
                ignore_index=-100
            )
        return logits, loss

# ---------------- Data ----------------
class ShardDataset(Dataset):
    def __init__(self, shard_path):
        # Expects input_ids.pt and labels.pt in the shard folder
        self.x = torch.load(os.path.join(shard_path, "input_ids.pt"), map_location="cpu", weights_only=False)
        self.y = torch.load(os.path.join(shard_path, "labels.pt"), map_location="cpu", weights_only=False)
        
        # Ensure LongTensor
        if not isinstance(self.x, torch.Tensor): self.x = torch.tensor(self.x, dtype=torch.long)
        if not isinstance(self.y, torch.Tensor): self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

# ---------------- Main ----------------
def train():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[i] Device: {device}")

    # 1. Tokenizer
    if not os.path.exists(SPM_MODEL):
        raise FileNotFoundError(f"Tokenizer model not found: {SPM_MODEL}")
    sp = spm.SentencePieceProcessor()
    sp.load(SPM_MODEL)
    vocab_size = sp.get_piece_size()
    pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") != sp.unk_id() else 0
    print(f"[i] Vocab: {vocab_size} | Pad ID: {pad_id}")

    # 2. Model
    model = TinyGPT(vocab_size, D_MODEL, N_LAYERS, N_HEADS, MLP_RATIO, CONTEXT_LEN, DROPOUT, pad_id).to(device)
    
    # 3. Load Weights
    start_step, resume_shard_idx, resume_batch_idx = smart_load(model, RESUME_CKPT)
    
    # 4. Optimizer & Scheduler (Always reset on resume for this script)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(enabled=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=4000, eta_min=1e-6)
    
    # 5. Shards
    all_shards = sorted(glob.glob(os.path.join(TRAIN_DIR, "shard_*")))
    if not all_shards:
        raise RuntimeError(f"No shards found in {TRAIN_DIR}")
    
    # Filter/Select shards
    # For now, just take all or slice
    selected_shards = all_shards   # train over all shards sequentially #selected_shards = all_shards[:MAX_SHARDS] 
    print(f"[i] Found {len(all_shards)} shards. Using {len(selected_shards)}.")

    step = start_step
    last_save = time.time()
    model.train()

    # 6. Training Loop
    for shard_idx, shard_path in enumerate(selected_shards):
        # Resume logic: skip shards we've already finished
        if shard_idx < resume_shard_idx:
            continue
            
        # Load shard
        try:
            ds = ShardDataset(shard_path)
        except Exception as e:
            print(f"[!] Failed to load {shard_path}: {e}")
            continue

        dl = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2
        )
        
        pbar = tqdm(dl, desc=f"Shard {shard_idx+1}/{len(selected_shards)}", dynamic_ncols=True)
        
        for batch_idx, (ids, lbls) in enumerate(pbar):
            # Resume logic: skip batches within the current shard
            if shard_idx == resume_shard_idx and batch_idx < resume_batch_idx:
                continue

            ids, lbls = ids.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            
            with torch.amp.autocast("cuda", enabled=True):
                _, loss = model(ids, lbls)
                loss = loss / GRAD_ACCUM
            
            scaler.scale(loss).backward()
            
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
                
                # Update logs
                lr_curr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{loss.item()*GRAD_ACCUM:.4f}", "lr": f"{lr_curr:.2e}", "step": step})
            
            step += 1
            
            # Autosave
            if time.time() - last_save > (AUTOSAVE_MINUTES * 60):
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                save_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step{step}.pt")
                atomic_save({
                    "model": model.state_dict(),
                    "step": step,
                    "current_shard_idx": shard_idx,
                    "current_batch_idx": batch_idx
                }, save_path)
                last_save = time.time()
                tqdm.write(f"[💾] Saved checkpoint: {save_path}")

    # Final Save
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    final_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_final_step{step}.pt")
    atomic_save({"model": model.state_dict(), "step": step}, final_path)
    print(f"[✓] Done. Final checkpoint: {final_path}")

if __name__ == "__main__":
    train()
