import os
import torch
import torch.nn as nn
import math
import time
import gc
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------- Environment ----------------
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 60011
D_MODEL = 384
LAYERS = 8
HEADS = 6
MAX_LEN = 4096  # Back to full context

# 1050 Ti Tuning
BATCH_SIZE = 1          # Must be 1 for 4096 context on 4GB VRAM
GRAD_ACCUM = 64         # 1 * 64 = Effective Batch Size 64
LR = 2.5e-4             # Slightly lowered for stability
wd = 1e-2               # Weight decay

# Paths
TRAIN_SHARD = "datasets/processed/wikitext_60k_4096/train_shard_0000"
RESUME_CKPT = "artifacts/zia_mixed_v1/checkpoint_mixed_step10703.pt"
OUTPUT_DIR = "artifacts/zia_recovery_v8"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# MODEL (Matches Sanity Check Architecture)
# ==========================================
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=HEADS, dim_feedforward=D_MODEL*4,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True
            )
            for _ in range(LAYERS)
        ])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok.weight 

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device)
        h = self.tok(x) + self.pos(pos)
        
        # Standard loop - no fancy checkpointing wrappers that break gradients
        for block in self.blocks:
            h = block(h)
            
        h = self.ln_f(h)
        return self.head(h)

# ==========================================
# DATASET
# ==========================================
class DiskDataset(Dataset):
    def __init__(self, path):
        print(f"[i] Mapping dataset from {path}...")
        self.x = torch.load(f"{path}/input_ids.pt", map_location="cpu")
        self.y = torch.load(f"{path}/labels.pt", map_location="cpu")
        self.length = len(self.x)

    def __len__(self): return self.length
    
    def __getitem__(self, i): 
        # Crop to MAX_LEN if dataset is larger, or pad if smaller (safety)
        x = self.x[i][:MAX_LEN].long()
        y = self.y[i][:MAX_LEN].long()
        return x, y

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    torch.manual_seed(1337)
    print(f"\n--- ZIA RECOVERY TRAINER v8 ---")
    print(f"Context: {MAX_LEN} | Device: {DEVICE}")
    print(f"Resume:  {RESUME_CKPT}")

    # 1. Load Model
    model = TinyGPT().to(DEVICE)
    
    # 2. Load Weights (Strict Mode)
    if os.path.exists(RESUME_CKPT):
        print("[i] Loading weights...")
        ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        
        # Fix for pos embedding size mismatch if resuming from shorter context
        curr_pos = model.pos.weight.shape[0]
        load_pos = state_dict['pos.weight'].shape[0]
        if load_pos != curr_pos:
            print(f"[!] Resizing Pos Embeddings: {load_pos} -> {curr_pos}")
            new_pos = model.pos.weight.data.clone()
            new_pos[:load_pos] = state_dict['pos.weight']
            state_dict['pos.weight'] = new_pos

        model.load_state_dict(state_dict, strict=True)
        print("[✓] Weights loaded successfully.")
    else:
        print("[!] Checkpoint not found. Starting from scratch.")

    # 3. Setup Optimizer (FRESH - discard old broken state)
    # Filter out params that shouldn't decay (biases, layernorm)
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': wd},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optim = torch.optim.AdamW(optim_groups, lr=LR, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda") # Safe AMP

    # 4. Data
    ds = DiskDataset(TRAIN_SHARD)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # 5. Training Loop
    model.train()
    print("\n[i] Starting Training Loop...")
    
    step = 0
    total_loss = 0
    optim.zero_grad()
    
    pbar = tqdm(dl, total=len(dl))
    
    for i, (x, y) in enumerate(pbar):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # --- FORWARD ---
        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, VOCAB_SIZE), 
                y.view(-1), 
                ignore_index=-100
            )
            loss = loss / GRAD_ACCUM # Scale loss
        
        # --- BACKWARD ---
        scaler.scale(loss).backward()
        
        # Accumulate loss for display
        total_loss += loss.item() * GRAD_ACCUM

        # --- UPDATE STEP ---
        if (i + 1) % GRAD_ACCUM == 0:
            # Unscale to check for infinity/NaN
            scaler.unscale_(optim)
            
            # Clip Gradients (Crucial for stability)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Step if gradients are valid
            if torch.isfinite(grad_norm):
                scaler.step(optim)
                scaler.update()
            else:
                print(f"\n[!] Skipped step {step}: GradNorm is {grad_norm}")
            
            optim.zero_grad()
            
            # Logging
            step += 1
            if step % 10 == 0:
                avg_loss = total_loss / GRAD_ACCUM
                pbar.set_description(f"Step {step} | Loss: {avg_loss:.4f} | Norm: {grad_norm:.2f}")
                # Reset tracking
                total_loss = 0
            
            # Save every 500 steps
            if step % 500 == 0:
                s_path = f"{OUTPUT_DIR}/recovery_step{step}.pt"
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'step': step,
                    'config': {'vocab': VOCAB_SIZE, 'dim': D_MODEL}
                }, s_path)
                print(f"\n[S] Saved {s_path}")

        # Garbage Collection (Help 1050 Ti)
        if i % 100 == 0:
            gc.collect()

if __name__ == "__main__":
    main()