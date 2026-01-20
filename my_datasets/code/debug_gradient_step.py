#!/usr/bin/env python3
# debug_gradient_step_v2.py
import torch, math, gc
import torch.nn as nn
from pathlib import Path

# --------- CONFIG ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "artifacts/zia_mixed_v1/checkpoint_mixed_step10703.pt"
TRAIN_SHARD = "datasets/processed/wikitext_60k_4096/train_shard_0000"
BATCHES = 8               # how many batches to read
ACCUM_STEPS = 4           # simulate accumulation
BATCH_SIZE = 1
VOCAB = 60011
D_MODEL = 384
LAYERS = 8
HEADS = 6
MAX_LEN = 4096

# --------- Model ----------
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=HEADS, dim_feedforward=D_MODEL*4,
                                      dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.tok.weight # tie weights

    def forward(self, x):
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.head(h)

# --------- Helpers ----------
def print_param_stats(model):
    print("\n[i] Scanning Parameters for NaN/Inf/Explosion...")
    tot = 0
    for n,p in model.named_parameters():
        if p is None: continue
        if p.numel() == 0: continue
        
        # Check integrity
        if torch.isnan(p).any():
            print(f"!!! FAIL: Found NaN in parameter: {n}")
        if torch.isinf(p).any():
            print(f"!!! FAIL: Found Inf in parameter: {n}")
            
        mx = float(p.detach().abs().max().cpu().item())
        mn = float(p.detach().abs().min().cpu().item())
        
        # Convert shape tuple to string for formatting
        shape_str = str(tuple(p.shape))
        print(f"P {n:40s} shape={shape_str:20s} | max={mx:.3e} min={mn:.3e}")
        tot += p.numel()
    print(f"Total params: {tot}\n")

def print_grad_stats(model):
    print("---- GRAD STATS PER LAYER ----")
    total_norm = 0.0
    found = False
    
    # Sort to keep order clean
    for n,p in model.named_parameters():
        if p.grad is None:
            continue
        found = True
        
        # Check bad gradients
        if torch.isnan(p.grad).any():
            print(f"!!! NAN GRADIENT DETECTED in {n}")
        if torch.isinf(p.grad).any():
            print(f"!!! INF GRADIENT DETECTED in {n}")

        gnorm = float(p.grad.detach().norm().cpu().item())
        total_norm += gnorm**2
        print(f"G {n:40s} | grad_norm={gnorm:.6e}")
        
    if not found:
        print("NO gradients found.")
        return 0.0
        
    total_norm = math.sqrt(total_norm)
    print(f"==> TOTAL GRAD NORM: {total_norm:.6f}")
    return total_norm

# --------- Main ----------
def main():
    torch.manual_seed(42)
    print(f"[i] Device: {DEVICE}")
    
    # 1. Init Model
    model = TinyGPT().to(DEVICE)
    
    # 2. Load Checkpoint
    print(f"[i] Loading {CKPT}...")
    try:
        ck = torch.load(CKPT, map_location="cpu")
        st = ck.get("model", ck)
    except Exception as e:
        print(f"[!] Failed to load checkpoint: {e}")
        return

    # Handle Positional Embedding Resizing
    if "pos.weight" in st:
        curr_pos = model.pos.weight.shape[0]
        load_pos = st["pos.weight"].shape[0]
        if load_pos != curr_pos:
            print(f"[i] Resizing pos.weight: ckpt({load_pos}) -> model({curr_pos})")
            new_pos = model.pos.weight.data.clone()
            loaded_w = st["pos.weight"]
            # Copy what fits
            min_len = min(load_pos, curr_pos)
            new_pos[:min_len] = loaded_w[:min_len]
            # If expanding, pad with last known embedding
            if curr_pos > load_pos:
                new_pos[load_pos:] = loaded_w[-1]
            st["pos.weight"] = new_pos

    model.load_state_dict(st, strict=True)
    model.train()
    
    # 3. Setup Optimizer (Lower LR for safety check)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9,0.95), weight_decay=1e-2)
    print("[i] Model loaded. LR set to 5e-5 for test.")
    
    # Check initial weights health
    print_param_stats(model)

    # 4. Load Data
    print(f"[i] Loading data from {TRAIN_SHARD}...")
    ds_x = torch.load(f"{TRAIN_SHARD}/input_ids.pt", map_location="cpu")[:BATCHES*BATCH_SIZE]
    ds_y = torch.load(f"{TRAIN_SHARD}/labels.pt", map_location="cpu")[:BATCHES*BATCH_SIZE]

    # 5. Run Manual Step
    print("[i] Running Forward/Backward pass (FP32)...")
    accum = 0
    step_count = 0
    
    for i in range(BATCHES):
        x = ds_x[i:i+1][:, :MAX_LEN].long().to(DEVICE)
        y = ds_y[i:i+1][:, :MAX_LEN].long().to(DEVICE)
        
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        
        # Scale loss manually for accumulation
        (loss / ACCUM_STEPS).backward()
        
        accum += 1
        print(f"   Batch {i}: loss={loss.item():.4f}")
        
        if accum == ACCUM_STEPS:
            print("\n[i] Accumulation complete. Inspecting Gradients...")
            
            # INSPECT BEFORE CLIP
            gnorm = print_grad_stats(model)
            
            # CLIP
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            # UPDATE
            optim.step()
            optim.zero_grad()
            
            accum = 0
            step_count += 1
            print(f"[✓] Step {step_count} completed.\n")
            gc.collect()

    print("[✓] Debug run finished successfully.")

if __name__ == "__main__":
    main()