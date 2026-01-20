import argparse, os, math, glob, time, torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import csv

# --- CONFIG (Defaults) ---
CKPT_DIR = "artifacts/zia_ift_v5"
VAL_DIR = "datasets/processed/chatml_60k_4096_stream"
VOCAB_SIZE = 60011  # The Golden Number
D_MODEL = 384
N_LAYERS = 8
N_HEADS = 6
MAX_LEN = 4096
# -------------------------

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, activation="gelu", batch_first=True, norm_first=True) 
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        B,T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        for b in self.blocks: h = b(h)
        h = self.ln_f(h)
        return self.head(h)

class ShardFolderDataset(Dataset):
    def __init__(self, shard_folder):
        p = Path(shard_folder)
        self.inp = torch.load(p / "input_ids.pt", map_location="cpu")
        self.lbl = torch.load(p / "labels.pt", map_location="cpu")
    def __len__(self): return self.inp.shape[0]
    def __getitem__(self, i): return self.inp[i].long(), self.lbl[i].long()

def find_val_shards(val_root):
    p = Path(val_root)
    # Prefer 'val_shard_*', fallback to 'shard_*' if not found
    shards = sorted(list(p.glob("val_shard_*")))
    if not shards:
        shards = sorted(list(p.glob("shard_*")))
    # Filter only those with input_ids.pt
    shards = [s for s in shards if (s / "input_ids.pt").exists()]
    return shards

def evaluate_checkpoint(ckpt_path, model, device, shards, max_batches):
    print(f"\n[i] Loading {os.path.basename(ckpt_path)}...")
    try:
        ck = torch.load(ckpt_path, map_location=device)
        state = ck.get("model", ck)
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return None

    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        batches_done = 0
        for shard in shards:
            ds = ShardFolderDataset(shard)
            loader = DataLoader(ds, batch_size=1, num_workers=0) # Batch 1 safety
            
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), reduction='sum', ignore_index=-100)
                
                valid_toks = (y.view(-1) != -100).sum().item()
                total_loss += loss.item()
                total_tokens += valid_toks
                
                batches_done += 1
                if batches_done >= max_batches: break
            if batches_done >= max_batches: break
            
    if total_tokens == 0: return float('inf')
    
    return total_loss / total_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default=CKPT_DIR)
    parser.add_argument("--val-dir", default=VAL_DIR)
    parser.add_argument("--out", default="artifacts/zia_ift_v5/eval_results.csv")
    parser.add_argument("--max-batches", type=int, default=50) # Speed limit
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[i] Device: {device}")
    
    # 1. Model
    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, MAX_LEN).to(device)
    
    # 2. Data
    shards = find_val_shards(args.val_dir)
    if not shards:
        # Fallback to train shards if no val found (just to verify code works)
        print("⚠️ No val shards found. Looking for train shards...")
        shards = sorted(list(Path(args.val_dir).glob("train_shard_*")))[:1] # Use just 1
        
    print(f"[i] Found {len(shards)} shards for evaluation.")
    
    # 3. Find Checkpoints
    ckpts = sorted(list(Path(args.ckpt_dir).glob("checkpoint_*.pt")), key=os.path.getmtime)
    print(f"[i] Found {len(ckpts)} checkpoints to evaluate.")
    
    results = []
    
    # 4. Loop
    for ckpt in ckpts:
        loss = evaluate_checkpoint(ckpt, model, device, shards, args.max_batches)
        if loss:
            ppl = math.exp(loss) if loss < 20 else float('inf')
            print(f"   -> Loss: {loss:.4f} | PPL: {ppl:.2f}")
            results.append({
                "checkpoint": os.path.basename(ckpt),
                "loss": loss,
                "ppl": ppl
            })
            
    # 5. Save
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "loss", "ppl"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n✅ Results saved to {args.out}")

if __name__ == "__main__":
    main()