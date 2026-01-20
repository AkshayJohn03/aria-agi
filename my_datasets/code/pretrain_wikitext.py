import os, time, gc, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------- Environment ----------------
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------- CONFIG ----------
WIKI_DIR = "datasets/processed/wikitext_60k_4096"
IFT_DIR  = "datasets/processed/chatml_60k_4096_stream"
SAVE_DIR = "artifacts/zia_mixed_v1"
RESUME_META = f"{SAVE_DIR}/resume_mixed.pt"

# Load Healthy Brain if starting fresh
ALIGNED_START = "artifacts/zia_ift_v5/checkpoint_pretrain_step1404_1764739461.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 60011
D_MODEL = 384
N_LAYERS = 8
N_HEADS = 6
MAX_LEN = 4096

BATCH_SIZE = 1
GRAD_ACC = 16 
LR = 2e-4
SAVE_EVERY_SECS = 1800
NUM_WORKERS = 2       
USE_CHECKPOINTING = True
# -----------------------------

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(42)
random.seed(42)

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, 0.1, "gelu", batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device))
        for block in self.blocks:
            if self.training:
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        return self.head(self.ln_f(h))

class ShardFolderDataset(Dataset):
    def __init__(self, folder):
        p = Path(folder)
        self.inp = torch.load(p / "input_ids.pt", map_location="cpu", weights_only=True)
        self.lbl = torch.load(p / "labels.pt", map_location="cpu", weights_only=True)
    def __len__(self): return self.inp.shape[0]
    def __getitem__(self, i): return self.inp[i].long(), self.lbl[i].long()

def save_resume(state):
    torch.save(state, RESUME_META)
    ts = int(time.time())
    path = f"{SAVE_DIR}/checkpoint_mixed_step{state['step']}.pt"
    torch.save(state, path)
    print(f"\n💾 Saved: {path}")

def main():
    print(f"[i] Device: {DEVICE}")
    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, MAX_LEN).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()

    # Resume State
    global_step = 0
    total_shards_done = 0
    
    if Path(RESUME_META).exists():
        print(f"[i] Resuming from {RESUME_META}...")
        r = torch.load(RESUME_META, map_location=DEVICE)
        model.load_state_dict(r["model"])
        optimizer.load_state_dict(r["optimizer"])
        scaler.load_state_dict(r["scaler"])
        global_step = r["step"]
        total_shards_done = r["shards_done"]
    elif Path(ALIGNED_START).exists():
        print(f"[i] Loading Healthy Brain: {ALIGNED_START}")
        ck = torch.load(ALIGNED_START, map_location=DEVICE)
        # Handle dict nesting
        st = ck.get("model", ck)
        model.load_state_dict(st, strict=False)
    
    model.head.weight = model.tok.weight # Tie weights
    model.train()

    # Build Mixed Shard List (Deterministic Order)
    wiki_shards = sorted(list(Path(WIKI_DIR).glob("train_shard_*")))
    ift_shards = sorted(list(Path(IFT_DIR).glob("train_shard_*")))
    
    print(f"[i] Found {len(wiki_shards)} Wiki shards, {len(ift_shards)} IFT shards.")
    
    # Create an interleaved list: [Wiki, Wiki, Wiki, IFT, Wiki...] (3:1 Ratio)
    # This ensures deterministic order for resuming.
    mixed_shards = []
    w_idx, i_idx = 0, 0
    
    while w_idx < len(wiki_shards) or i_idx < len(ift_shards):
        # Add 3 Wiki shards
        for _ in range(3):
            if w_idx < len(wiki_shards):
                mixed_shards.append((wiki_shards[w_idx], "LM"))
                w_idx += 1
        # Add 1 IFT shard
        if i_idx < len(ift_shards):
            mixed_shards.append((ift_shards[i_idx], "IFT"))
            i_idx += 1
            
    print(f"[i] Total Mixed Queue: {len(mixed_shards)} shards.")
    
    # Fast Forward
    if total_shards_done > 0:
        print(f"[i] Skipping {total_shards_done} shards...")
        mixed_shards = mixed_shards[total_shards_done:]

    last_save = time.time()
    
    # Training Loop
    for shard_path, shard_type in mixed_shards:
        # print(f"\n⚡ Processing {shard_type}: {shard_path.name}")
        
        ds = ShardFolderDataset(shard_path)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        
        pbar = tqdm(loader, desc=f"{shard_type} Shard {total_shards_done}", leave=False)
        
        for x, y in pbar:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=-100)
                loss = loss / GRAD_ACC

            scaler.scale(loss).backward()
            
            if (global_step + 1) % GRAD_ACC == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update bar
                pbar.set_postfix(loss=loss.item() * GRAD_ACC, type=shard_type)
                
                # Time-based Save
                if time.time() - last_save > SAVE_EVERY_SECS:
                    save_resume({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "step": global_step,
                        "shards_done": total_shards_done
                    })
                    last_save = time.time()

            global_step += 1
        
        # End of Shard
        total_shards_done += 1
        
        # Cleanup
        del ds, loader
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()