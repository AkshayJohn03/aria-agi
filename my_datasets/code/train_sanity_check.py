import os
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# ==========================================
# CONFIGURATION (The "Safe Mode" Setup)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 60011
D_MODEL = 384
LAYERS = 8
HEADS = 6
MAX_LEN = 1024  # Reduced to 1024 to ensure stability on 1050 Ti FP32
BATCH_SIZE = 2  # Small batch for safety
LR = 3e-4       # Standard Karpathy constant

TRAIN_SHARD = "datasets/processed/wikitext_60k_4096/train_shard_0000"
TOKENIZER_FILE = "artifacts/zia_tokenizer_60k_clean/tokenizer.json"

# ==========================================
# MODEL DEFINITION
# ==========================================
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        
        # Standard PyTorch Transformer Layer (Reliable)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=HEADS, dim_feedforward=D_MODEL*4,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True
            )
            for _ in range(LAYERS)
        ])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok.weight # Weight tying

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device)
        h = self.tok(x) + self.pos(pos)
        
        for block in self.blocks:
            h = block(h)
            
        h = self.ln_f(h)
        return self.head(h)

# ==========================================
# DATASET
# ==========================================
class RamDataset(Dataset):
    def __init__(self, path):
        # Loading only first 2000 samples for this speed test
        print(f"[i] Loading small subset of {path}...")
        self.x = torch.load(f"{path}/input_ids.pt", map_location="cpu")[:2000]
        self.y = torch.load(f"{path}/labels.pt", map_location="cpu")[:2000]
        
        # Crop to MAX_LEN
        self.x = self.x[:, :MAX_LEN].long()
        self.y = self.y[:, :MAX_LEN].long()

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

# ==========================================
# TRAIN LOOP
# ==========================================
def main():
    torch.manual_seed(1337)
    print(f"[i] Device: {DEVICE}")
    print(f"[i] Mode:   FP32 (No AMP) | Context: {MAX_LEN}")

    # 1. Init
    model = TinyGPT().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # 2. Data
    ds = RamDataset(TRAIN_SHARD)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Loop
    model.train()
    print("\n[i] Starting Sanity Check Training...")
    print(f"{'Step':<6} | {'Loss':<8} | {'GradNorm':<8} | {'Status'}")
    print("-" * 40)

    step = 0
    for epoch in range(5): # Run a few epochs
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=-100)

            # Backward
            optim.zero_grad()
            loss.backward()

            # MONITOR GRADIENTS (The Ghost Buster)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update
            optim.step()

            step += 1
            
            if step % 5 == 0:
                status = "Alive!" if total_norm > 0 else "DEAD 💀"
                print(f"{step:<6} | {loss.item():.4f}   | {total_norm:.4f}   | {status}")

            if step >= 100:
                print("\n[✓] Sanity check complete.")
                if loss.item() < 9.0:
                    print(f"SUCCESS: Loss dropped from ~11.0 to {loss.item():.2f}. The model is learning.")
                else:
                    print("FAILURE: Loss is still high. Hardware or Logic issue persists.")
                return

if __name__ == "__main__":
    main()