import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Model Config ----------------
VOCAB_SIZE = 60011
D_MODEL = 384
LAYERS = 8
HEADS = 6
SEQ_LEN = 4096

VAL_SHARD = "datasets/processed/wikitext_60k_4096/val_shard_0000"

# ---------------- Model Definition ----------------
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=HEADS, dim_feedforward=D_MODEL*4,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True
            ) for _ in range(LAYERS)
        ])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))

# ---------------- Evaluate ----------------
@torch.no_grad()
def eval_checkpoint(path):
    print(f"\n[i] Loading checkpoint: {path}")
    ck = torch.load(path, map_location=DEVICE)
    state = ck["model"] if "model" in ck else ck

    model = TinyGPT().to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Load validation shard
    print("[i] Loading validation shard...")
    x = torch.load(Path(VAL_SHARD) / "input_ids.pt", map_location="cpu")[:4, :SEQ_LEN].long().to(DEVICE)
    y = torch.load(Path(VAL_SHARD) / "labels.pt", map_location="cpu")[:4, :SEQ_LEN].long().to(DEVICE)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=-100).item()
    ppl = math.exp(min(loss, 50))

    print(f"\n=== EVAL RESULT ===")
    print(f"Loss: {loss:.6f}")
    print(f"PPL : {ppl:.2f}")
    print("===================\n")

if __name__ == "__main__":
    # edit this path to any ckpt
    path = "artifacts/zia_lm_v20/v20_step102_t1764999441.pt"
    eval_checkpoint(path)
