import torch
import torch.nn as nn

# ---------- Define Your Training Architecture ----------
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, mlp_ratio, max_len, dropout):
        super().__init__()
        
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=int(d_model * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.head(h)

# ---------- Instantiate Model ----------
VOCAB = 60011
D_MODEL = 384
N_LAYERS = 8
N_HEADS = 6
MLP_RATIO = 4
MAX_LEN = 256  # <-- USE THIS UNLESS 100% SURE CHECKPOINT USED 4096

model = TinyGPT(VOCAB, D_MODEL, N_LAYERS, N_HEADS, MLP_RATIO, MAX_LEN, 0.1)

# ---------- Load Checkpoint ----------
ckpt_path = "artifacts/zia_mixed_v1/checkpoint_mixed_step10703.pt"
ck = torch.load(ckpt_path, map_location="cpu")

state = ck.get("model", ck)

# ---------- Try Loading ----------
print("\n=== Checking STATE DICT COMPATIBILITY ===\n")

missing, unexpected = model.load_state_dict(state, strict=False)

print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# ---------- Check Embedding Shapes ----------
print("\n--- SHAPE CHECKS ---")
print("tok.weight:", model.tok.weight.shape)
print("ckpt tok.weight:", state.get("tok.weight", None).shape if "tok.weight" in state else "MISSING")

print("pos.weight:", model.pos.weight.shape)
print("ckpt pos.weight:", state.get("pos.weight", None).shape if "pos.weight" in state else "MISSING")

# ---------- Check if Head is Tied Properly ----------
print("\n--- WEIGHT TIE CHECK ---")
same = torch.allclose(model.head.weight, model.tok.weight)
print("head.weight tied to tok.weight:", same)

# ---------- Check Number of Blocks ----------
print("\n--- BLOCK COUNT CHECK ---")
print("Model blocks:", len(model.blocks))
print("Checkpoint blocks present:",
      sum("blocks." in k for k in state.keys()))

print("\n=== DONE ===")
