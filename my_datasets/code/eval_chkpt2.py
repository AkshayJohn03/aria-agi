#!/usr/bin/env python3
"""
eval_ift_fixed4k.py
Evaluation for ZIA TinyGPT checkpoints trained on fixed 4k .pt shards.

- Loads .pt val shard (input_ids.pt, labels.pt)
- Evaluates checkpoints in artifacts/zia_ift_fixed4k/checkpoints
- Computes average cross-entropy loss
- Handles positional embedding mismatch
"""

import os, glob, torch, warnings, time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------- Environment ----------------
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ================================================================
# CONFIG
# ================================================================
CHECKPOINT_DIR = "artifacts/zia_ift_fixed4k/checkpoints"
VAL_SHARD = "datasets/processed/zia_superift_4k_torch/val/shard_000"

VOCAB = 60004
D_MODEL = 384
N_LAYERS = 8
N_HEADS = 6
MLP_RATIO = 4
DROPOUT = 0.1
MAX_CTX = 4096    # model supports 4096 (even if training used 2048)

BATCH_SIZE = 2
MAX_EVAL_BATCHES = 100   # evaluate up to 100 batches
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[i] Device: {DEVICE}")

# ================================================================
# DATASET LOADER FOR PT SHARD
# ================================================================
class PTShardValDataset(Dataset):
    def __init__(self, shard_path):
        ids_path = os.path.join(shard_path, "input_ids.pt")
        lbl_path = os.path.join(shard_path, "labels.pt")

        if not os.path.exists(ids_path):
            raise RuntimeError(f"Missing file: {ids_path}")
        if not os.path.exists(lbl_path):
            raise RuntimeError(f"Missing file: {lbl_path}")

        self.ids = torch.load(ids_path)
        self.lbl = torch.load(lbl_path)

        assert self.ids.size(0) == self.lbl.size(0), "Row mismatch in shard"

    def __len__(self):
        return self.ids.size(0)

    def __getitem__(self, idx):
        return self.ids[idx], self.lbl[idx]


def collate_stack(batch):
    ids, lbl = zip(*batch)
    return torch.stack(ids), torch.stack(lbl)


# ================================================================
# MODEL DEFINITION
# ================================================================
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
    def __init__(self, d, heads, m, drop):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff = FeedForward(d, m, drop)

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones((T, T), device=x.device, dtype=torch.bool), 1)
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, m, max_len, drop, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len

        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(
            [Block(d, heads, m, drop) for _ in range(layers)]
        )
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # tied

    def forward(self, ids, labels=None):
        B, T = ids.shape
        T = min(T, self.max_len)
        ids = ids[:, :T]
        pos_ids = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.tok(ids) + self.pos(pos_ids)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln(x)
        logits = self.head(x)

        if labels is None:
            return logits, None

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:T].reshape(-1),
            ignore_index=-100
        )
        return loss


# ================================================================
# LOAD VALIDATION DATA
# ================================================================
print(f"[i] Loading validation shard: {VAL_SHARD}")
val_ds = PTShardValDataset(VAL_SHARD)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_stack
)

# ================================================================
# EVALUATION LOOP
# ================================================================
checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_step*.pt")))
print(f"[i] Found {len(checkpoints)} checkpoints")

results = []

for ckpt in checkpoints:
    ckpt_name = os.path.basename(ckpt)
    print(f"\n=== Evaluating {ckpt_name} ===")

    # Build model
    model = TinyGPT(VOCAB, D_MODEL, N_LAYERS, N_HEADS,
                    MLP_RATIO, MAX_CTX, DROPOUT, pad_id=60003).to(DEVICE)

    # Load checkpoint weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ck = torch.load(ckpt, map_location="cpu")

    state = ck.get("model", ck)
    step = ck.get("step", "?")
    ck_ctx = state["pos.weight"].shape[0]

    # Fix positional embedding size mismatch
    if ck_ctx != MAX_CTX:
        new_pos = torch.zeros((MAX_CTX, D_MODEL))
        new_pos[:ck_ctx] = state["pos.weight"]
        new_pos[ck_ctx:] = state["pos.weight"][-1]
        state["pos.weight"] = new_pos

    model.load_state_dict(state, strict=False)
    model.eval()

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for i, (ids, lbls) in enumerate(val_loader):
            ids = ids.to(DEVICE)
            lbls = lbls.to(DEVICE)

            with torch.amp.autocast("cuda", enabled=False):
                loss = model(ids, lbls)

            total_loss += loss.item()
            count += 1

            if i >= MAX_EVAL_BATCHES:
                break

    avg_loss = total_loss / count
    results.append((ckpt_name, avg_loss, step))

    print(f"[✓] step={step} | ctx={ck_ctx} | loss={avg_loss:.4f}")


# ================================================================
# SUMMARY
# ================================================================
print("\n================ FINAL SUMMARY ================")
best = min(results, key=lambda x: x[1])
print(f"Best Checkpoint: {best[0]}  |  Step={best[2]}  |  Loss={best[1]:.4f}")
print("================================================\n")
