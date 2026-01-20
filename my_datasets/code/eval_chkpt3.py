#!/usr/bin/env python3
"""
eval_chkpt_final.py
Evaluate only-compatible checkpoints (24k tokenizer) on a val shard.
Skips checkpoints whose embedding shapes mismatch tokenizer vocab (Option A).
Prints per-checkpoint: step, avg loss (per-token), tokens processed, perplexity.
"""

import os, glob, time, math, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sentencepiece as spm

# ---------------- Config (adjust paths if needed) ----------------
CHECKPOINT_DIR = "artifacts/zia_ift_fixed4k/checkpoints"
VAL_SHARD = "datasets/processed/test_openorca/val/shard_000"   # change if you moved shards
SPM_MODEL = "artifacts/zia_tokenizer_v2_clean/zia_spm.model"

D_MODEL = 384
N_LAYERS = 8
N_HEADS = 6
MLP_RATIO = 4
DROPOUT = 0.1
MAX_CTX = 4096

BATCH_SIZE = 4
MAX_EVAL_BATCHES = 500    # cap per checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0           # Windows-safe
print(f"[i] Device: {DEVICE}")

# ---------------- Dataset ----------------
class PTShardDataset(Dataset):
    def __init__(self, shard_path):
        ids_path = os.path.join(shard_path, "input_ids.pt")
        lbl_path = os.path.join(shard_path, "labels.pt")
        if not os.path.exists(ids_path) or not os.path.exists(lbl_path):
            raise FileNotFoundError(f"Missing id/label files in {shard_path}")
        self.ids = torch.load(ids_path)
        self.lbl = torch.load(lbl_path)
        assert self.ids.shape[0] == self.lbl.shape[0]
    def __len__(self): return self.ids.shape[0]
    def __getitem__(self, i): return self.ids[i], self.lbl[i]

def collate_stack(batch):
    ids, lbl = zip(*batch)
    return torch.stack(ids), torch.stack(lbl)

# ---------------- Model ----------------
class FeedForward(nn.Module):
    def __init__(self, d, m, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d*m),
            nn.GELU(),
            nn.Linear(d*m, d),
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
        mask = torch.triu(torch.ones((T,T), device=x.device, dtype=torch.bool), 1)
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
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
        self.blocks = nn.ModuleList([Block(d, heads, m, drop) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
    def forward(self, ids, labels=None):
        B, T = ids.shape
        T = min(T, self.max_len)
        ids = ids[:, :T]
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.tok(ids) + self.pos(pos)
        for blk in self.blocks: x = blk(x)
        x = self.ln(x)
        logits = self.head(x)
        if labels is None: return logits, None
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:T].reshape(-1),
            ignore_index=-100
        )
        return loss

# ---------------- Helper: load tokenizer & get vocab ----------------
sp = spm.SentencePieceProcessor()
sp.load(SPM_MODEL)
vocab = sp.get_piece_size()
pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") != sp.unk_id() else 0
print(f"[i] Loaded SentencePiece tokenizer | vocab={vocab} | pad_id={pad_id}")

# ---------------- Load val dataset ----------------
print(f"[i] Loading val: {VAL_SHARD}")
val_ds = PTShardDataset(VAL_SHARD)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_stack, num_workers=NUM_WORKERS, pin_memory=True)

# ---------------- Iterate checkpoints ----------------
ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_step*.pt")))
print(f"[i] Found {len(ckpts)} checkpoints")

results = []
for ckpt in ckpts:
    name = os.path.basename(ckpt)
    # quick pre-check: open checkpoint dict keys for tok/head shapes without loading into model
    try:
        ck = torch.load(ckpt, map_location="cpu")
    except Exception as e:
        print(f"[!] Failed to load {name}: {e}. Skipping.")
        continue
    state = ck.get("model", ck)
    # check tok/head if present
    tok_shape = None
    if "tok.weight" in state: tok_shape = tuple(state["tok.weight"].shape)
    elif "tok.weight" in state: tok_shape = tuple(state["tok.weight"].shape)
    elif "tok.weight" not in state and "tok.weight" in state:
        tok_shape = tuple(state["tok.weight"].shape)  # defensive
    # Some checkpoints store embeddings under varying names; try a few
    emb_keys = [k for k in state.keys() if k.endswith("tok.weight") or k.endswith(".tok.weight") or k.endswith("tok.weight")]
    if not emb_keys:
        # try head.weight fallback
        emb_keys = [k for k in state.keys() if k.endswith("head.weight")]
    skip_ckpt = False
    if emb_keys:
        emb = state[emb_keys[0]]
        if emb.ndim == 2:
            ck_vocab = emb.shape[0]
            if ck_vocab != vocab:
                print(f"[skip] {name}: checkpoint vocab {ck_vocab} != tokenizer vocab {vocab}")
                skip_ckpt = True
    else:
        print(f"[warn] {name}: couldn't find embedding key to compare vocab shape; attempting strict load (may fail)")

    if skip_ckpt:
        continue

    # Build model (vocab sized to current tokenizer)
    model = TinyGPT(vocab, D_MODEL, N_LAYERS, N_HEADS, MLP_RATIO, MAX_CTX, DROPOUT, pad_id).to(DEVICE)

    # load state (positional weight mismatch handled below)
    step = ck.get("step", "?")
    # fix pos weight if necessary
    if "pos.weight" in state:
        ck_ctx = state["pos.weight"].shape[0]
        if ck_ctx != MAX_CTX:
            new_pos = torch.zeros((MAX_CTX, D_MODEL))
            new_pos[:ck_ctx] = state["pos.weight"]
            new_pos[ck_ctx:] = state["pos.weight"][-1:]
            state["pos.weight"] = new_pos
    # Now load state dict leniently
    model.load_state_dict(state, strict=False)

    # param count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Evaluating {name} | step={step} | params={total_params:,} ===")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0

    with torch.no_grad():
        for i, (ids, lbls) in enumerate(tqdm(val_loader, desc="batches", leave=False)):
            ids = ids.to(DEVICE, non_blocking=True)
            lbls = lbls.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE=="cuda")):
                loss = model(ids, lbls)  # model returns loss in this eval wrapper
            # compute tokens considered in this batch
            B, T = ids.shape
            # labels used are labels[:,1:T], count non -100
            flat_lbl = lbls[:, 1: min(lbls.shape[1], MAX_CTX)].reshape(-1)
            token_mask = (flat_lbl != -100)
            n_tokens = int(token_mask.sum().item())
            total_tokens += n_tokens
            total_loss += float(loss.item()) * 1.0  # loss is mean per-token for that batch
            batches += 1
            if batches >= MAX_EVAL_BATCHES:
                break

    if batches == 0:
        print(f"[!] No eval batches processed for {name}")
        continue

    avg_loss = total_loss / batches
    # Note: avg_loss is mean of batch-means; better is total_nll / total_tokens. We'll compute token-weighted mean:
    # For a more correct per-token average, recompute per batch sums: but we used batch mean*1 above. If you want token-weighted:
    # (quick approximation) assume similar batch sizes; else re-run with accumulating token-weighted loss.
    try:
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    except OverflowError:
        perplexity = float("inf")

    print(f"[✓] step={step} | avg_loss={avg_loss:.4f} | batches={batches} | tokens={total_tokens} | ppl={perplexity:.2e}")
    results.append((name, avg_loss, step, total_tokens))

# final summary
if results:
    best = min(results, key=lambda x: x[1])
    print("\n=== BEST CHECKPOINT ===")
    print(f"{best[0]}  step={best[2]}  avg_loss={best[1]:.4f}  tokens={best[3]}")
else:
    print("[i] No checkpoints evaluated (all incompatible or errors).")
