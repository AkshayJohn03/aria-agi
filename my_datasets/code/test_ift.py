#!/usr/bin/env python3
import os, torch, warnings, time, gc
from datasets import load_from_disk
from transformers import AutoTokenizer

# ---------------- Basic Config ----------------
DATASET_DIR = "datasets/processed/zia_superift_v2_balanced/train"
TOKENIZER_DIR = "artifacts/zia_tokenizer_60k"
CKPT = "artifacts/zia_ift_v4_longctx/checkpoints/checkpoint_auto_step5255.pt"
CONTEXT_LEN = 4096

VOCAB = 60004
D = 384
LAYERS = 8
HEADS = 6
MLP = 4
DROPOUT = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Tiny Model ----------------
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class FeedForward(nn.Module):
    def __init__(self, d, m, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d*m),
            nn.GELU(),
            nn.Linear(d*m, d),
            nn.Dropout(drop),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, d, heads, m, drop):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff = FeedForward(d, m, drop)

    def _inner(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        a,_ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x

    def forward(self, x):
        # Activation checkpointing enabled
        return checkpoint.checkpoint(self._inner, x, use_reentrant=False)

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(CONTEXT_LEN, D)
        self.blocks = nn.ModuleList([Block(D, HEADS, MLP, DROPOUT) for _ in range(LAYERS)])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, ids, labels=None):
        B,T = ids.shape
        T = min(T, CONTEXT_LEN)
        ids = ids[:,:T]

        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.tok(ids) + self.pos(pos)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln(x)
        logits = self.head(x)

        if labels is None:
            return logits, None

        loss = F.cross_entropy(
            logits[:,:-1,:].reshape(-1, VOCAB),
            labels[:,1:T].reshape(-1),
            ignore_index=-100
        )
        return logits, loss


# ---------------- Helper ----------------
def print_mem(tag=""):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()/1024/1024
    reserved  = torch.cuda.memory_reserved()/1024/1024
    print(f"[MEM] {tag}: allocated={allocated:.1f} MB  reserved={reserved:.1f} MB")

# ---------------- MAIN TEST ----------------
def main():
    print("### TEST RUN START ###")
    print(f"Using device = {device}")

    # Load dataset shard
    shards = [os.path.join(DATASET_DIR, d) for d in os.listdir(DATASET_DIR)]
    shards = [s for s in shards if os.path.isdir(s)]
    if not shards:
        print("[ERR] No shards found.")
        return
    first_shard = shards[0]

    print("Loading first shard:", first_shard)
    ds = load_from_disk(first_shard)

    # Load 1 example only
    ex = ds[0]
    ids = torch.tensor([ex["input_ids"]], dtype=torch.long).to(device)
    labels = torch.tensor([ex["labels"]], dtype=torch.long).to(device)
    print("Example length:", len(ex["input_ids"]))

    # Load model
    model = TinyGPT().to(device)
    print_mem("after model load")

    # Load checkpoint weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ck = torch.load(CKPT, map_location="cpu")
    state = ck.get("model", ck)
    sd = model.state_dict()
    matched = 0
    for k,v in state.items():
        if k in sd and sd[k].shape == v.shape:
            sd[k].copy_(v)
            matched += 1
    print(f"Loaded gold weights, matched={matched}")

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.cuda.amp.GradScaler()

    print_mem("before forward")

    # forward
    start = time.time()
    with torch.cuda.amp.autocast():
        logits, loss = model(ids, labels)
    torch.cuda.synchronize()
    print("Forward done, loss=", loss.item())
    print_mem("after forward")

    # backward
    scaler.scale(loss).backward()
    torch.cuda.synchronize()
    print("Backward done")
    print_mem("after backward")

    # step
    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optim)
    scaler.update()
    optim.zero_grad(set_to_none=True)
    print("Optimizer step done")
    print_mem("after optim step")

    print("### TEST RUN COMPLETE ###")


if __name__ == "__main__":
    main()
