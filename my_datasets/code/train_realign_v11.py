#!/usr/bin/env python3
"""
train_realign_v11_memopt.py
Memory-optimized trainer:
 - gradient checkpointing (torch.utils.checkpoint)
 - bitsandbytes 8-bit optimizer (AdamW8bit) where available
 - dynamic accum based on free VRAM
 - optional torch.compile
 - safe DataLoader defaults for Windows
"""
import os, glob, time, random, math, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sentencepiece as spm

# Optional: bitsandbytes
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

# ----------------- Config (edit as needed) -----------------
TRAIN_DIR = "datasets/processed/test_openorca/train"
SPM_MODEL = "artifacts/zia_tokenizer_v2_clean/zia_spm.model"
CHECKPOINT_DIR = "artifacts/zia_ift_fixed4k/checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Base model config
D_MODEL=384; N_LAYERS=8; N_HEADS=6; MLP_RATIO=4
CONTEXT_LEN = 4096
BATCH_SIZE = 4          # micro-batch
GRAD_ACCUM = 80         # target effective accumulation
LR = 2.5e-4

NUM_WORKERS = 0         # Windows-safe default
FP16 = True

# ----------------- Helpers -----------------
def set_seed(s=42):
    random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(42)

def get_free_mem_fraction():
    if torch.cuda.is_available():
        try:
            import torch.cuda as tc
            free, total = tc.mem_get_info()
            return free / total
        except Exception:
            return 0.0
    return 0.0

# ----------------- Model (TinyGPT-like) -----------------
class FeedForward(nn.Module):
    def __init__(self,d,m,drop): super().__init__(); self.net = nn.Sequential(nn.Linear(d,d*m), nn.GELU(), nn.Linear(d*m,d), nn.Dropout(drop))
    def forward(self,x): return self.net(x)

class Block(nn.Module):
    def __init__(self,d,h,m,drop):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d,h,dropout=drop,batch_first=True)
        self.ff = FeedForward(d,m,drop)
    def forward(self,x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),1)
        a,_ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self,vocab,d,l,h,m,max_len,drop,pad):
        super().__init__()
        self.tok = nn.Embedding(vocab,d)
        self.pos = nn.Embedding(max_len,d)
        self.blocks = nn.ModuleList([Block(d,h,m,drop) for _ in range(l)])
        self.ln_f = nn.LayerNorm(d); self.head = nn.Linear(d,vocab,bias=False)
        self.head.weight = self.tok.weight
        self.max_len=max_len; self.pad_token_id=pad

    def forward(self, ids, labels=None):
        T = ids.size(1)
        if T > self.max_len: ids = ids[:,:self.max_len]; T = self.max_len
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device).unsqueeze(0))
        for blk in self.blocks: x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:,:-1,:].reshape(-1,logits.size(-1)), labels[:,1:T].reshape(-1), ignore_index=-100)
        return logits, loss

# ----------------- Gradient checkpoint wrapper -----------------
from torch.utils.checkpoint import checkpoint

# We'll checkpoint groups of layers to save memory:
def apply_gradient_checkpointing(model, checkpoint_every=1):
    """
    Wrap forward of each nth block with torch.checkpoint to reduce activation memory.
    checkpoint_every: checkpoint every N blocks
    """
    for i, blk in enumerate(model.blocks):
        if (i % checkpoint_every) == 0:
            # replace with a lambda wrapper that checkpoint the block
            orig = blk
            def make_fn(b):
                return lambda x: b(x)
            model.blocks[i] = lambda_module(orig)

class lambda_module(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        return checkpoint(self.module, x)

# ----------------- Dataset (shard loader) -----------------
class ShardDataset(Dataset):
    def __init__(self, shard_path):
        self.x = torch.load(os.path.join(shard_path, "input_ids.pt"), map_location="cpu")
        self.y = torch.load(os.path.join(shard_path, "labels.pt"), map_location="cpu")
        if not isinstance(self.x, torch.Tensor): self.x = torch.tensor(self.x,dtype=torch.long)
        if not isinstance(self.y, torch.Tensor): self.y = torch.tensor(self.y,dtype=torch.long)
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return self.x[i], self.y[i]

def collate_pad(batch, pad_id=-100):
    maxlen = max(len(a[0]) for a in batch)
    ids = [list(a[0]) + [pad_id]*(maxlen - len(a[0])) for a in batch]
    lbls = [list(a[1]) + [-100]*(maxlen - len(a[1])) for a in batch]
    return torch.tensor(ids, dtype=torch.long), torch.tensor(lbls, dtype=torch.long)

# ----------------- Train -----------------
def train():
    # tokenizer
    sp = spm.SentencePieceProcessor(); sp.load(SPM_MODEL)
    vocab = sp.get_piece_size(); pad_id = 0
    print("[i] Vocab:",vocab,"pad:",pad_id)

    device = torch.device(DEVICE)
    model = TinyGPT(vocab, D_MODEL, N_LAYERS, N_HEADS, MLP_RATIO, CONTEXT_LEN, 0.1, pad_id).to(device)

    # apply coarse gradient checkpointing (checkpoint every 2 blocks)
    # this wraps half the blocks into checkpointed segments (adjustable)
    apply_gradient_checkpointing(model, checkpoint_every=2)
    print("[i] Applied gradient checkpointing (every 2 blocks).")

    # optional torch.compile for speed if available
    try:
        model = torch.compile(model)
        print("[i] torch.compile applied.")
    except Exception:
        pass

    # optimizer: try bitsandbytes 8-bit AdamW if available
    if BNB_AVAILABLE:
        optim = bnb.optim.AdamW8bit(model.parameters(), lr=LR)
        print("[i] Using bitsandbytes AdamW8bit optimizer.")
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=LR)
        print("[i] Using torch AdamW optimizer (bnb not available).")

    scaler = torch.cuda.amp.GradScaler(enabled=FP16 and device.type=="cuda")

    # scheduler: small warmup, cosine
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=20000, eta_min=1e-6)

    shards = sorted(glob.glob(os.path.join(TRAIN_DIR, "shard_*")))
    if not shards: raise RuntimeError("No shards found.")
    print("[i] Found shards:",len(shards))

    step = 0; last_save=time.time()
    target_grad_accum = GRAD_ACCUM

    # TRAIN LOOP (sequential shards)
    for shard_idx, shard in enumerate(shards):
        ds = ShardDataset(shard)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_pad)
        pbar = tqdm(dl, desc=f"Shard {shard_idx+1}/{len(shards)}", dynamic_ncols=True)
        micro_counter = 0
        for ids, lbls in pbar:
            # dynamic memory check: if free fraction low, reduce local grad_accum to avoid OOM stall
            free_frac = get_free_mem_fraction()
            if free_frac < 0.06:
                # force a smaller accumulation to push more frequent optimizer steps
                effective_accum = max(1, int(target_grad_accum // 2))
            else:
                effective_accum = target_grad_accum

            ids = ids.to(device, non_blocking=True); lbls = lbls.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=FP16 and device.type=="cuda"):
                _, loss = model(ids, lbls)
                loss = loss / effective_accum

            scaler.scale(loss).backward()
            micro_counter += 1

            if (micro_counter % effective_accum) == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim); scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                # logging
                pbar.set_postfix({"step": step, "loss": f"{(loss.item()*effective_accum):.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            # autosave
            if time.time() - last_save > 60*30:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                atomic = os.path.join(CHECKPOINT_DIR, f"checkpoint_step{step}.pt")
                torch.save({"model": model.state_dict(), "optimizer": optim.state_dict(), "step": step}, atomic)
                last_save = time.time()
                tqdm.write(f"[💾] Saved checkpoint {atomic}")

    # final save
    final = os.path.join(CHECKPOINT_DIR, f"final_step{step}.pt")
    torch.save({"model": model.state_dict(), "optimizer": optim.state_dict(), "step": step}, final)
    print("[✓] Done. final:", final)

if __name__=="__main__":
    train()
