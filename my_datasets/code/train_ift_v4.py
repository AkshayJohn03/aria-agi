#!/usr/bin/env python3
"""
train_ift_minimal_forced_fast.py

Minimal fast trainer — strict fast-path (tokenized dataset). No fallbacks, no resizing.
Assumes dataset_dir contains a 'train' split with examples having 'input_ids' lists.
"""
import os, time, glob, random, traceback
from dataclasses import dataclass
from typing import Optional
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer

# ---------------- Environment ----------------
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------- CONFIG ----------------
@dataclass
class CFG:
    DATASET_DIR: str = "artifacts/tokenized_dataset/zia_ift_v3_cleaned"
    TOKENIZER_DIR: str = "artifacts/zia_tokenizer_60k"   # used only for pad token id
    OUT_DIR: str = "artifacts/zia_ift_essential_runs"
    GOLD_CKPT: Optional[str] = "artifacts/zia_ift_v4_longctx/checkpoints/checkpoint_auto_step5255.pt"

    # model
    VOCAB: int = 60004
    D_MODEL: int = 384
    N_LAYERS: int = 8
    N_HEADS: int = 6
    MLP_RATIO: int = 4
    MAX_LEN: int = 4096
    DROPOUT: float = 0.1

    # training
    MICRO_BATCH: int = 4
    GRAD_ACCUM: int = 8
    LR: float = 3e-5
    WEIGHT_DECAY: float = 0.0
    MAX_STEPS: int = 20000
    FP16: bool = True
    SEED: int = 42

    # io/perf
    NUM_WORKERS: int = 4
    PREFETCH_FACTOR: int = 2
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True

    # checkpoint
    SAVE_EVERY_SECONDS: int = 30 * 60   # 30 minutes
    KEEP_LAST: int = 5

    # grad clip
    CLIP_NORM: float = 1.0

CFG = CFG()
# lower workers on Windows
if os.name == "nt":
    CFG.NUM_WORKERS = min(2, CFG.NUM_WORKERS)

# ---------------- Model (compact TinyGPT) ----------------
class FeedForward(nn.Module):
    def __init__(self, d, m, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * m), nn.GELU(),
            nn.Linear(d * m, d), nn.Dropout(drop)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, d, h, m, drop):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=h, batch_first=True, dropout=drop)
        self.ff = FeedForward(d, m, drop)
    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones((T, T), device=x.device, dtype=torch.bool), diagonal=1)
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
        return x + a + self.ff(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, mlp_ratio, max_len, drop, pad_id=0):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads, mlp_ratio, drop) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.max_len = max_len
        self.pad_token_id = pad_id

    def forward(self, ids, labels=None):
        B, T = ids.shape
        if T > self.max_len:
            ids = ids[:, -self.max_len:]
            T = ids.shape[1]
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.tok(ids) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                   labels[:, 1:].reshape(-1),
                                   ignore_index=self.pad_token_id)
        return logits, loss

# ---------------- Utilities ----------------
def atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def prune(out_dir, keep):
    ck_dir = os.path.join(out_dir, "checkpoints")
    files = sorted(glob.glob(os.path.join(ck_dir, "step_*.pt")))
    if len(files) > keep:
        for old in files[:-keep]:
            try: os.remove(old)
            except Exception: pass

def save_model_only(model, step, out_dir, keep_last=5):
    ck_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    fname = os.path.join(ck_dir, f"step_{step:06d}.pt")
    atomic_save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 "step": step, "timestamp": time.time()}, fname)
    prune(out_dir, keep_last)
    print(f"[CHKPT] saved {fname}")

def load_matching_weights(model, ckpt_path):
    """
    Load weights-only but only copy parameters that match exact shapes.
    No positional resizing, no vocab padding — silent skip on mismatch.
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        return False
    print(f"[i] Loading weights-only (matching shapes) from {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck.get("model", ck) if isinstance(ck, dict) else ck
    sd = model.state_dict()
    copied = 0; skipped = 0
    for k, v in state.items():
        if k in sd and isinstance(v, torch.Tensor) and sd[k].shape == v.shape:
            sd[k].copy_(v.to(sd[k].device))
            copied += 1
        else:
            skipped += 1
    model.load_state_dict(sd, strict=False)
    model.head.weight = model.tok.weight
    print(f"[i] copied={copied} skipped={skipped}")
    return True

# ---------------- Collator (FAST PATH only) ----------------
class CollateFast:
    """Expect each example to have 'input_ids' (list[int]) or tuple (input_ids, labels)."""
    def __init__(self, pad_id):
        self.pad = pad_id
    def __call__(self, batch):
        ids_list = []
        lbls_list = []
        for ex in batch:
            if isinstance(ex, dict):
                inp = ex["input_ids"]
                lab = ex.get("labels", None)
            elif isinstance(ex, (list, tuple)) and len(ex) >= 1:
                inp = ex[0]
                lab = ex[1] if len(ex) > 1 else None
            else:
                raise RuntimeError("Dataset example not supported by fast trainer.")
            ids_list.append(inp)
            lbls_list.append(lab if lab is not None else inp)
        maxl = max(len(x) for x in ids_list)
        padded = [ x + [self.pad]*(maxl - len(x)) for x in ids_list ]
        padded_lbl = [ (y + [-100]*(maxl - len(y))) if y is not None else [-100]*maxl for y in lbls_list ]
        return torch.tensor(padded, dtype=torch.long), torch.tensor(padded_lbl, dtype=torch.long)

# ---------------- TRAIN ----------------
def train():
    torch.manual_seed(CFG.SEED); random.seed(CFG.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Device={device} FP16={CFG.FP16}")

    # tokenizer only to get pad token id (no tokenization used)
    pad_id = 0
    if os.path.exists(CFG.TOKENIZER_DIR):
        try:
            tok = AutoTokenizer.from_pretrained(CFG.TOKENIZER_DIR, use_fast=True, local_files_only=True)
            if tok.pad_token is None:
                tok.add_special_tokens({"pad_token":"<pad>"})
            pad_id = tok.pad_token_id
        except Exception:
            pad_id = 0

    # load tokenized dataset (fast-path)
    if not os.path.exists(CFG.DATASET_DIR):
        raise RuntimeError(f"Dataset path missing: {CFG.DATASET_DIR}")
    ds = load_from_disk(CFG.DATASET_DIR)
    train_ds = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
    print(f"[i] Training samples = {len(train_ds)}")

    # quick sanity: check first example has 'input_ids'
    e0 = train_ds[0]
    if isinstance(e0, dict) and "input_ids" not in e0:
        raise RuntimeError("Dataset does not contain 'input_ids' field. This trainer requires tokenized dataset.")

    collate = CollateFast(pad_id)
    loader = DataLoader(train_ds, batch_size=CFG.MICRO_BATCH, shuffle=True,
                        num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY,
                        collate_fn=collate, drop_last=True,
                        persistent_workers=CFG.PERSISTENT_WORKERS,
                        prefetch_factor=CFG.PREFETCH_FACTOR)

    # model
    model = TinyGPT(CFG.VOCAB, CFG.D_MODEL, CFG.N_LAYERS, CFG.N_HEADS, CFG.MLP_RATIO, CFG.MAX_LEN, CFG.DROPOUT, pad_id)
    model.pad_token_id = pad_id
    model.to(device)
    print("[i] Model instantiated.")

    # load gold checkpoint by matching shapes (no resizing)
    if CFG.GOLD_CKPT and os.path.exists(CFG.GOLD_CKPT):
        try:
            load_matching_weights(model, CFG.GOLD_CKPT)
        except Exception as e:
            print("[!] gold ckpt load failed:", e)

    # optimizer & scaler
    optim = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(CFG.FP16 and device.type=="cuda"))

    step = 0; micro = 0
    last_save = time.time()
    pbar = tqdm(total=CFG.MAX_STEPS, desc="IFT Training", initial=0, unit="step")

    try:
        model.train()
        while step < CFG.MAX_STEPS:
            for ids, labels in loader:
                ids = ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=(CFG.FP16 and device.type=="cuda")):
                    _, loss = model(ids, labels)
                    if loss is None:
                        continue
                    loss = loss / CFG.GRAD_ACCUM

                scaler.scale(loss).backward()
                micro += 1

                if micro % CFG.GRAD_ACCUM == 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.CLIP_NORM)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)
                    step += 1
                    pbar.update(1)

                    # time-based save (model-only)
                    if time.time() - last_save >= CFG.SAVE_EVERY_SECONDS:
                        save_model_only(model, step, CFG.OUT_DIR, CFG.KEEP_LAST)
                        last_save = time.time()

                if step >= CFG.MAX_STEPS:
                    break
            if step >= CFG.MAX_STEPS:
                break

        save_model_only(model, step, CFG.OUT_DIR, CFG.KEEP_LAST)
        print("[✓] Training finished. Final checkpoint saved.")
    except KeyboardInterrupt:
        print("[!] Interrupted — saving checkpoint.")
        save_model_only(model, step, CFG.OUT_DIR, CFG.KEEP_LAST)
    except Exception:
        traceback.print_exc()
        print("[!] Exception — saving checkpoint before exit.")
        try: save_model_only(model, step, CFG.OUT_DIR, CFG.KEEP_LAST)
        except: pass
        raise
    finally:
        pbar.close()

if __name__ == "__main__":
    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    print("[i] Starting forced-fast minimal IFT trainer (no fallback).")
    train()
