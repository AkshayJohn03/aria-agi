#!/usr/bin/env python3
"""
train_ift_fixed4k.py
Clean, fast, fixed-length training for ZIA.

- Uses preprocessed 4096-token padded dataset (.pt shards)
- No dynamic padding, no bucketing (fastest possible path)
- FP16 autocast + GradScaler
- Gradient checkpointing (optional)
- Shard streaming (low RAM usage)
- Single tqdm bar showing true training steps
- Autosave every 30 minutes (async, model-only)
"""

import os, time, random, glob, warnings, threading
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# -----------------------------------------
# ENV
# -----------------------------------------
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]

os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

# CUDA allocator config for 4GB GPU (prevents allocation errors)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")

# -----------------------------------------
# CONFIG
# -----------------------------------------
@dataclass
class CFG:
    DATASET_DIR = "datasets/processed/zia_superift_4k_torch"
    TOKENIZER_PAD_ID = 60003         # from your tokenizer
    OUT_DIR = "artifacts/zia_ift_fixed4k"
    GOLD_CKPT = "artifacts/zia_ift_v4_longctx/checkpoints/checkpoint_auto_step5255.pt"

    VOCAB = 60004
    D_MODEL = 384
    N_LAYERS = 8
    N_HEADS = 6
    MLP_RATIO = 4
    DROPOUT = 0.1
    CONTEXT_LEN = 4096  # Model architecture supports 4096
    TRAIN_SEQ_LEN = 4096  # Train on full 4k sequences (use 3072 if OOM)

    MICRO_BATCH = 1
    GRAD_ACCUM = 32
    LR = 3e-5
    CLIP = 1.0
    FP16 = True

    MAX_STEPS = 20000
    AUTOSAVE_MIN = 30
    USE_CHECKPOINTING = True

    NUM_WORKERS = 0
    PIN_MEMORY = False
    
    # Test mode (quick check before full training)
    TEST_RUN = True
    TEST_STEPS = 4

CFG = CFG()

# -----------------------------------------
# MODEL
# -----------------------------------------
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
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d, heads, m, drop, use_checkpoint=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff = FeedForward(d, m, drop)
        self.use_checkpoint = use_checkpoint

    def _forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones((T, T), device=x.device, dtype=torch.bool), 1)
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x

    def forward(self, x):
        if self.use_checkpoint and self.training and x.requires_grad:
            return cp.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class TinyGPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, m, max_len, drop, pad_id, use_checkpoint=False):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len

        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([
            Block(d, heads, m, drop, use_checkpoint=use_checkpoint)
            for _ in range(layers)
        ])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # tie weights

    def forward(self, ids, labels=None):
        B, T = ids.shape
        if T > self.max_len:
            ids = ids[:, -self.max_len:]
            T = ids.size(1)

        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.tok(ids) + self.pos(pos)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln(x)
        logits = self.head(x)

        if labels is None:
            return logits, None

        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:T].reshape(-1),
            ignore_index=-100
        )
        return logits, loss


# -----------------------------------------
# GOLD WEIGHT LOADER
# -----------------------------------------
def load_gold(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        return 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ck = torch.load(ckpt_path, map_location="cpu")

    state = ck.get("model", ck)
    sd = model.state_dict()
    matched = 0
    for k, v in state.items():
        if k in sd and sd[k].shape == v.shape:
            sd[k].copy_(v)
            matched += 1
    model.load_state_dict(sd, strict=False)
    return matched

def get_ckpt_context(ckpt_path):
    """Return context length from checkpoint (pos.weight rows) or None."""
    if not os.path.exists(ckpt_path):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ck = torch.load(ckpt_path, map_location="cpu")
        state = ck.get("model", ck)
        pos = state.get("pos.weight", None)
        if pos is not None:
            return pos.shape[0]
        return ck.get("context_len", None)
    except Exception:
        return None

def expand_pos_embedding(model, new_ctx):
    """Expand positional embeddings safely (copy + extend)."""
    old = model.pos.weight.data
    old_n, dim = old.shape
    if new_ctx <= old_n:
        return
    new = torch.zeros((new_ctx, dim), device=old.device, dtype=old.dtype)
    new[:old_n] = old
    # Repeat last vector for new positions
    new[old_n:] = old[-1:].repeat(new_ctx - old_n, 1)
    model.pos = nn.Embedding(new_ctx, dim).to(model.pos.weight.device)
    model.pos.weight.data.copy_(new)
    model.max_len = new_ctx


# -----------------------------------------
# DATA WRAPPER
# -----------------------------------------
class PTShardDataset(Dataset):
    """Loads .pt shards containing fixed-length tensors."""
    def __init__(self, ids_tensor, lbl_tensor, max_len=None):
        self.ids = ids_tensor
        self.lbl = lbl_tensor
        self.max_len = max_len  # Truncate to this length during training

    def __len__(self):
        return self.ids.size(0)

    def __getitem__(self, idx):
        ids = self.ids[idx]
        lbl = self.lbl[idx]
        # Truncate to max_len if specified (for memory efficiency)
        if self.max_len and ids.size(0) > self.max_len:
            ids = ids[:self.max_len]
            lbl = lbl[:self.max_len]
        return ids, lbl


# -----------------------------------------
# ASYNC CHECKPOINT SAVER
# -----------------------------------------
_save_queue = []
_save_lock = threading.Lock()
_save_thread = None

def _async_save_worker():
    while True:
        with _save_lock:
            if not _save_queue:
                time.sleep(0.1)
                continue
            obj, path = _save_queue.pop(0)
        try:
            tmp = path + ".tmp"
            torch.save(obj, tmp)
            os.replace(tmp, path)
        except Exception as e:
            print(f"[WARNING] async save failed: {e}")

def async_save(obj, path):
    global _save_thread
    if _save_thread is None or not _save_thread.is_alive():
        _save_thread = threading.Thread(target=_async_save_worker, daemon=True)
        _save_thread.start()
    with _save_lock:
        _save_queue.append((obj, path))


# -----------------------------------------
# TRAIN
# -----------------------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[i] Device={device}  FP16={CFG.FP16}")
    print(f"[i] Dataset: 4096 tokens | Training on: {CFG.TRAIN_SEQ_LEN} tokens")
    print(f"[i] Batch: {CFG.MICRO_BATCH} × {CFG.GRAD_ACCUM} = {CFG.MICRO_BATCH * CFG.GRAD_ACCUM} effective")

    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(CFG.OUT_DIR, "checkpoints"), exist_ok=True)

    model = TinyGPT(
        CFG.VOCAB, CFG.D_MODEL, CFG.N_LAYERS, CFG.N_HEADS,
        CFG.MLP_RATIO, CFG.CONTEXT_LEN, CFG.DROPOUT,
        CFG.TOKENIZER_PAD_ID, use_checkpoint=CFG.USE_CHECKPOINTING
    ).to(device)

    matched = load_gold(model, CFG.GOLD_CKPT)
    print(f"[i] Loaded gold checkpoint — matched {matched} tensors")
    
    # Expand positional embeddings if checkpoint has smaller context
    ckpt_ctx = get_ckpt_context(CFG.GOLD_CKPT)
    if ckpt_ctx and ckpt_ctx < model.pos.num_embeddings:
        expand_pos_embedding(model, model.pos.num_embeddings)
        torch.cuda.synchronize()
        print(f"[i] Expanded pos embedding from {ckpt_ctx} → {model.pos.num_embeddings}")

    optim = torch.optim.AdamW(model.parameters(), lr=CFG.LR)
    scaler = torch.amp.GradScaler('cuda', enabled=CFG.FP16)  # Fixed deprecation warning

    # discover shards
    train_shards = sorted(glob.glob(os.path.join(CFG.DATASET_DIR, "train", "*")))
    print(f"[i] Total shards: {len(train_shards)}")

    step = 0
    last_save = time.time()

    pbar = tqdm(total=CFG.MAX_STEPS, desc="Training", dynamic_ncols=True)
    
    if CFG.TEST_RUN:
        print(f"[TEST MODE] Will stop after {CFG.TEST_STEPS} steps")

    try:
        while step < CFG.MAX_STEPS:
            random.shuffle(train_shards)

            for shard in train_shards:
                if step >= CFG.MAX_STEPS:
                    break

                # load tensors (use weights_only=False for safety, but may fail on old pickles)
                try:
                    ids = torch.load(os.path.join(shard, "input_ids.pt"), weights_only=False)
                    lbl = torch.load(os.path.join(shard, "labels.pt"), weights_only=False)
                except Exception:
                    # Fallback for old format
                    ids = torch.load(os.path.join(shard, "input_ids.pt"))
                    lbl = torch.load(os.path.join(shard, "labels.pt"))

                # Create dataset with sequence length truncation for training
                ds = PTShardDataset(ids, lbl, max_len=CFG.TRAIN_SEQ_LEN)
                del ids, lbl  # Free memory immediately

                loader = DataLoader(
                    ds,
                    batch_size=CFG.MICRO_BATCH,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=True
                )

                acc = 0
                for ids_batch, lbl_batch in loader:
                    if step >= CFG.MAX_STEPS:
                        break

                    ids_batch = ids_batch.to(device, non_blocking=True)
                    lbl_batch = lbl_batch.to(device, non_blocking=True)

                    with torch.amp.autocast(device_type="cuda", enabled=CFG.FP16):
                        _, loss = model(ids_batch, lbl_batch)
                        raw_loss = loss.item()
                        loss = loss / CFG.GRAD_ACCUM

                    scaler.scale(loss).backward()
                    acc += 1

                    if acc % CFG.GRAD_ACCUM == 0:
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.CLIP)
                        scaler.step(optim)
                        scaler.update()
                        optim.zero_grad(set_to_none=True)

                        step += 1
                        pbar.update(1)
                        pbar.set_postfix({"loss": f"{raw_loss:.4f}", "step": step})
                        
                        # Test mode check
                        if CFG.TEST_RUN and step >= CFG.TEST_STEPS:
                            tqdm.write(f"[TEST] Reached {CFG.TEST_STEPS} steps, stopping...")
                            raise KeyboardInterrupt()

                        # Autosave every 30 min
                        if time.time() - last_save > CFG.AUTOSAVE_MIN * 60:
                            ck = os.path.join(CFG.OUT_DIR, "checkpoints",
                                              f"checkpoint_step{step}.pt")
                            async_save({"model": model.state_dict(),
                                        "step": step,
                                        "context_len": CFG.CONTEXT_LEN}, ck)
                            last_save = time.time()
                            tqdm.write(f"[💾] queued autosave at step {step}")

                del ds, loader
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete before next shard

    except KeyboardInterrupt:
        tqdm.write("[!] Interrupted — saving checkpoint...")
        final_ck = os.path.join(CFG.OUT_DIR, "checkpoints",
                                f"final_step{step}.pt")
        torch.save({"model": model.state_dict(),
                    "step": step,
                    "context_len": CFG.CONTEXT_LEN}, final_ck)
        tqdm.write(f"[DONE] saved interrupt checkpoint: {final_ck}")

    finally:
        pbar.close()
        # Final save
        final = os.path.join(CFG.OUT_DIR, "checkpoints", "final.pt")
        torch.save({"model": model.state_dict(),
                    "step": step,
                    "context_len": CFG.CONTEXT_LEN}, final)
        print(f"[DONE] Training finished → {final}")


if __name__ == "__main__":
    train()
