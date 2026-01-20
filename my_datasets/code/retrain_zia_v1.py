#!/usr/bin/env python3
# train_zia_cursor_optimized.py
# Faster ZIA IFT retraining with cursor-style context and longer sequence handling (e.g. 1024)
# - flexible/resumable checkpoint loading (optimizer + scheduler restore)
# - pos-embedding expansion (512 -> 1024) and vocab resizing
# - cursor random cropping for long sequences for faster convergence
# - safer FP16 (auto/force/off)
# - checkpointing saves optimizer + scheduler

import os, glob, time, traceback
from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# --- environment
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    dataset_path: str = "artifacts/tokenized_dataset/zia_ift_v3_verified"
    tokenizer_path: str = "artifacts/zia_tokenizer_60k"
    output_dir: str = "artifacts/zia_ift_v4_cursor"
    # fallback (weights-only) checkpoint to warm-start from (512 positions likely)
    fallback_ckpt: str = "artifacts/zia_ift_v3_retrain/best_val/checkpoint.pt"
    # If you have a resume dir with step_*.pt saved by this script, it'll resume
    resume_dir: str = "artifacts/zia_ift_v4_cursor"
    # Model arch (explicit)
    max_len: int = 1024
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1
    # training
    batch_size: int = 6              # micro-batch (tweak to fit VRAM)
    grad_accum: int = 6
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 50
    max_steps: int = 20000
    eval_every: int = 400
    save_every: int = 200
    # perf
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16_mode: str = "auto"   # "auto" | "force" | "off"
    seed: int = 42
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

cfg = TrainConfig()

# reproducibility / perf
torch.manual_seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)
torch.backends.cudnn.benchmark = True

# -----------------------------
# Model
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, mlp_ratio, dropout):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mlp_ratio, dropout)

    def forward(self, x, key_padding_mask=None):
        h = self.ln1(x)
        T = h.size(1)
        # causal mask (bool) — MultiheadAttention supports bool masks for attn_mask
        causal_mask = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(h, h, h, attn_mask=causal_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + out
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model=384, n_layers=8, n_heads=6, mlp_ratio=4, max_len=1024, dropout=0.1, pad_token_id=0):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.max_len = max_len
        self.pad_token_id = pad_token_id

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape
        if T > self.max_len:
            input_ids = input_ids[:, -self.max_len:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_len:]
            T = input_ids.size(1)
        pos_ids = torch.arange(0, T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok(input_ids) + self.pos(pos_ids)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        for blk in self.blocks:
            x = blk(x, key_padding_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1),
                                   ignore_index=-100)
        return logits, loss

# -----------------------------
# Utilities for flexible loading + checkpointing
# -----------------------------
def _atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_checkpoint(model, optimizer, scheduler, out_dir, step, keep_last=5):
    ck_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    path = os.path.join(ck_dir, f"step_{step:06d}.pt")
    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step
    }
    _atomic_save(payload, path)
    files = sorted(glob.glob(os.path.join(ck_dir, "step_*.pt")))
    if len(files) > keep_last:
        for f in files[:-keep_last]:
            try: os.remove(f)
            except Exception: pass
    print(f"[💾] checkpoint saved: {path}")

def save_best_model(model, out_dir, step, metric):
    best_dir = os.path.join(out_dir, "best_val")
    os.makedirs(best_dir, exist_ok=True)
    path = os.path.join(best_dir, "checkpoint.pt")
    payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "step": step, "metric": metric}
    _atomic_save(payload, path)
    print(f"[🏆] Saved best -> {path}")

def expand_pos_and_vocab(state_dict, model, tokenizer):
    """Expand pos.weight if checkpoint shorter than model, and handle tok/head vocab mismatch"""
    sd = model.state_dict()
    # positional
    if "pos.weight" in state_dict and "pos.weight" in sd:
        src = state_dict["pos.weight"]
        dst = sd["pos.weight"]
        old_len, d_old = src.size()
        new_len, d_new = dst.size()
        if d_old != d_new:
            # unlikely, but copy min dims
            print(f"[!] pos dim mismatch: {d_old} vs {d_new}. Copying min dims.")
            tmp = dst.clone()
            tmp[:old_len, :min(d_old, d_new)].copy_(src[:, :min(d_old, d_new)])
            if new_len > old_len:
                # repeat last row to expand
                last = src[-1:].detach().clone()
                reps = last.repeat(new_len - old_len, 1)
                tmp[old_len:].copy_(reps)
            state_dict["pos.weight"] = tmp
        else:
            if new_len > old_len:
                print(f"[i] Expanding pos.weight {old_len} -> {new_len} by copying + repeating last row.")
                dst_copy = dst.clone()
                dst_copy[:old_len].copy_(src)
                last_row = src[-1:].detach().clone()
                if last_row.size(0) > 0:
                    dst_copy[old_len:].copy_(last_row.repeat(new_len - old_len, 1))
                else:
                    nn.init.normal_(dst_copy[old_len:], std=0.02)
                state_dict["pos.weight"] = dst_copy
            elif new_len < old_len:
                print(f"[i] Truncating pos.weight {old_len} -> {new_len}.")
                state_dict["pos.weight"] = src[:new_len]

    # token embeddings / head
    for key in ("tok.weight", "head.weight"):
        if key in state_dict and key in sd:
            src = state_dict[key]
            dst = sd[key]
            old_vocab, d_old = src.size()
            new_vocab, d_new = dst.size()
            if d_old != d_new:
                min_dim = min(d_old, d_new)
                tmp = dst.clone()
                tmp[:old_vocab, :min_dim].copy_(src[:, :min_dim])
                if new_vocab > old_vocab:
                    nn.init.normal_(tmp[old_vocab:], std=0.02)
                state_dict[key] = tmp
                print(f"[i] Adjusted embedding dims for {key}.")
            else:
                if new_vocab != old_vocab:
                    print(f"[i] Handling vocab mismatch for {key}: ckpt={old_vocab} model={new_vocab}")
                    dst_copy = dst.clone()
                    n_copy = min(old_vocab, new_vocab)
                    dst_copy[:n_copy].copy_(src[:n_copy])
                    if new_vocab > old_vocab:
                        nn.init.normal_(dst_copy[old_vocab:], std=0.02)
                    state_dict[key] = dst_copy
    return state_dict

def find_latest_checkpoint(resume_dir):
    ck_dir = os.path.join(resume_dir, "checkpoints")
    if not os.path.exists(ck_dir):
        return None
    ckpts = sorted(glob.glob(os.path.join(ck_dir, "step_*.pt")))
    return ckpts[-1] if ckpts else None

# -----------------------------
# Collate
# -----------------------------
def collate_fn(batch):
    ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return ids, mask, labels

# -----------------------------
# Evaluate
# -----------------------------
@torch.no_grad()
def evaluate(model, dl, device, max_batches=200):
    model.eval()
    tot, n = 0.0, 0
    for i, (ids, mask, labels) in enumerate(dl):
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        _, loss = model(ids, attention_mask=mask, labels=labels)
        tot += loss.item()
        n += 1
        if n >= max_batches: break
    model.train()
    return tot / max(1, n)

# -----------------------------
# Training (cursor)
# -----------------------------
def train_cursor(cfg: TrainConfig):
    device = torch.device(cfg.device)
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    print(f"[i] Tokenizer vocab={len(tok)} | pad={tok.pad_token_id}")

    print(f"[i] Loading dataset from: {cfg.dataset_path}")
    ds = load_from_disk(cfg.dataset_path)
    train_ds, val_ds = ds["train"], ds["validation"]
    print(f"[i] Loaded train={len(train_ds)} | val={len(val_ds)}")

    # DataLoader perf
    num_workers = min(cfg.num_workers, max(0, (os.cpu_count() or 2) - 1))
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=num_workers,
                          pin_memory=cfg.pin_memory, persistent_workers=cfg.persistent_workers,
                          prefetch_factor=2)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=min(2, num_workers),
                        pin_memory=cfg.pin_memory)

    # model (explicit arch from cfg)
    model = TinyGPT(len(tok), d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                    mlp_ratio=cfg.mlp_ratio, max_len=cfg.max_len, dropout=cfg.dropout,
                    pad_token_id=tok.pad_token_id).to(device)

    # FP16 handling
    allow_fp16 = False
    if cfg.fp16_mode == "force":
        allow_fp16 = True
        print("[⚙️] FP16 forced ON (even if GPU unsupported).")
    elif cfg.fp16_mode == "auto":
        allow_fp16 = torch.cuda.is_available()
        if allow_fp16:
            try:
                props = torch.cuda.get_device_properties(device)
                cc = float(props.major) + float(props.minor) / 10.0
                if cc < 7.0:
                    print(f"[!] GPU compute capability {cc:.1f} < 7.0, disabling FP16 for safety.")
                    allow_fp16 = False
            except Exception:
                allow_fp16 = True
    else:
        allow_fp16 = False
    scaler = torch.cuda.amp.GradScaler(enabled=allow_fp16)
    print(f"[i] FP16 enabled: {allow_fp16}")

    # Resume detection
    latest = find_latest_checkpoint(cfg.resume_dir)
    ck = None
    if latest and os.path.exists(latest):
        print(f"[↩] Found resume checkpoint: {latest}")
        ck = torch.load(latest, map_location=device)
    else:
        # fallback weights-only checkpoint (may be from previous 512-length run)
        if os.path.exists(cfg.fallback_ckpt):
            print(f"[↩] No resume checkpoint found. Loading fallback weights: {cfg.fallback_ckpt}")
            ck = torch.load(cfg.fallback_ckpt, map_location=device)
        else:
            raise FileNotFoundError("No checkpoint found to initialize from. Provide fallback_ckpt or resume checkpoint.")

    # Prepare state dict & expand embeddings/vocab if needed
    state = ck.get("model", ck) if isinstance(ck, dict) else ck
    state = expand_pos_and_vocab(state, model, tok)

    # Load weights (non-strict)
    model.load_state_dict({k:v for k,v in state.items() if k in model.state_dict()}, strict=False)
    print("[✓] Model weights loaded (flexible).")

    # optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.max_steps)

    start_step = 0
    # If ck contained optimizer/scheduler/step, restore them
    if isinstance(ck, dict) and "optimizer" in ck and "scheduler" in ck:
        try:
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            start_step = int(ck.get("step", 0))
            print(f"[↩] Restored optimizer/scheduler from checkpoint. Resuming from step {start_step}")
        except Exception as e:
            print("[!] Failed to restore optimizer/scheduler (will start fresh):", e)
            start_step = 0

    # training loop state
    step = start_step
    micro_steps = 0
    running_loss = 0.0
    best_val = float("inf")

    pbar = tqdm(total=cfg.max_steps, initial=step, desc="Training", unit="step", dynamic_ncols=True)
    pbar.set_postfix({"loss":"n/a"})

    try:
        model.train()
        for epoch in range(999999):
            for ids, mask, labels in train_dl:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)

                # Cursor cropping: take a random slice when sequence longer than model.max_len
                seq_len = ids.size(1)
                if seq_len > cfg.max_len:
                    start = torch.randint(0, seq_len - cfg.max_len + 1, (1,)).item()
                    ids = ids[:, start:start+cfg.max_len]
                    mask = mask[:, start:start+cfg.max_len]
                    labels = labels[:, start:start+cfg.max_len]

                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=allow_fp16):
                    _, loss = model(ids, attention_mask=mask, labels=labels)
                    if loss is None:
                        continue
                    loss = loss / cfg.grad_accum

                if allow_fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                micro_steps += 1
                running_loss += loss.item() * cfg.grad_accum

                if micro_steps % cfg.grad_accum == 0:
                    if allow_fp16:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    step += 1
                    pbar.update(1)
                    avg_loss = running_loss / cfg.grad_accum
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "step": step})
                    running_loss = 0.0

                    # logging evaluation
                    if step % cfg.eval_every == 0:
                        val_loss = evaluate(model, val_dl, device)
                        print(f"\n[Eval] step={step} val_loss={val_loss:.4f}")
                        if val_loss < best_val:
                            best_val = val_loss
                            save_best_model(model, cfg.output_dir, step, best_val)

                    # checkpoint (save optimizer + scheduler)
                    if step % cfg.save_every == 0:
                        save_checkpoint(model, optimizer, scheduler, cfg.output_dir, step, keep_last=5)

                    if step >= cfg.max_steps:
                        print("[✓] Training complete.")
                        save_checkpoint(model, optimizer, scheduler, cfg.output_dir, step, keep_last=5)
                        return
            # end epoch
        # end for
    except KeyboardInterrupt:
        print("[⛔] Interrupted — saving checkpoint...")
        save_checkpoint(model, optimizer, scheduler, cfg.output_dir, step, keep_last=5)
    except Exception as e:
        traceback.print_exc()
        save_checkpoint(model, optimizer, scheduler, cfg.output_dir, step, keep_last=5)
        raise
    finally:
        pbar.close()

if __name__ == "__main__":
    os.makedirs(cfg.output_dir, exist_ok=True)
    train_cursor(cfg)
