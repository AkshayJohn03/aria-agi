#!/usr/bin/env python3
# retrain_zia_IFT_resume.py
# Resumable training for TinyGPT (ZIA IFT v3)
# - Auto-detect last checkpoint and resume optimizer/scheduler
# - Fallback to external base checkpoint if no resume found
# - Optional FP16 override ("force" mode for GTX cards)
# - Safe for power interruptions

import os, time, math, glob, traceback
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

# --- Silence TRANSFORMERS_CACHE warning & set cache dirs early ---
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
# On some Windows builds this option might not be supported; we leave it but expect a warning if unsupported.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# --------------------
# Config
# --------------------
@dataclass
class ZiaConfig:
    dataset_path: str = "artifacts/tokenized_dataset/zia_ift_v3"
    tokenizer_path: str = "artifacts/zia_tokenizer_60k"
    output_dir: str = "artifacts/zia_ift_v3_retrain"
    fallback_ckpt: str = "artifacts/zia_ift_v3_fix_stage_2/best_val/checkpoint.pt"
    resume_dir: str = "artifacts/zia_ift_v3_retrain"
    # model
    max_len: int = 1024
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1
    # training
    batch_size: int = 8
    grad_accum: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 0
    max_steps: int = 20000
    eval_every: int = 500
    save_every: int = 100
    num_workers: int = 0
    fp16_mode: str = "force"  # "auto" or "force"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tb_comment: str = "retrain_zia_resume"

# --------------------
# Model definitions
# --------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, mlp_ratio, dropout):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout)
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
        causal = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(h, h, h, attn_mask=causal, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + out
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, mlp_ratio, max_len, dropout, pad_token_id=0):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
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
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        return logits, loss

# --------------------
# Utils
# --------------------
def _atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_checkpoint_step(model, optimizer, scheduler, out_dir, step, keep_last=3):
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

def save_best(model, out_dir, best_metric, step):
    best_dir = os.path.join(out_dir, "anl")
    os.makedirs(best_dir, exist_ok=True)
    path = os.path.join(best_dir, "checkpoint.pt")
    payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "step": step, "metric": best_metric}
    _atomic_save(payload, path)
    print(f"[🏆] Saved best -> {path}")

def collate_fn(batch):
    ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return ids, mask, labels

@torch.no_grad()
def evaluate(model, data_loader, device, max_batches=200):
    model.eval()
    tot, n = 0.0, 0
    for i, (ids, mask, labels) in enumerate(data_loader):
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        _, loss = model(ids, attention_mask=mask, labels=labels)
        tot += loss.item()
        n += 1
        if i + 1 >= max_batches: break
    model.train()
    return tot / max(1, n)

# --------------------
# Training Loop
# --------------------
def train_loop(cfg: ZiaConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    print(f"[i] Tokenizer vocab={len(tok)} | pad={tok.pad_token_id}")

    print(f"[i] Loading dataset from: {cfg.dataset_path}")
    ds_dict = load_from_disk(cfg.dataset_path)
    train_ds, val_ds = ds_dict["train"], ds_dict["validation"]
    print(f"[i] Loaded train={len(train_ds)} | val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=min(2, cfg.num_workers), pin_memory=torch.cuda.is_available(),
                            collate_fn=collate_fn)

    model = TinyGPT(len(tok), cfg.d_model, cfg.n_layers, cfg.n_heads,
                    cfg.mlp_ratio, cfg.max_len, cfg.dropout,
                    pad_token_id=tok.pad_token_id).to(device)

    # ---- FP16 Handling ----
    allow_fp16 = torch.cuda.is_available()
    if cfg.fp16_mode == "force":
        allow_fp16 = True
        print("[⚙️] FP16 forced ON (even if GPU unsupported).")
    elif cfg.fp16_mode == "auto":
        props = torch.cuda.get_device_properties(device)
        cc = float(props.major) + float(props.minor)/10
        if cc < 7.0:
            print(f"[!] GPU compute capability {cc:.1f} < 7.0, disabling FP16.")
            allow_fp16 = False
        else:
            print("[i] Using automatic FP16 (safe mode).")

    scaler = torch.cuda.amp.GradScaler(enabled=allow_fp16)
    print(f"[i] FP16 status: {allow_fp16}")

    # ---- Resume Logic ----
    ckpt_dir = os.path.join(cfg.resume_dir, "checkpoints")
    latest_ckpt = None
    if os.path.exists(ckpt_dir):
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
        if ckpts: latest_ckpt = ckpts[-1]

    if latest_ckpt:
        print(f"[↩] Found resume checkpoint: {latest_ckpt}")
        ck = torch.load(latest_ckpt, map_location=device)
    else:
        print(f"[↩] No resume checkpoint found, using fallback: {cfg.fallback_ckpt}")
        ck = torch.load(cfg.fallback_ckpt, map_location=device)

    # Load weights
    model.load_state_dict(ck["model"], strict=False)
    step = ck.get("step", 0)
    print(f"[i] Model loaded, resuming from step {step}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.max_steps)
    if "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
        print("[↩] Optimizer restored.")
    if "scheduler" in ck:
        scheduler.load_state_dict(ck["scheduler"])
        print("[↩] Scheduler restored.")

    os.makedirs(cfg.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg.output_dir, "runs"), comment=cfg.tb_comment)

    best_val = float("inf")
    micro_steps = 0
    running_loss = 0.0

    pbar = tqdm(total=cfg.max_steps, initial=step, desc="Training", unit="step", dynamic_ncols=True)
    pbar.set_postfix({"loss": "n/a"})

    try:
        model.train()
        while step < cfg.max_steps:
            for ids, mask, labels in train_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                with torch.amp.autocast("cuda", enabled=allow_fp16):
                    _, loss = model(ids, attention_mask=mask, labels=labels)
                    if loss is None: continue
                    loss = loss / cfg.grad_accum

                if allow_fp16: scaler.scale(loss).backward()
                else: loss.backward()

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
                    writer.add_scalar("train/loss", avg_loss, step)
                    running_loss = 0.0

                    if step % cfg.eval_every == 0:
                        val_loss = evaluate(model, val_loader, device)
                        print(f"\n[Eval] step={step} val_loss={val_loss:.4f}")
                        writer.add_scalar("eval/loss", val_loss, step)
                        if val_loss < best_val:
                            best_val = val_loss
                            save_best(model, cfg.output_dir, best_val, step)

                    if step % cfg.save_every == 0:
                        save_checkpoint_step(model, optimizer, scheduler, cfg.output_dir, step)

                    if step >= cfg.max_steps: break
            if step >= cfg.max_steps: break

        save_checkpoint_step(model, optimizer, scheduler, cfg.output_dir, step)
        print("[✓] Training complete.")

    except KeyboardInterrupt:
        print("[!] KeyboardInterrupt — saving checkpoint before exit.")
        save_checkpoint_step(model, optimizer, scheduler, cfg.output_dir, step)
    except Exception as e:
        traceback.print_exc()
        save_checkpoint_step(model, optimizer, scheduler, cfg.output_dir, step)
        raise
    finally:
        writer.close()
        pbar.close()

if __name__ == "__main__":
    cfg = ZiaConfig()
    train_loop(cfg)
