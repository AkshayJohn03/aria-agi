#!/usr/bin/env python3
# train_zia_fast.py
# Fast training loop for Zia — optimized for speed (~2 it/s).
# Keeps checkpoints, resumability, AMP, grad accumulation, and TensorBoard logging.

import os, glob, time, math, traceback
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk, concatenate_datasets, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
@dataclass
class ZiaConfig:
    max_len: int = 256
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1

    dataset_path: str = "my_datasets/processed/arrow_cleaned_v1"
    tokenizer_path: str = "artifacts/zia_tokenizer_60k"
    output_dir: str = "artifacts/zia_dense_runs"

    batch_size: int = 8
    grad_accum: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 3000
    max_steps: int = 100000
    log_train_every_batches: int = 200
    eval_every_batches: int = 5000
    save_every_batches: int = 1000
    keep_last: int = 2

    num_workers: int = 4
    fp16: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    resume_from_best: bool = False
    load_initial_weights_from_best: bool = True


# -------------------------
# Model (TinyGPT core)
# -------------------------
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
        B, T, _ = h.size()
        causal = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(h, h, h, attn_mask=causal, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + out
        return x + self.ff(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, mlp_ratio, max_len, dropout):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.pad_token_id = 0
        self.head.weight = self.tok.weight  # weight tying

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
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id
            )
        return logits, loss


# -------------------------
# Dataset + Collator
# -------------------------
class CollateWrapper:
    def __init__(self, tokenizer, eos_token: str, max_len: int):
        self.tok = tokenizer
        self.eos_token = eos_token
        self.max_len = max_len

    def _sample_to_text(self, ex):
        if isinstance(ex, dict):
            if "text" in ex and isinstance(ex["text"], str):
                return ex["text"]
            if "content" in ex and isinstance(ex["content"], str):
                return ex["content"]
            if "messages" in ex and isinstance(ex["messages"], list):
                parts = []
                for m in ex["messages"]:
                    role = m.get("role", "")
                    content = m.get("content") or m.get("text") or ""
                    parts.append(f"{role}: {content}")
                return "\n".join(parts)
        return str(ex)

    def __call__(self, batch):
        texts = []
        for ex in batch:
            t = self._sample_to_text(ex)
            if self.eos_token and not t.endswith(self.eos_token):
                t += self.eos_token
            texts.append(t)
        enc = self.tok(texts, truncation=True, padding=True, max_length=self.max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]


def load_split(split_dir: str) -> Optional[Dataset]:
    if not os.path.exists(split_dir):
        return None
    if os.path.exists(os.path.join(split_dir, "dataset_info.json")):
        return load_from_disk(split_dir)
    parts = [os.path.join(split_dir, d) for d in os.listdir(split_dir)]
    datasets = []
    for p in parts:
        try:
            datasets.append(load_from_disk(p))
        except: pass
    return concatenate_datasets(datasets) if datasets else None


# -------------------------
# Checkpoint helpers
# -------------------------
def _atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_checkpoint(model, optimizer, scheduler, scaler, step, out_dir, keep_last, tag=None):
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    name = f"step_{step}.pt" if not tag else f"{tag}_{step}.pt"
    path = os.path.join(ckpt_dir, name)
    _atomic_save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
    }, path)
    print(f"[💾] Saved {path}")
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    if len(ckpts) > keep_last:
        for old in ckpts[:-keep_last]:
            os.remove(old)


def save_best_val(model, step, out_dir):
    path = os.path.join(out_dir, "best_val", "checkpoint.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_save({"model": model.state_dict(), "step": step}, path)
    print(f"[🏆] Best validation saved at step {step}")


def load_model_only(model, device, out_dir):
    path = os.path.join(out_dir, "best_val", "checkpoint.pt")
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[i] Loaded best_val weights from step {ckpt.get('step', 0)}")
        return True
    return False


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(model, data_loader, device, max_batches=None):
    model.eval()
    total_loss, n_batches = 0.0, 0
    pad = model.pad_token_id
    for i, (ids, mask) in enumerate(data_loader, 1):
        ids, mask = ids.to(device), mask.to(device)
        logits, _ = model(ids, attention_mask=mask, labels=ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               ids[:, 1:].reshape(-1),
                               ignore_index=pad)
        total_loss += loss.item()
        n_batches += 1
        if max_batches and i >= max_batches:
            break
    model.train()
    return total_loss / max(1, n_batches)


# -------------------------
# Training Loop
# -------------------------
def train_loop(cfg: ZiaConfig):
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    print(f"[i] Training on {device}")

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    eos_token = tok.eos_token or "</s>"

    # Load dataset
    train_ds = load_split(os.path.join(cfg.dataset_path, "train"))
    val_ds = load_split(os.path.join(cfg.dataset_path, "validation")) or load_split(os.path.join(cfg.dataset_path, "test"))
    if val_ds is None:
        split = train_ds.train_test_split(test_size=0.01, seed=cfg.seed)
        train_ds, val_ds = split["train"], split["test"]

    collate = CollateWrapper(tok, eos_token, cfg.max_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=min(4, cfg.num_workers), pin_memory=True,
                            collate_fn=collate)

    model = TinyGPT(len(tok), cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio,
                    cfg.max_len, cfg.dropout).to(device)
    model.pad_token_id = tok.pad_token_id

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, cfg.warmup_steps, cfg.max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    writer = SummaryWriter(os.path.join(cfg.output_dir, "runs"))

    if cfg.load_initial_weights_from_best:
        load_model_only(model, device, cfg.output_dir)

    step, micro, raw_loss, tokens_seen, best_val = 0, 0, 0.0, 0, float("inf")
    t0 = time.time()
    pbar = tqdm(total=cfg.max_steps * cfg.grad_accum, desc="Training", dynamic_ncols=True)

    try:
        while step < cfg.max_steps:
            for ids, mask in train_loader:
                ids, mask = ids.to(device), mask.to(device)
                with torch.cuda.amp.autocast(enabled=cfg.fp16):
                    _, loss = model(ids, attention_mask=mask, labels=ids)

                if torch.isnan(loss) or torch.isinf(loss):
                    print("[!] Bad loss detected, skipping batch")
                    continue

                raw_loss += loss.item()
                tokens_seen += ids.numel()

                loss = loss / cfg.grad_accum
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                micro += 1
                pbar.update(1)

                if micro % cfg.grad_accum == 0:
                    step += 1
                    if scaler:
                        scaler.unscale_(opt)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()

                    if step % (cfg.log_train_every_batches // cfg.grad_accum) == 0:
                        avg_loss = raw_loss / cfg.log_train_every_batches
                        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                        elapsed = time.time() - t0
                        tps = tokens_seen / max(1e-6, elapsed)
                        writer.add_scalar("train/loss", avg_loss, step)
                        writer.add_scalar("train/ppl", ppl, step)
                        writer.add_scalar("train/grad_norm", grad_norm, step)
                        writer.add_scalar("sys/tokens_per_sec", tps, step)
                        print(f"[{time.strftime('%H:%M:%S')}] step={step} loss={avg_loss:.4f} ppl={ppl:.2f}")
                        raw_loss, tokens_seen, t0 = 0.0, 0, time.time()

                    if step % (cfg.eval_every_batches // cfg.grad_accum) == 0:
                        val_loss = evaluate(model, val_loader, device, max_batches=200)
                        val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                        writer.add_scalar("eval/loss", val_loss, step)
                        writer.add_scalar("eval/ppl", val_ppl, step)
                        print(f"[Eval] step={step} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
                        if val_loss < best_val:
                            best_val = val_loss
                            save_best_val(model, step, cfg.output_dir)

                    if step % (cfg.save_every_batches // cfg.grad_accum) == 0:
                        save_checkpoint(model, opt, sched, scaler, step, cfg.output_dir, cfg.keep_last)

            if step >= cfg.max_steps:
                break

        save_checkpoint(model, opt, sched, scaler, step, cfg.output_dir, cfg.keep_last, tag="final")
        print("[✓] Training complete")

    except KeyboardInterrupt:
        save_checkpoint(model, opt, sched, scaler, step, cfg.output_dir, cfg.keep_last, tag="interrupt")
    except Exception:
        traceback.print_exc()
        save_checkpoint(model, opt, sched, scaler, step, cfg.output_dir, cfg.keep_last, tag="error")
    finally:
        writer.close()
        pbar.close()


if __name__ == "__main__":
    cfg = ZiaConfig()
    train_loop(cfg)
