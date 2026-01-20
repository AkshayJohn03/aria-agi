#!/usr/bin/env python3
"""
train_zia_ift_v3_rope_alibi.py

Adds RoPE / ALiBi support and flexible loading. Also provides options
to reduce inference/eval overhead to speed up training on constrained GPUs.
"""
import os
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
import json
import math
import glob
import shutil
import random
import argparse
import hashlib
import traceback
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# ----------------------
# Config dataclass
# ----------------------
@dataclass
class V3Config:
    tokenizer_path: str = "artifacts/zia_tokenizer_60k"
    load_weights_only_from: Optional[str] = "artifacts/zia_dense_base/recovered_best.pt"
    resume_from: Optional[str] = None
    out_dir: str = "artifacts/zia_ift_v3_runs"
    project_root: str = "."
    use_alpaca: bool = True
    use_oasst: bool = True
    use_dolly: bool = True
    use_sharegpt: bool = False
    use_openorca: bool = True
    use_stack: bool = True
    use_indic_instruct: bool = True
    use_indic_parallel: bool = True
    alpaca_name: str = "yahma/alpaca-cleaned"
    oasst_name: str = "OpenAssistant/oasst1"
    dolly_name: str = "databricks/databricks-dolly-15k"
    openorca_name: str = "Open-Orca/OpenOrca"
    stack_name: str = "lvwerra/stack-exchange-paired"
    indic_instruct_name: str = "ai4bharat/indic-align"
    indic_parallel_name: str = "ai4bharat/samanantar"
    indic_parallel_lang: str = "ta"
    alpaca_subset: int = 10000
    oasst_subset: int = 10000
    dolly_subset: int = 5000
    openorca_subset: int = 10000
    stack_subset: int = 10000
    indic_instruct_subset: int = 10000
    indic_parallel_subset: int = 10000
    vocab_size: int = 60004
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    mlp_ratio: int = 4
    max_len: int = 512
    dropout: float = 0.1
    batch_size: int = 8
    grad_accum: int = 8
    lr: float = 3e-05
    weight_decay: float = 0.0
    max_steps: int = 20000
    warmup_steps: int = 200
    eval_every: int = 1000
    save_every: int = 1000
    infer_every: int = 5000
    keep_last: int = 5
    num_workers: int = 4
    fp16: bool = True
    seed: int = 42
    resume_optimizer: bool = False
    early_stop_patience: int = 6
    log_every: int = 100
    tb_comment: str = "zia_ift_v3_Rope"
    max_train_examples: Optional[int] = None
    min_tokens_in_example: int = 3
    # Positional type: 'absolute' (learned), 'rope' (rotary), 'alibi' (linear biases)
    pos_type: str = "absolute"
    # options to disable eval/infer during training for speed
    disable_inference: bool = False
    disable_eval: bool = False

# ----------------------
# Tiny model components
# ----------------------
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

# Rotary embeddings helper
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        # only supports even dim
        super().__init__()
        assert dim % 2 == 0, "rotary dim must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def get_sin_cos(self, seq_len, device, dtype):
        # returns sin and cos of shape (seq_len, dim)
        t = torch.arange(seq_len, device=device).type(self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # interleave trick for rotation
        sin = torch.sin(emb).to(dtype)
        cos = torch.cos(emb).to(dtype)
        return sin, cos

def apply_rotary(q, k, sin, cos):
    # q,k: (B, nheads, seqlen, head_dim)
    # sin,cos: (seqlen, head_dim)
    # Implementation uses interleaved dims: (d0,d1,d0,d1,...)
    # reshape for broadcast
    B, H, S, D = q.shape
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1,1,S,D)
    cos = cos.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

def rotate_half(x):
    # x: (..., D) where D is even: [x0,x1,x2,x3] -> [-x1, x0, -x3, x2]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x_rot

# ALiBi slopes helper
def get_alibi_slopes(n):
    # Implementation adapted from common ALiBi helper
    import math
    def get_slopes_power_of_two(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start ** i for i in range(n)]
    if math.log2(n).is_integer():
        return get_slopes_power_of_two(n)
    else:
        m = 2 ** math.floor(math.log2(n))
        slopes = get_slopes_power_of_two(m)
        extra = get_alibi_slopes(2 * m)[0::2][: n - m]
        return slopes + extra

# Manual multi-head self-attention to enable RoPE/ALiBi
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, max_len=512, pos_type="absolute"):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_len = max_len
        self.pos_type = pos_type

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        if pos_type == "rope":
            # RoPE requires half of head_dim to define rotation frequencies;
            # we let rotary dim = head_dim (common approach) but ensure even
            assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
            self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_len)
        else:
            self.rotary = None

        if pos_type == "alibi":
            slopes = torch.tensor(get_alibi_slopes(self.n_heads), dtype=torch.float32)
            self.register_buffer("alibi_slopes", slopes)  # (n_heads,)
        else:
            self.alibi_slopes = None

    def forward(self, x, key_padding_mask=None):
        # x: (B, T, D)
        B, T, D = x.size()
        qkv = self.qkv(x)  # (B, T, 3D)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, heads, T, head_dim)

        # Apply rotary if requested
        if self.pos_type == "rope":
            sin, cos = self.rotary.get_sin_cos(T, device=x.device, dtype=q.dtype)
            q, k = apply_rotary(q, k, sin, cos)

        # scaled dot-product
        # (B, heads, T, head_dim) @ (B, heads, head_dim, T) -> (B, heads, T, T)
        q_scaled = q / math.sqrt(self.head_dim)
        scores = torch.einsum("bhqd,bhkd->bhqk", q_scaled, k)

        # ALiBi bias
        if self.pos_type == "alibi":
            # slopes: (heads,)
            # create distances matrix: q_idx - k_idx
            # bias = -slope * (q_idx - k_idx)
            device = x.device
            q_pos = torch.arange(T, device=device).unsqueeze(1)  # (T,1)
            k_pos = torch.arange(T, device=device).unsqueeze(0)  # (1,T)
            distances = (q_pos - k_pos).to(torch.float32)  # (T,T)
            # slopes per head
            slopes = self.alibi_slopes.to(device=device).view(1, self.n_heads, 1, 1)  # (1,heads,1,1)
            # distances shape must become (1,1,T,T) then multiplied by slopes -> (1,heads,T,T)
            bias = -distances.view(1, 1, T, T) * slopes  # negative so further tokens get lower score
            scores = scores + bias.to(scores.dtype)

        # Causal mask: block future (k > q)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        if causal_mask.any():
            neginf = -1e9 if scores.dtype == torch.float32 else torch.finfo(scores.dtype).min / 2
            scores = scores.masked_fill(causal_mask.view(1, 1, T, T), neginf)

        # key padding mask: (B, T) bool where True means pad (we follow HF convention)
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) where 0/1? We expect bool where True indicates padding
            # Expand to (B, 1, 1, T)
            neginf = -1e9 if scores.dtype == torch.float32 else torch.finfo(scores.dtype).min / 2
            kpm = key_padding_mask.bool().view(B, 1, 1, T)
            scores = scores.masked_fill(kpm, neginf)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)  # (B,heads,T,head_dim)
        out = out.contiguous().view(B, T, D)
        out = self.out(out)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout, max_len, pos_type):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout, max_len=max_len, pos_type=pos_type)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mlp_ratio, dropout)

    def forward(self, x, attention_mask=None):
        # attention_mask: (B, T) where 0 means padding, 1 means keep
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        h = self.ln1(x)
        out = self.attn(h, key_padding_mask=key_padding_mask)
        x = x + out
        return x + self.ff(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, mlp_ratio, max_len, dropout, pad_token_id=0, pos_type="absolute"):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos_type = pos_type
        if pos_type == "absolute":
            self.pos = nn.Embedding(max_len, d_model)
        else:
            # no absolute positional embeddings when using RoPE/ALiBi
            self.register_buffer("dummy_pos", torch.zeros(1))
            self.pos = None
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, mlp_ratio, dropout, max_len, pos_type) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        # tie weights
        self.head.weight = self.tok.weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape
        if T > self.max_len:
            input_ids = input_ids[:, -self.max_len:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_len:]
            T = input_ids.size(1)
        pos_ids = None
        if self.pos_type == "absolute" and self.pos is not None:
            pos_ids = torch.arange(0, T, device=input_ids.device).unsqueeze(0).expand(B, T)
            x = self.tok(input_ids) + self.pos(pos_ids)
        else:
            x = self.tok(input_ids)  # RoPE/ALiBi will be applied in attention

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        for blk in self.blocks:
            x = blk(x, attention_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                 shift_labels.view(-1),
                                 ignore_index=self.pad_token_id)
            valid_len = (labels != self.pad_token_id).sum(dim=1).float().mean()
            penalty = torch.relu(10 - valid_len / 10) * 0.01
            ce = ce + penalty
            loss = ce
        return logits, loss

# ----------------------
# Helpers: save/load, flexible loader aware of pos_type
# ----------------------
def _atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_checkpoint(model, optimizer, scheduler, scaler, step, out_dir, keep_last=5, tag=None, metric=None):
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    name = f"step_{step:06d}.pt" if tag is None else f"{tag}_step_{step:06d}.pt"
    path = os.path.join(ckpt_dir, name)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "tag": tag or f"step_{step}",
        "metric": metric
    }
    _atomic_save(payload, path)
    step_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    if len(step_ckpts) > keep_last:
        for old in step_ckpts[:-keep_last]:
            try: os.remove(old)
            except Exception: pass
    print(f"[💾] Saved checkpoint {path}")
    return path

def save_best(model, optimizer, scheduler, scaler, step, out_dir, best_metric):
    best_dir = os.path.join(out_dir, "best_val")
    os.makedirs(best_dir, exist_ok=True)
    best_path = os.path.join(best_dir, "checkpoint.pt")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "best_metric": best_metric
    }
    _atomic_save(payload, best_path)
    ts = time.strftime("%Y%m%d-%H%M%S")
    copy_path = os.path.join(best_dir, f"best_{best_metric:.6f}_step{step}_{ts}.pt")
    _atomic_save({"model": model.state_dict(), "step": step, "metric": best_metric}, copy_path)
    print(f"[🏆] Saved best_val to {best_path} and copy {copy_path}")

def load_weights_flexibly(model: TinyGPT, ckpt_path: str, tokenizer: AutoTokenizer, device: torch.device, pos_type="absolute"):
    """
    Loads weights from ckpt_path into model, tolerating:
      - vocab (tok weight) size mismatch (copy overlap + init rest)
      - head.weight same handling
      - if pos_type != 'absolute', skip loading pos.weight from ckpt entirely
      - prints extra/missing keys
    """
    print(f"[i] Loading weights-only from {ckpt_path} (flexible loader, pos_type={pos_type})")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)

    sd = model.state_dict()

    # Token embeddings handle
    if "tok.weight" in state and "tok.weight" in sd:
        src = state["tok.weight"]
        dst = sd["tok.weight"]
        if src.size(0) != dst.size(0):
            print(f"[!] Vocab size mismatch: checkpoint {src.size(0)} vs model {dst.size(0)}. Copying overlap & init rest.")
            n_copy = min(src.size(0), dst.size(0))
            new_tok = dst.clone()
            new_tok[:n_copy].copy_(src[:n_copy])
            if new_tok.size(0) > n_copy:
                nn.init.normal_(new_tok[n_copy:], mean=0.0, std=0.02)
            state["tok.weight"] = new_tok
            # head weight mapping if present
            if "head.weight" in state and "head.weight" in sd:
                hw_src = state["head.weight"]
                hw_dst = sd["head.weight"]
                new_hw = hw_dst.clone()
                n_copy2 = min(hw_src.size(0), hw_dst.size(0))
                new_hw[:n_copy2].copy_(hw_src[:n_copy2])
                if new_hw.size(0) > n_copy2:
                    nn.init.normal_(new_hw[n_copy2:], mean=0.0, std=0.02)
                state["head.weight"] = new_hw

    # Positional embeddings: skip if model expects RoPE/ALiBi
    if pos_type != "absolute":
        if "pos.weight" in state:
            print("[i] Skipping absolute pos.weight from checkpoint because model uses RoPE/ALiBi.")
            state.pop("pos.weight", None)

    # Debug keys
    sd_keys = set(sd.keys())
    ckpt_keys = set(state.keys())
    extra_in_ckpt = sorted(list(ckpt_keys - sd_keys))
    missing_in_ckpt = sorted(list(sd_keys - ckpt_keys))
    print("extra_in_ckpt (first 10):", extra_in_ckpt[:10])
    print("missing_in_ckpt (first 10):", missing_in_ckpt[:10])

    # Filter to allowed keys
    filtered = {k: v for k, v in state.items() if k in sd_keys}
    model.load_state_dict(filtered, strict=False)
    # re-tie head to tok
    model.head.weight = model.tok.weight
    print("[i] Weights loaded (flexible).")

# ----------------------
# Dataset formatting & cleaning helpers (unchanged from your version, simplified)
# ----------------------
def _format_alpaca(ex):
    inst = ex.get("instruction", "") or ""
    inp = ex.get("input", "") or ""
    out = ex.get("output") or ex.get("response") or ""
    prompt = inst
    if inp and len(inp.strip()) > 0:
        prompt = f"{inst}\n{inp}"
    text = f"### Instruction:\n{prompt}\n\n### Response:\n{out}"
    return {"text": text}

# Keep your other dataset formatters same (oasst/dolly/openorca/stack/indic etc.)
# For brevity include only ones used in earlier runs (you can paste in the full ones)
def _format_oasst(ex):
    if "messages" in ex:
        msgs = ex["messages"]
        assistant_parts = []
        prompt_parts = []
        for m in msgs:
            if not isinstance(m, dict): continue
            role = m.get("role", "")
            content = m.get("content") or m.get("text") or ""
            if role == "assistant":
                assistant_parts.append(content)
            else:
                prompt_parts.append(content)
        prompt = "\n".join(prompt_parts).strip()
        out = "\n".join(assistant_parts).strip()
        text = f"### Instruction:\n{prompt}\n\n### Response:\n{out}"
    else:
        prompt = ex.get("input") or ex.get("prompt") or ex.get("instruction") or ""
        out = ex.get("response") or ex.get("text") or ex.get("output") or ""
        text = f"### Instruction:\n{prompt}\n\n### Response:\n{out}"
    return {"text": text}

def _format_dolly(ex):
    instr = ex.get("instruction") or ex.get("title") or ""
    out = ex.get("text") or ex.get("response") or ex.get("output") or ""
    text = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
    return {"text": text}

def _format_openorca(ex):
    sys_prompt = ex.get("system_prompt", "") or ""
    question = ex.get("question", "") or ""
    response = ex.get("response", "") or ""
    prompt_parts = []
    if sys_prompt:
        prompt_parts.append(f"[SYSTEM] {sys_prompt}")
    prompt_parts.append(question)
    prompt = "\n".join(prompt_parts).strip()
    text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
    return {"text": text}

def _format_stack(ex):
    question = ex.get("question", "") or ""
    response = ex.get("response_j", "") or ""
    text = f"### Instruction:\n{question}\n\n### Response:\n{response}"
    return {"text": text}

def _format_indic_instruct(ex):
    if "interactions" in ex:
        interactions = ex["interactions"]
        if interactions and len(interactions) > 0:
            if isinstance(interactions[0], list) and len(interactions[0]) >= 2:
                prompt = interactions[0][0]
                response = interactions[0][1]
                text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
                return {"text": text}
    prompt = ex.get("prompt") or ex.get("instruction") or ""
    response = ex.get("completion") or ex.get("response") or ex.get("output") or ""
    if prompt and response:
        text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
        return {"text": text}
    text = ex.get("text") or str(ex)
    return {"text": text}

def _format_samanantar(ex, lang_code: str):
    # try both schemas (some versions have 'src'/'tgt')
    en_src = ex.get("en") or ex.get("src") or ""
    target_tgt = ex.get(lang_code) or ex.get("tgt") or ""
    prompt = f"Translate the following English sentence into the language with code '{lang_code}':\n{en_src}"
    text = f"### Instruction:\n{prompt}\n\n### Response:\n{target_tgt}"
    return {"text": text}

def is_english_like(s: str) -> bool:
    if not s or not isinstance(s, str): return False
    letters = sum(1 for c in s if c.isalpha())
    if letters == 0: return False
    ascii_letters = sum(1 for c in s if ord(c) < 128 and c.isalpha())
    return (ascii_letters / letters) > 0.5

def prepare_merged_dataset(cfg: V3Config):
    ds_list = []
    load_kwargs = {}
    ZIA_PREFIX = "You are ZIA, a reasoning assistant created by Akshay. You think clearly, respond naturally, and prefer factual, conversational explanations. Avoid being verbose or robotic.\n"

    def _add_prefix(ex):
        ex_text = ex.get("text", "")
        if not ex_text.startswith("### Instruction") and ZIA_PREFIX not in ex_text:
             return {"text": ZIA_PREFIX + ex_text}
        return ex

    if cfg.use_alpaca:
        try:
            n = cfg.alpaca_subset if cfg.alpaca_subset>0 else ""
            split = f"train[:{n}]" if n else "train"
            print(f"[i] Loading Alpaca ({split}) ...")
            d = load_dataset(cfg.alpaca_name, split=split, **load_kwargs)
            d = d.map(lambda ex: _format_alpaca(ex), remove_columns=d.column_names if hasattr(d, "column_names") else None)
            d = d.map(_add_prefix)
            ds_list.append(d)
        except Exception as e:
            print("[!] Alpaca load failed:", e)

    if cfg.use_oasst:
        try:
            n = cfg.oasst_subset if cfg.oasst_subset>0 else ""
            split = f"train[:{n}]" if n else "train"
            print(f"[i] Loading OASST ({split}) ...")
            d = load_dataset(cfg.oasst_name, split=split, **load_kwargs)
            d = d.map(lambda ex: _format_oasst(ex), remove_columns=d.column_names if hasattr(d, "column_names") else None)
            d = d.map(_add_prefix)
            ds_list.append(d)
        except Exception as e:
            print("[!] OASST load failed:", e)

    if cfg.use_dolly:
        try:
            n = cfg.dolly_subset if cfg.dolly_subset>0 else ""
            split = f"train[:{n}]" if n else "train"
            print(f"[i] Loading Dolly ({split}) ...")
            d = load_dataset(cfg.dolly_name, split=split, **load_kwargs)
            d = d.map(lambda ex: _format_dolly(ex), remove_columns=d.column_names if hasattr(d, "column_names") else None)
            d = d.map(_add_prefix)
            ds_list.append(d)
        except Exception as e:
            print("[!] Dolly load failed:", e)

    if cfg.use_openorca:
        try:
            n = cfg.openorca_subset if cfg.openorca_subset > 0 else ""
            split = f"train[:{n}]" if n else "train"
            print(f"[i] Loading OpenOrca ({split}) ...")
            d = load_dataset(cfg.openorca_name, split=split, token=True, **load_kwargs)
            d = d.map(lambda ex: _format_openorca(ex), remove_columns=d.column_names if hasattr(d, "column_names") else None)
            d = d.map(_add_prefix)
            ds_list.append(d)
        except Exception as e:
            print(f"[!] OpenOrca load failed: {e}")

    if cfg.use_stack:
        try:
            n = cfg.stack_subset if cfg.stack_subset > 0 else ""
            split = f"train[:{n}]" if n else "train"
            print(f"[i] Loading Stack Exchange Paired ({split})...")
            d = load_dataset(cfg.stack_name, split=split, **load_kwargs)
            d = d.map(lambda ex: _format_stack(ex), remove_columns=d.column_names if hasattr(d, "column_names") else None)
            d = d.map(_add_prefix)
            ds_list.append(d)
        except Exception as e:
            print("[!] Stack Exchange load failed:", e)

    if cfg.use_indic_instruct:
        try:
            n = cfg.indic_instruct_subset if cfg.indic_instruct_subset > 0 else ""
            split = f"train[:{n}]" if n else "train"
            CONFIG_NAME = 'Indic_ShareLlama'
            print(f"[i] Loading Indic Align Instruct (Config: {CONFIG_NAME}, {split}) ...")
            d = load_dataset(cfg.indic_instruct_name, CONFIG_NAME, split=split, **load_kwargs)
            d = d.map(lambda ex: _format_indic_instruct(ex), remove_columns=d.column_names if hasattr(d, "column_names") else None)
            d = d.map(_add_prefix)
            ds_list.append(d)
        except Exception as e:
            print("[!] Indic Align Instruct load failed:", e)

    if cfg.use_indic_parallel:
        try:
            n = cfg.indic_parallel_subset if cfg.indic_parallel_subset > 0 else ""
            split = f"train[:{n}]" if n else "train"
            CONFIG_NAME = cfg.indic_parallel_lang
            print(f"[i] Loading Samanantar Parallel (Config: {CONFIG_NAME}, {split}) ...")
            d = load_dataset(cfg.indic_parallel_name, CONFIG_NAME, split=split, **load_kwargs)
            # robust mapping: try possible column names
            d = d.map(lambda ex: _format_samanantar(ex, CONFIG_NAME), remove_columns=d.column_names if hasattr(d, "column_names") else None)
            d = d.map(_add_prefix)
            ds_list.append(d)
        except Exception as e:
            print(f"[!] Samanantar Parallel load failed:", e)
            print(f"    Check config/lang code.")

    if not ds_list:
        raise RuntimeError("No datasets could be loaded. Enable at least one dataset.")

    merged = concatenate_datasets(ds_list) if len(ds_list) > 1 else ds_list[0]

    def clean_fn(ex):
        txt = (ex.get("text") or "").strip()
        if not txt: return {"text": ""}
        if len(txt.split()) < 3: return {"text": ""}
        return {"text": txt}
    merged = merged.map(clean_fn)

    def has_valid_response(ex):
        txt = ex.get("text") or ""
        if "### Response:" not in txt: return False
        resp = txt.split("### Response:")[-1].strip()
        return len(resp.split()) >= 3
    merged = merged.filter(has_valid_response)

    merged = merged.map(lambda ex: {"is_english_like": bool(is_english_like(ex.get("text","")))})
    try:
        merged = merged.sort("is_english_like", reverse=True)
    except Exception:
        pass

    def add_hash(ex):
        h = hashlib.sha1((ex.get("text") or "").encode("utf-8")).hexdigest()
        return {"hash": h}
    merged = merged.map(add_hash, remove_columns=[])

    seen = set()
    keep_indices = []
    for i, rec in enumerate(merged):
        h = rec["hash"]
        if h in seen: continue
        seen.add(h)
        keep_indices.append(i)
    merged = merged.select(keep_indices)

    print(f"[i] Merged dataset size after dedupe/filter: {len(merged)}")
    return merged

# ----------------------
# Collator
# ----------------------
class Collator:
    def __init__(self, tokenizer: AutoTokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len
    def __call__(self, batch):
        texts = [ (b["text"] if isinstance(b, dict) else str(b)) for b in batch ]
        eos = self.tok.eos_token or "</s>"
        texts = [ (t if t.strip().endswith(eos) else t + eos) for t in texts ]
        enc = self.tok(texts, truncation=True, padding=True, max_length=self.max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

# ----------------------
# Evaluate & generate helpers
# ----------------------
@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    pad = model.pad_token_id if hasattr(model, "pad_token_id") else 0
    losses = []
    total = 0
    for i, (ids, mask) in enumerate(loader, start=1):
        ids = ids.to(device)
        mask = mask.to(device)
        logits, _ = model(ids, attention_mask=mask, labels=ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), ignore_index=pad)
        losses.append(loss.item())
        total += 1
        if max_batches and i >= max_batches:
            break
    model.train()
    return sum(losses)/max(1, total)

@torch.no_grad()
def sample_generate(model, tokenizer, prompt: str, device, max_new_tokens=128, temperature=0.8, top_k=50):
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.max_len).to(device)
    input_ids = enc["input_ids"]
    cur = input_ids
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        attention_mask = (cur != model.pad_token_id).long()
        logits, _ = model(cur, attention_mask=attention_mask)
        next_logits = logits[:, -1, :] / max(temperature, 1e-8)
        k = min(top_k, next_logits.size(-1))
        k = max(1, k)
        top_vals, top_idx = torch.topk(next_logits, k, dim=-1)
        probs = torch.softmax(top_vals, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        next_token = torch.gather(top_idx, 1, idx)
        if eos_id is not None and cur.size(1) > 5 and (next_token == eos_id).all():
            break
        cur = torch.cat([cur, next_token], dim=1)
        if cur.size(1) > model.max_len:
            cur = cur[:, -model.max_len:]
    out = cur[:, input_ids.size(1):]
    txt = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    model.train()
    return txt

def simple_hallucination_filter(text: str) -> bool:
    if not text or not isinstance(text, str): return False
    t = text.strip()
    return len(t.split()) > 1

# ----------------------
# Training loop
# ----------------------
def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[i] model param count: total={tot:,}  trainable={trainable:,}")

def train(cfg: V3Config):
    if os.name == "nt":
        print("[i] Windows detected -> lowering num_workers & disabling mmap.")
        cfg.num_workers = min(2, cfg.num_workers)
        os.environ["HF_DATASETS_DISABLE_MMAP"] = "1"

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Device: {device} | CUDA available: {torch.cuda.is_available()}")

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path, use_fast=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token":"<pad>"})
    if tok.eos_token is None:
        tok.add_special_tokens({"eos_token":"<eos>"})
    cfg.vocab_size = len(tok)
    print(f"[i] Tokenizer: vocab={len(tok)} pad={tok.pad_token_id} eos={tok.eos_token_id}")

    assistant_preamble = (
        "You are ZIA, a reasoning assistant created by Akshay. "
        "You think clearly, respond naturally, and prefer factual, conversational explanations. "
        "Avoid being verbose or robotic. "
    )

    merged = prepare_merged_dataset(cfg)
    if cfg.max_train_examples:
        merged = merged.select(range(min(cfg.max_train_examples, len(merged))))
    splits = merged.train_test_split(test_size=0.02, seed=cfg.seed)
    train_ds = splits["train"]
    val_ds = splits["test"]
    print(f"[i] train={len(train_ds)} val={len(val_ds)}")

    collate = Collator(tok, cfg.max_len)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=pin,
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=min(2, cfg.num_workers), pin_memory=pin,
                            collate_fn=collate)

    model = TinyGPT(cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio, cfg.max_len, cfg.dropout, pad_token_id=tok.pad_token_id, pos_type=cfg.pos_type).to(device)
    model.pad_token_id = tok.pad_token_id
    count_params(model)

    start_step = 0
    if cfg.load_weights_only_from and os.path.exists(cfg.load_weights_only_from):
        try:
            load_weights_flexibly(model, cfg.load_weights_only_from, tok, device, pos_type=cfg.pos_type)
        except Exception as e:
            print("[!] Flexible weight load failed, trying strict load:", e)
            ckpt = torch.load(cfg.load_weights_only_from, map_location=device)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)

    if cfg.resume_from and os.path.exists(cfg.resume_from):
        try:
            print(f"[i] Resuming from checkpoint {cfg.resume_from}")
            ckpt = torch.load(cfg.resume_from, map_location=device)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
                start_step = ckpt.get("step", 0)
            else:
                model.load_state_dict(ckpt, strict=False)
                start_step = ckpt.get("step", 0) if isinstance(ckpt, dict) else 0
        except Exception as e:
            print("[!] Resume from failed:", e)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.max_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler() if (cfg.fp16 and torch.cuda.is_available()) else None

    if cfg.resume_from and cfg.resume_optimizer and os.path.exists(cfg.resume_from):
        try:
            ckpt = torch.load(cfg.resume_from, map_location=device)
            if "optimizer" in ckpt and ckpt["optimizer"]:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt and ckpt["scheduler"]:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt and ckpt["scaler"] and scaler is not None:
                scaler.load_state_dict(ckpt["scaler"])
            start_step = ckpt.get("step", start_step)
            print(f"[i] Resumed optimizer/scheduler/scaler from checkpoint at step {start_step}")
        except Exception as e:
            print("[!] Failed to resume optimizer/scheduler/scaler:", e)

    os.makedirs(cfg.out_dir, exist_ok=True)
    tok.save_pretrained(os.path.join(cfg.out_dir, "tokenizer"))
    writer = SummaryWriter(os.path.join(cfg.out_dir, "runs"), comment=cfg.tb_comment)

    step = int(start_step)
    micro_steps = 0
    running_loss = 0.0
    best_val = float("inf")
    early_stop_counter = 0
    grad_norm = 0.0

    global_pbar = tqdm(total=cfg.max_steps, desc="Training", unit="step")
    global_pbar.update(step)

    try:
        model.train()
        while step < cfg.max_steps:
            for ids, mask in train_loader:
                ids = ids.to(device)
                mask = mask.to(device)
                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=(scaler is not None)):
                    logits, loss = model(ids, attention_mask=mask, labels=ids)
                    if loss is None:
                        continue
                    loss = loss / cfg.grad_accum
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                micro_steps += 1
                running_loss += loss.item() * cfg.grad_accum
                if micro_steps % cfg.grad_accum == 0:
                    step += 1
                    if scaler:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    if step % cfg.log_every == 0:
                        avg_loss = running_loss / cfg.log_every if cfg.log_every>0 else running_loss
                        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                        lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else cfg.lr
                        writer.add_scalar("train/loss", avg_loss, step)
                        writer.add_scalar("train/cross_entropy", avg_loss, step)
                        writer.add_scalar("train/ppl", math.exp(min(avg_loss, 20)), step)
                        writer.add_scalar("train/lr", lr_now, step)
                        writer.add_scalar("train/grad_norm", grad_norm, step)
                        print(f"[{time.strftime('%H:%M:%S')}] step={step} avg_loss={avg_loss:.4f} ppl={ppl:.2f} lr={lr_now:.2e} grad_norm={grad_norm:.4f}")
                        running_loss = 0.0

                    if (not cfg.disable_eval) and (step % cfg.eval_every == 0):
                        val_loss = evaluate(model, val_loader, device, max_batches=200)
                        val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                        writer.add_scalar("eval/loss", val_loss, step)
                        writer.add_scalar("eval/ppl", val_ppl, step)
                        print(f"[{time.strftime('%H:%M:%S')}] [Eval] step={step} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
                        if val_loss < best_val:
                            best_val = val_loss
                            save_best(model, optimizer, scheduler, scaler, step, cfg.out_dir, best_val)
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1

                        if (not cfg.disable_inference) and (step % cfg.infer_every == 0):
                            try:
                                sample_indices = list(range(len(val_ds)))
                                random.shuffle(sample_indices)
                                printed = 0
                                for idx in sample_indices:
                                    if printed >= 5: break
                                    s = val_ds[idx]["text"]
                                    prompt = assistant_preamble + (s.split("### Response:")[0][:300] if "### Response:" in s else s[:300])
                                    generated = sample_generate(model, tok, prompt, device, max_new_tokens=128, temperature=0.8, top_k=50)
                                    ok = simple_hallucination_filter(generated)
                                    print(f"[Infer sample {printed}] Prompt (truncated): {prompt!s}\n-> {generated}\nOK={ok}\n")
                                    writer.add_text(f"infer/sample_{printed}", f"PROMPT: {prompt}\nGENERATED: {generated}\nOK={ok}", step)
                                    printed += 1
                            except Exception as e:
                                print("[!] inference sampling failed:", e)

                        if cfg.early_stop_patience and early_stop_counter >= cfg.early_stop_patience:
                            print(f"[!] Early stopping triggered (no val improvement in {cfg.early_stop_patience} evals).")
                            save_checkpoint(model, optimizer, scheduler, scaler, step, cfg.out_dir, keep_last=cfg.keep_last, tag="earlystop", metric=best_val)
                            raise KeyboardInterrupt("Early stopping")

                    if step % cfg.save_every == 0:
                        save_checkpoint(model, optimizer, scheduler, scaler, step, cfg.out_dir, keep_last=cfg.keep_last, metric=best_val)
                        print(f"[⏺] Checkpoint saved at step {step}")
                    global_pbar.update(1)

                if step >= cfg.max_steps:
                    break
            if step >= cfg.max_steps:
                break

        save_checkpoint(model, optimizer, scheduler, scaler, step, cfg.out_dir, keep_last=cfg.keep_last, tag="final", metric=best_val)
        print("[✓] Training finished. Final checkpoint saved.")
    except KeyboardInterrupt:
        print("[!] KeyboardInterrupt — saving interrupt checkpoint...")
        save_checkpoint(model, optimizer, scheduler, scaler, step, cfg.out_dir, tag="interrupt", metric=best_val)
    except Exception as e:
        print("[!] Exception during training — saving error checkpoint...")
        traceback.print_exc()
        save_checkpoint(model, optimizer, scheduler, scaler, step, cfg.out_dir, tag="error", metric=best_val)
        raise
    finally:
        writer.close()
        global_pbar.close()

# ----------------------
# CLI
# ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--load_weights_only_from", type=str, default=None)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--alpaca_subset", type=int, default=None)
    p.add_argument("--oasst_subset", type=int, default=None)
    p.add_argument("--dolly_subset", type=int, default=None)
    p.add_argument("--use_openorca", action="store_true")
    p.add_argument("--use_stack", action="store_true")
    p.add_argument("--use_indic_instruct", action="store_true")
    p.add_argument("--use_indic_parallel", action="store_true")
    p.add_argument("--openorca_subset", type=int, default=None)
    p.add_argument("--stack_subset", type=int, default=None)
    p.add_argument("--indic_instruct_subset", type=int, default=None)
    p.add_argument("--indic_parallel_subset", type=int, default=None)
    p.add_argument("--indic_parallel_lang", type=str, default=None)
    p.add_argument("--eval_every", type=int, default=None)
    p.add_argument("--infer_every", type=int, default=None)
    p.add_argument("--resume_optimizer", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--early_stop_patience", type=int, default=None)
    p.add_argument("--pos_type", type=str, default=None, choices=["absolute", "rope", "alibi"], help="Positional scheme")
    p.add_argument("--disable_inference", action="store_true", help="Disable inference during eval to speed up runs")
    p.add_argument("--disable_eval", action="store_true", help="Disable evaluation (not recommended for final runs)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = V3Config()
    if args.tokenizer_path: cfg.tokenizer_path = args.tokenizer_path
    if args.load_weights_only_from: cfg.load_weights_only_from = args.load_weights_only_from
    if args.resume_from: cfg.resume_from = args.resume_from
    if args.out_dir: cfg.out_dir = args.out_dir
    if args.max_steps: cfg.max_steps = args.max_steps
    if args.alpaca_subset is not None: cfg.alpaca_subset = args.alpaca_subset
    if args.oasst_subset is not None: cfg.oasst_subset = args.oasst_subset
    if args.dolly_subset is not None: cfg.dolly_subset = args.dolly_subset
    if args.use_openorca: cfg.use_openorca = True
    if args.use_stack: cfg.use_stack = True
    if args.use_indic_instruct: cfg.use_indic_instruct = True
    if args.use_indic_parallel: cfg.use_indic_parallel = True
    if args.openorca_subset is not None: cfg.openorca_subset = args.openorca_subset
    if args.stack_subset is not None: cfg.stack_subset = args.stack_subset
    if args.indic_instruct_subset is not None: cfg.indic_instruct_subset = args.indic_instruct_subset
    if args.indic_parallel_subset is not None: cfg.indic_parallel_subset = args.indic_parallel_subset
    if args.indic_parallel_lang: cfg.indic_parallel_lang = args.indic_parallel_lang
    if args.eval_every is not None: cfg.eval_every = args.eval_every
    if args.infer_every is not None: cfg.infer_every = args.infer_every
    if args.resume_optimizer: cfg.resume_optimizer = True
    if args.fp16: cfg.fp16 = True
    if args.early_stop_patience is not None: cfg.early_stop_patience = args.early_stop_patience
    if args.pos_type: cfg.pos_type = args.pos_type
    if args.disable_inference: cfg.disable_inference = True
    if args.disable_eval: cfg.disable_eval = True
    print("[i] Training config:\n", cfg)
    train(cfg)
