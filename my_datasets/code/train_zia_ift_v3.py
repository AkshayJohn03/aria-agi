#!/usr/bin/env python3
"""
train_zia_ift_v3.py

Comprehensive, resumable Instruction Fine-Tuning (IFT) script with:
 - dataset download+clean+dedupe (saved under project folder)
 - tokenizer compatibility and flexible weight loading (handles vocab mismatch & pos padding)
 - FP16 (AMP) training, grad-accum, grad-clipping, LR scheduler
 - timestamped/resumable checkpoints, best_val timestamped copies
 - TensorBoard logging (loss, cross-entropy, ppl, LR, grad-norm)
 - tqdm progress bars, inference prompt rotation, hallucination filter
 - early stopping based on validation loss
 - safer and more informative flexible checkpoint loading
"""
import os
# --- Silence TRANSFORMERS_CACHE warning & set cache dirs early ---
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
# On some Windows builds this option might not be supported; we leave it but expect a warning if unsupported.
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
    # IO / paths
    tokenizer_path: str = "artifacts/zia_tokenizer_60k"
    load_weights_only_from: Optional[str] = "artifacts/zia_dense_base/recovered_best.pt"
    resume_from: Optional[str] = None   # full checkpoint to resume (includes optimizer, scaler)
    out_dir: str = "artifacts/zia_ift_v3_runs"
    project_root: str = "."  # used to force dataset cache under project

    # dataset flags
    use_alpaca: bool = True
    use_oasst: bool = True
    use_dolly: bool = True
    use_sharegpt: bool = False

    # New English datasets (updated from original file)
    use_openorca: bool = True
    use_stack: bool = True

    # Indic language datasets (updated from original file)
    use_indic_instruct: bool = True
    use_indic_parallel: bool = True
    use_indic: bool = False # Legacy/deprecated flag. Use _instruct instead.

    # dataset names
    alpaca_name: str = "yahma/alpaca-cleaned"
    oasst_name: str = "OpenAssistant/oasst1"
    dolly_name: str = "databricks/databricks-dolly-15k"
    openorca_name: str = "Open-Orca/OpenOrca"
    stack_name: str = "lvwerra/stack-exchange-paired"
    indic_instruct_name: str = "ai4bharat/indic-align"
    indic_parallel_name: str = "ai4bharat/samanantar"
    indic_parallel_lang: str = "ta" # Target language code for Samanantar (e.g., 'ta' for Tamil)

    # subset limits
    alpaca_subset: int = 10000
    oasst_subset: int = 10000
    dolly_subset: int = 5000
    openorca_subset: int = 10000
    stack_subset: int = 10000
    indic_instruct_subset: int = 10000
    indic_parallel_subset: int = 10000
    indic_subset: int = 0 # Legacy limit

    # model architecture (match your prior Small Dense)
    vocab_size: int = 60004 # will be adjusted to tokenizer size at runtime
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    mlp_ratio: int = 4
    max_len: int = 512
    dropout: float = 0.1

    # training
    batch_size: int = 8     # micro-batch
    grad_accum: int = 8
    lr: float = 3e-05
    weight_decay: float = 0.0
    max_steps: int = 20000
    warmup_steps: int = 200
    eval_every: int = 500
    save_every: int = 500
    infer_every: int = 500
    keep_last: int = 5
    num_workers: int = 4
    fp16: bool = True
    seed: int = 42
    resume_optimizer: bool = False
    early_stop_patience: int = 6
    log_every: int = 100
    tb_comment: str = "zia_ift_v3"
    max_train_examples: Optional[int] = None  # None => use all
    min_tokens_in_example: int = 3

# ----------------------
# Model (TinyGPT)
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
        # Causal mask: upper triangle is True (attention blocked)
        causal = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(h, h, h, attn_mask=causal, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + out
        return x + self.ff(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, mlp_ratio, max_len, dropout, pad_token_id=0):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])
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

            # --- PATCH: Light response length reward / anti-collapse regularizer ---
            ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1),
                                   ignore_index=self.pad_token_id)
            valid_len = (labels != self.pad_token_id).sum(dim=1).float().mean()
            penalty = torch.relu(10 - valid_len / 10) * 0.01
            ce = ce + penalty
            loss = ce
        return logits, loss

# ----------------------
# Parameter counting helper
# ----------------------
def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[i] model param count: total={tot:,}  trainable={trainable:,}")

# ----------------------
# Save / load utilities
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
        # store model state on CPU to avoid device-specific state and fp16 headaches
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "tag": tag or f"step_{step}",
        "metric": metric
    }
    _atomic_save(payload, path)
    # prune
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
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "best_metric": best_metric
    }
    _atomic_save(payload, best_path)
    ts = time.strftime("%Y%m%d-%H%M%S")
    copy_path = os.path.join(best_dir, f"best_{best_metric:.6f}_step{step}_{ts}.pt")
    _atomic_save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "step": step, "metric": best_metric}, copy_path)
    print(f"[🏆] Saved best_val to {best_path} and copy {copy_path}")

# ----------------------
# Flexible weight loading (handles vocab & pos mismatch)
# ----------------------
def load_weights_flexibly(model: TinyGPT, ckpt_path: str, tokenizer: AutoTokenizer, device: torch.device):
    """
    Loads weights but tolerates:
      - vocab size mismatch (token embedding / head): copy overlap, init new rows
      - positional embedding mismatch (max_len): copy overlap, init or repeat last
    After adjustments, loads model.load_state_dict(state, strict=False)
    """
    print(f"[i] Loading weights-only from {ckpt_path} (flexible loader)")
    ckpt = torch.load(ckpt_path, map_location=device)
    # checkpoint might be payload or raw state dict
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if isinstance(state, dict) and all(isinstance(v, torch.Tensor) or not isinstance(v, dict) for v in state.values()):
        pass

    sd = model.state_dict()

    # ----------------
    # Positional embeddings: handle mismatch
    # ----------------
    if "pos.weight" in state and "pos.weight" in sd:
        src = state["pos.weight"]
        dst = sd["pos.weight"]
        old_len, d_old = src.size()
        new_len, d_new = dst.size()
        if (d_old != d_new):
            print(f"[!] Pos embedding hidden dim mismatch {d_old} vs {d_new}; will try to copy min dim and init rest.")
        if new_len != old_len:
            print(f"[i] Positional embedding length mismatch: ckpt={old_len} new_model={new_len}")
            # create a copy of the destination to modify
            dst_copy = dst.clone()
            if new_len > old_len:
                # copy old rows
                dst_copy[:old_len].copy_(src[:old_len])
                # initialize remaining rows by repeating the last row (stable) rather than random noise
                last_row = src[-1:].detach().clone()
                if last_row.size(0) > 0:
                    reps = last_row.repeat(new_len - old_len, 1)
                    dst_copy[old_len:].copy_(reps)
                else:
                    nn.init.normal_(dst_copy[old_len:], mean=0.0, std=0.02)
                state["pos.weight"] = dst_copy
                print(f"[i] Padded pos.weight from {old_len} -> {new_len} by repeating last vector.")
            else:
                # truncate
                state["pos.weight"] = src[:new_len]
                print(f"[i] Truncated pos.weight from {old_len} -> {new_len}.")
    # ----------------
    # Token embedding / head weight: handle vocab mismatch
    # keys in this model: 'tok.weight' and 'head.weight'
    # ----------------
    for emb_key, model_key in [("tok.weight", "tok.weight"), ("head.weight", "head.weight")]:
        if emb_key in state and model_key in sd:
            src = state[emb_key]
            dst = sd[model_key]
            old_vocab, d_old = src.size()
            new_vocab, d_new = dst.size()
            if new_vocab != old_vocab:
                print(f"[i] Vocab mismatch for {emb_key}: ckpt={old_vocab} model={new_vocab}")
                dst_copy = dst.clone()
                n_copy = min(old_vocab, new_vocab)
                dst_copy[:n_copy].copy_(src[:n_copy])
                if new_vocab > old_vocab:
                    # initialize new rows
                    nn.init.normal_(dst_copy[old_vocab:], mean=0.0, std=0.02)
                    print(f"[i] Padded {emb_key} with {new_vocab-old_vocab} new rows (normal_ init).")
                state[model_key] = dst_copy
    # Done adjusting, now filter state to keys that exist in model
    sd_keys = set(sd.keys())
    ckpt_keys = set(state.keys())
    extra_in_ckpt = sorted(list(ckpt_keys - sd_keys))
    missing_in_ckpt = sorted(list(sd_keys - ckpt_keys))
    print("extra_in_ckpt (first 10):", extra_in_ckpt[:10])
    print("missing_in_ckpt (first 10):", missing_in_ckpt[:10])

    # Filter down to only model keys for safety
    state_filtered = {k: v for k, v in state.items() if k in sd_keys}
    # final load
    model.load_state_dict(state_filtered, strict=False)
    # re-tie head to tok
    model.head.weight = model.tok.weight
    print("[i] Weights loaded (flexible).")

# ----------------------
# Dataset formatting & cleaning helpers
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
    response = ex.get("response_j", "") or ex.get("response_k", "") or ""
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
    # flexible: support either 'en'/'ta' or 'src'/'tgt' column names (we normalize earlier)
    en_src = ex.get("en", "") or ex.get("src", "") or ""
    target_tgt = ex.get(lang_code, "") or ex.get("tgt", "") or ""
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

    # Prepend this prefix to ALL instruction-response pairs
    ZIA_PREFIX = "You are ZIA, a reasoning assistant created by Akshay. You think clearly, respond naturally, and prefer factual, conversational explanations. Avoid being verbose or robotic.\n"

    def _add_prefix(ex):
        ex_text = ex.get("text", "")
        if not ex_text.startswith("### Instruction") and ZIA_PREFIX not in ex_text:
            return {"text": ZIA_PREFIX + ex_text}
        return ex

    # Alpaca
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

    # OASST
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

    # Dolly
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

    # OpenOrca (NEW)
    if cfg.use_openorca:
        try:
            n = cfg.openorca_subset if cfg.openorca_subset > 0 else ""
            split = f"train[:{n}]" if n else "train"
            print(f"[i] Loading OpenOrca ({split}) (Gated Access)...")
            d = load_dataset(cfg.openorca_name, split=split, token=True, **load_kwargs)
            d = d.map(lambda ex: _format_openorca(ex), remove_columns=d.column_names if hasattr(d, "column_names") else None)
            d = d.map(_add_prefix)
            ds_list.append(d)
        except Exception as e:
            print(f"[!] OpenOrca load failed (Did you run `huggingface-cli login` and accept the license?): {e}")

    # Stack Exchange (NEW)
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

    # Indic Instruct (ai4bharat/indic-align)
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

    # Indic Parallel (ai4bharat/samanantar)
    if cfg.use_indic_parallel:
        try:
            n = cfg.indic_parallel_subset if cfg.indic_parallel_subset > 0 else ""
            split = f"train[:{n}]" if n else "train"
            CONFIG_NAME = cfg.indic_parallel_lang
            print(f"[i] Loading Samanantar Parallel (Config: {CONFIG_NAME}, Target Lang: {CONFIG_NAME}) ({split}) ...")
            # load with explicit config
            d = load_dataset(cfg.indic_parallel_name, CONFIG_NAME, split=split, **load_kwargs)
            # columns vary by dataset; common forms: ('en','ta') or ('src','tgt')
            cols = d.column_names
            # normalize to 'en' and lang_code
            if "src" in cols and "tgt" in cols:
                d = d.rename_column("src", "en")
                d = d.rename_column("tgt", CONFIG_NAME)
            # else some versions already have 'en' and the language code name
            columns_to_keep = ["en", CONFIG_NAME] if all(c in d.column_names for c in ["en", CONFIG_NAME]) else d.column_names
            d = d.select_columns(columns_to_keep)
            d = d.filter(lambda ex: ex.get(CONFIG_NAME) is not None and len(ex.get(CONFIG_NAME).strip()) > 0)
            d = d.map(lambda ex: _format_samanantar(ex, CONFIG_NAME), remove_columns=d.column_names if hasattr(d, "column_names") else None)
            d = d.map(_add_prefix)
            ds_list.append(d)
        except Exception as e:
            print(f"[!] Samanantar Parallel load failed:", e)
            print(f"    Check that '{cfg.indic_parallel_lang}' is a valid language code in the dataset (e.g., 'ta', 'ml').")

    if not ds_list:
        raise RuntimeError("No datasets could be loaded. Enable at least one dataset.")

    merged = concatenate_datasets(ds_list) if len(ds_list) > 1 else ds_list[0]

    # basic cleaning: remove very short examples, strip whitespace
    def clean_fn(ex):
        txt = (ex.get("text") or "").strip()
        if not txt:
            return {"text": ""}
        if len(txt.split()) < cfg.min_tokens_in_example:
            return {"text": ""}
        return {"text": txt}
    merged = merged.map(clean_fn)

    # Drop examples with missing or too-short responses
    def has_valid_response(ex):
        txt = ex.get("text") or ""
        if "### Response:" not in txt:
            return False
        resp = txt.split("### Response:")[-1].strip()
        return len(resp.split()) >= 3

    merged = merged.filter(has_valid_response)

    # mark language-like heuristics
    merged = merged.map(lambda ex: {"is_english_like": bool(is_english_like(ex.get("text","")))})
    try:
        merged = merged.sort("is_english_like", reverse=True)
    except Exception:
        pass

    # dedupe by hash
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
        top_vals, top_idx = torch.topk(next_logits, k, dim=-1)          # (B, k)
        probs = torch.softmax(top_vals, dim=-1)                         # (B, k)
        idx = torch.multinomial(probs, num_samples=1)                   # (B, 1)
        next_token = torch.gather(top_idx, 1, idx)                      # (B, 1)

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
def train(cfg: V3Config):
    # Windows safety
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

    # tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path, use_fast=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token":"<pad>"})
    if tok.eos_token is None:
        tok.add_special_tokens({"eos_token":"<eos>"})

    # ensure model vocab matches tokenizer
    cfg.vocab_size = len(tok)
    print(f"[i] Tokenizer: vocab={len(tok)} pad={tok.pad_token_id} eos={tok.eos_token_id}")

    # Personality preamble that will be prepended to inference prompts
    assistant_preamble = (
        "You are ZIA, a reasoning assistant created by Akshay. "
        "You think clearly, respond naturally, and prefer factual, conversational explanations. "
        "Avoid being verbose or robotic. "
    )

    # dataset prepare
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

    # model
    model = TinyGPT(cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio, cfg.max_len, cfg.dropout, pad_token_id=tok.pad_token_id).to(device)
    model.pad_token_id = tok.pad_token_id

    # Print parameter counts
    count_params(model)

    start_step = 0
    # Load weights-only base first (flexible)
    if cfg.load_weights_only_from and os.path.exists(cfg.load_weights_only_from):
        try:
            load_weights_flexibly(model, cfg.load_weights_only_from, tok, device)
        except Exception as e:
            print("[!] Flexible weight load failed, trying strict load:", e)
            ckpt = torch.load(cfg.load_weights_only_from, map_location=device)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)

    # If resume_from specified, resume full checkpoint (model + optimizer)
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

    # optimizer & scheduler
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.max_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler() if (cfg.fp16 and torch.cuda.is_available()) else None

    # resume optimizer if requested
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

    # logging
    os.makedirs(cfg.out_dir, exist_ok=True)
    # save tokenizer for future inference & reproducibility
    tok.save_pretrained(os.path.join(cfg.out_dir, "tokenizer"))
    writer = SummaryWriter(os.path.join(cfg.out_dir, "runs"), comment=cfg.tb_comment)

    # training loop state
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
                # autocast device_type select
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

                    # logging
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

                    # evaluation
                    if step % cfg.eval_every == 0:
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

                        # inference sampling
                        if step % cfg.infer_every == 0:
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

                        # early stopping
                        if cfg.early_stop_patience and early_stop_counter >= cfg.early_stop_patience:
                            print(f"[!] Early stopping triggered (no val improvement in {cfg.early_stop_patience} evals).")
                            save_checkpoint(model, optimizer, scheduler, scaler, step, cfg.out_dir, keep_last=cfg.keep_last, tag="earlystop", metric=best_val)
                            raise KeyboardInterrupt("Early stopping")

                    # checkpointing
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
    # New dataset arguments
    p.add_argument("--use_openorca", action="store_true", help="Enable Open-Orca/OpenOrca dataset.")
    p.add_argument("--use_stack", action="store_true", help="Enable lvwerra/stack-exchange-paired dataset.")
    p.add_argument("--use_indic_instruct", action="store_true", help="Enable ai4bharat/indic-align instruct dataset.")
    p.add_argument("--use_indic_parallel", action="store_true", help="Enable ai4bharat/samanantar parallel corpus.")
    p.add_argument("--openorca_subset", type=int, default=None)
    p.add_argument("--stack_subset", type=int, default=None)
    p.add_argument("--indic_instruct_subset", type=int, default=None)
    p.add_argument("--indic_parallel_subset", type=int, default=None)
    p.add_argument("--indic_parallel_lang", type=str, default=None, help="Target language code for Samanantar (e.g., 'ta').")
    p.add_argument("--indic_subset", type=int, default=None)
    p.add_argument("--eval_every", type=int, default=None)
    p.add_argument("--infer_every", type=int, default=None)
    p.add_argument("--resume_optimizer", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--early_stop_patience", type=int, default=None)
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
    if args.indic_parallel_lang is not None: cfg.indic_parallel_lang = args.indic_parallel_lang
    if args.indic_subset is not None: cfg.indic_subset = args.indic_subset
    if args.eval_every is not None: cfg.eval_every = args.eval_every
    if args.infer_every is not None: cfg.infer_every = args.infer_every
    if args.resume_optimizer: cfg.resume_optimizer = True
    if args.fp16: cfg.fp16 = True
    if args.early_stop_patience is not None: cfg.early_stop_patience = args.early_stop_patience

    print("[i] Starting ZIA IFT v3 with config:")
    print(json.dumps(cfg.__dict__, indent=2, default=str))
    train(cfg)
