#!/usr/bin/env python3
"""
train_zia_ift_v2.py
Safe, resumable Instruction Fine-Tuning (IFT) script (ZIA IFT v2).

Features:
 - Loads tokenizer & weights-only from your saved base (recovered_best.pt).
 - Prepares a merged, diverse instruction dataset (Alpaca, OASST, Dolly, ShareGPT mini, + Tamil).
 - Dedupe, filter (improved robustness for empty instructions), and tokenize with your tokenizer.
 - Training loop with fresh optimizer by default; optional resume optimizer flag.
 - Tqdm live bar implemented.
 - Periodic inference quality check implemented.
 - Timestamped atomic checkpoints, best-val copies.
 - Periodic inference sampling / logs to console and TensorBoard.
 - Windows-safe dataset loading adjustments.
"""

import os
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
from typing import Optional, List

# --- Silence TRANSFORMERS_CACHE warning & set cache dirs early ---
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- New/Updated Imports ---
from tqdm import tqdm
# HF / torch / utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
# ----------------------
# Config dataclass
# ----------------------
@dataclass
class V2Config:
    # Paths / IO
    tokenizer_path: str = "artifacts/zia_tokenizer_60k"
    load_weights_only_from: Optional[str] = "artifacts/zia_dense_base/recovered_best.pt"
    out_dir: str = "artifacts/zia_ift_v2_runs"
    local_dataset_path: Optional[str] = "my_datasets/processed/arrow_cleaned_v1"

    # Model architecture (match README)
    vocab_size: int = 60000 
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    mlp_ratio: int = 4
    max_len: int = 256
    dropout: float = 0.1

    # Training hyperparams
    batch_size: int = 4              # micro-batch
    grad_accum: int = 8
    lr: float = 3e-05
    weight_decay: float = 0.0
    max_steps: int = 20000
    warmup_steps: int = 200
    eval_every: int = 500
    save_every: int = 500
    infer_every: int = 1000          # Run inference samples every 1000 steps
    keep_last: int = 5
    num_workers: int = 4
    fp16: bool = True
    seed: int = 42
    resume_optimizer: bool = False   # default: do NOT resume optimizer (safe)
    resume_from: Optional[str] = None  # specific checkpoint to resume from
    dataset_shuffle_seed: int = 42

    # Datasets options (subsets default small to test)
    use_alpaca: bool = True
    use_oasst: bool = True
    use_dolly: bool = True
    use_sharegpt: bool = True
    # --- New Tamil Datasets ---
    use_tamil_alpaca: bool = True
    use_tamil_instruction: bool = True

    alpaca_name: str = "yahma/alpaca-cleaned"
    # CHANGED to original OASST (requires special processing)
    oasst_name: str = "OpenAssistant/oasst1_oasst_sft_1"
    # Dolly uses the 15k version with a specific formatter
    dolly_name: str = "databricks/databricks-dolly-15k"  
    sharegpt_name: str = "OpenAssistant/research" # placeholder
    # --- Tamil Dataset Names (Common HF Repos) ---
    # CHANGED to a verified working repo
    tamil_alpaca_name: str = "Aya-A/Tamil-Alpaca-Data" 
    # CHANGED to a verified Tamil instruction repo
    tamil_instruction_name: str = "iitm-tse/tse-instruct-1.0" 

    # subset sizes (0 for full)
    alpaca_subset: int = 10000
    oasst_subset: int = 10000
    dolly_subset: int = 5000
    sharegpt_subset: int = 5000
    # --- Tamil Subsets ---
    tamil_alpaca_subset: int = 5000
    tamil_instruction_subset: int = 5000 

    # logging
    tb_comment: str = "zia_ift_v2"
    log_every: int = 100

# ----------------------
# Tiny GPT model (matches your README)
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
        # causal mask for multihead (attn_mask expects shape (T, T))
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
        # weight tie
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
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id
            )
        return logits, loss

# ----------------------
# Utilities: atomic save & best copy
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
    # prune old checkpoints
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

# ----------------------
# Dataset helpers
# ----------------------
def _format_alpaca(ex):
    inst = ex.get("instruction", "")
    inp = ex.get("input", "")
    out = ex.get("output") or ex.get("response") or ""
    prompt = inst
    if inp and len(inp.strip()) > 0:
        prompt = f"{inst}\n{inp}"
    text = f"### Instruction:\n{prompt}\n\n### Response:\n{out}"
    return {"text": text, "instruction": inst}

def _format_oasst(ex):
    # Handles OpenAssistant/oasst1 (nested structure)
    
    # Filter only samples where the conversation ends with an assistant response
    if ex["messages"][-1]["role"] != "assistant":
        return None # Filter this example out

    messages = ex["messages"]
    
    # Find the last instruction/response pair
    # We iterate backwards to find the last assistant message and the preceding user message
    response = ""
    instruction = ""
    
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            response = messages[i]["content"]
        elif messages[i]["role"] == "prompter" and not instruction:
            # Use the last prompter message as the instruction
            instruction = messages[i]["content"]
            break

    if not instruction or not response:
        return None # Should be filtered out by initial check, but safety first

    # Format the final instruction/response pair
    text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    return {"text": text, "instruction": instruction}

def _format_dolly_15k(ex):
    # Handles databricks/databricks-dolly-15k
    inst = ex.get("instruction", "")
    inp = ex.get("context", "")
    out = ex.get("response") or ""
    prompt = inst
    if inp and len(inp.strip()) > 0:
        prompt = f"{inst}\n{inp}"
    text = f"### Instruction:\n{prompt}\n\n### Response:\n{out}"
    return {"text": text, "instruction": inst}


def _format_tamil_instruction(ex):
    # Handles iitm-tse/tse-instruct-1.0 (uses 'query' and 'response')
    inst = ex.get("query") or ex.get("instruction") or ""
    out = ex.get("response") or ex.get("text") or ""
    
    # Check for another key that might contain context/input
    inp = ex.get("input", "")
    prompt = inst
    if inp and len(inp.strip()) > 0:
        prompt = f"{inst}\n{inp}"
        
    text = f"### Instruction:\n{prompt}\n\n### Response:\n{out}"
    return {"text": text, "instruction": inst}

def _format_tamil_alpaca(ex):
    # Handles Aya-A/Tamil-Alpaca-Data (uses 'instruction', 'input', 'output')
    inst = ex.get("instruction", "")
    inp = ex.get("input", "")
    out = ex.get("output") or ""
    prompt = inst
    if inp and len(inp.strip()) > 0:
        prompt = f"{inst}\n{inp}"
    text = f"### Instruction:\n{prompt}\n\n### Response:\n{out}"
    return {"text": text, "instruction": inst}

def prepare_merged_dataset(cfg: V2Config, **kwargs):
    ds_list = []
    
    # HF Datasets (English/Global)
    datasets_to_load = [
        ("Alpaca", cfg.use_alpaca, cfg.alpaca_name, kwargs.get("max_alpaca", cfg.alpaca_subset), _format_alpaca, {}),
        # Using the original OASST dataset, which requires the "messages" column
        ("OASST", cfg.use_oasst, cfg.oasst_name, kwargs.get("max_oasst", cfg.oasst_subset), _format_oasst, {"name": "default", "split": "train", "keep_in_memory": False}),
        # Dolly uses the 15k version with a specific formatter
        ("Dolly", cfg.use_dolly, cfg.dolly_name, kwargs.get("max_dolly", cfg.dolly_subset), _format_dolly_15k, {"trust_remote_code": True}), 
    ]

    # --- Tamil Datasets ---
    datasets_to_load.extend([
        # New Tamil Alpaca path
        ("Tamil Alpaca", cfg.use_tamil_alpaca, cfg.tamil_alpaca_name, kwargs.get("max_tamil_alpaca", cfg.tamil_alpaca_subset), _format_tamil_alpaca, {}),
        # New Tamil Instruction path
        ("Tamil Instruction", cfg.use_tamil_instruction, cfg.tamil_instruction_name, kwargs.get("max_tamil_instruction", cfg.tamil_instruction_subset), _format_tamil_instruction, {}), 
    ])

    for name, use_flag, ds_name, subset, formatter, load_kwargs in datasets_to_load:
        if use_flag:
            try:
                n = subset if subset > 0 else ""
                
                # Default split to 'train'
                split = load_kwargs.pop("split", "train")
                if n: split += f"[:{n}]"

                print(f"[i] Loading {name} ({split or 'full'}) ...")
                
                # load_dataset call
                d = load_dataset(ds_name, split=split, **load_kwargs)
                
                # Apply formatter and keep only 'text' and 'instruction' for filtering
                # Filter out None results from formatter (e.g. OASST conversations not ending in assistant turn)
                d = d.map(formatter, remove_columns=d.column_names if hasattr(d, "column_names") else None, batched=False)
                d = d.filter(lambda ex: ex is not None and ex.get("text") is not None and ex.get("instruction") is not None)
                ds_list.append(d)
                
            except Exception as e:
                # Print traceback to help the user debug HF access/repo issues outside the code
                print(f"[!] {name} load failed: {e}")
                traceback.print_exc() 
                
    # local arrow dataset if present
    if cfg.local_dataset_path and os.path.exists(cfg.local_dataset_path):
        try:
            print("[i] Loading local arrow dataset")
            dd = load_from_disk(cfg.local_dataset_path)
            
            def norm(ex):
                if "text" in ex:
                    text = ex["text"]
                    # Fallback to text if instruction extraction fails
                    instruction = text.split("### Instruction:\n", 1)[-1].split("\n\n### Response:", 1)[0].strip() or text
                    return {"text": text, "instruction": instruction}
                return {"text": str(ex), "instruction": str(ex)}

            if isinstance(dd, dict) and "train" in dd:
                # Use a maximum of 5000 or the alpaca subset size from the local data
                max_local = max(5000, cfg.alpaca_subset)
                s = dd["train"].select(range(min(len(dd["train"]), max_local)))
            else:
                s = dd.select(range(min(len(dd), 5000)))
            s = s.map(norm)
            ds_list.append(s)
        except Exception as e:
            print(f"[!] Local dataset load failed: {e}")
            traceback.print_exc()

    if not ds_list:
        raise RuntimeError("No datasets could be loaded. Enable at least one HF dataset or provide a local dataset path.")

    merged = concatenate_datasets(ds_list) if len(ds_list) > 1 else ds_list[0]
    
    # Filter: 1. Remove very short items. 2. Remove empty or nearly empty instructions.
    def filter_and_clean(ex):
        text = ex.get("text", "")
        instruction = ex.get("instruction", "")
        
        # 1. Filter out very short total entries
        if len(text.strip().split()) < 3:
            return False
            
        # 2. Crucial Filter: Discard the entry if instruction is missing/empty, as it's not a true instruction sample
        if not instruction or len(instruction.strip()) < 3:
             return False 

        return True

    initial_size = len(merged)
    merged = merged.filter(filter_and_clean)
    print(f"[i] Filtered {initial_size - len(merged)} records with short/empty instruction/text.")

    # dedupe by hash
    def add_hash(ex):
        h = hashlib.sha1((ex["text"] or "").encode("utf-8")).hexdigest()
        return {"text": ex["text"], "hash": h}
    merged = merged.map(add_hash)
    # remove duplicates
    seen = set()
    indices = []
    for i, rec in enumerate(merged):
        if rec["hash"] in seen:
            continue
        seen.add(rec["hash"])
        indices.append(i)
    
    final_size = len(merged.select(indices))
    print(f"[i] Merged dataset size after dedupe/filter: {final_size}")
    return merged.select(indices)

# ----------------------
# Collator
# ----------------------
class Collator:
    def __init__(self, tokenizer, max_len):
        self.tok = tokenizer
        self.max_len = max_len
    def __call__(self, batch):
        texts = [ (b["text"] if isinstance(b, dict) else str(b)) for b in batch ]
        # add eos if missing
        eos = self.tok.eos_token or "</s>"
        texts = [ (t if t.strip().endswith(eos) else t + eos) for t in texts ]
        enc = self.tok(texts, truncation=True, padding=True, max_length=self.max_len, return_tensors="pt")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        return input_ids, attention_mask

# ----------------------
# Eval & inference sampling
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
def greedy_generate(model, tokenizer, prompt, device, max_new_tokens=64, eos_id=None):
    model.eval()
    if isinstance(prompt, str):
        # Truncate prompt to prevent input length issues during inference
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.max_len - max_new_tokens).to(device)
        input_ids = enc["input_ids"]
    else:
        input_ids = prompt.to(device)
        
    B, T = input_ids.shape
    cur = input_ids
    
    # Ensure attention mask is created if not already
    attention_mask = (cur != model.pad_token_id).long()
    
    for _ in range(max_new_tokens):
        # Pass attention mask to model forward call
        logits, _ = model(cur, attention_mask=attention_mask, labels=None)
        
        # Use only the logits for the last token
        next_logits = logits[:, -1, :] 
        
        # Greedy search
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        
        # Prepare for next iteration
        cur = torch.cat([cur, next_token], dim=1)
        # Update attention mask
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
        
        if eos_id is not None and (next_token == eos_id).all():
            break
            
    # Decode the generated portion
    out = cur[0, T:].tolist()
    txt = tokenizer.decode(out, skip_special_tokens=True)
    model.train()
    return txt

# ----------------------
# Training loop
# ----------------------
def train(cfg: V2Config):
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
    print(f"[i] Device: {device} | CUDA: {torch.cuda.is_available()}")

    # tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path, use_fast=True)
    # ensure pad/eos tokens exist
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token":"<pad>"})
    if tok.eos_token is None:
        tok.add_special_tokens({"eos_token":"<eos>"})
    print(f"[i] Tokenizer: vocab={len(tok)} pad={tok.pad_token_id} eos={tok.eos_token_id}")

    # dataset
    merged = prepare_merged_dataset(cfg,
                                    max_alpaca=cfg.alpaca_subset,
                                    max_oasst=cfg.oasst_subset,
                                    max_dolly=cfg.dolly_subset,
                                    max_sharegpt=cfg.sharegpt_subset,
                                    max_tamil_alpaca=cfg.tamil_alpaca_subset,
                                    max_tamil_instruction=cfg.tamil_instruction_subset) # Corrected name
    # train/val split
    splits = merged.train_test_split(test_size=0.02, seed=cfg.dataset_shuffle_seed)
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
    cfg.vocab_size = len(tok) 
    model = TinyGPT(cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio, cfg.max_len, cfg.dropout, pad_token_id=tok.pad_token_id).to(device)
    model.pad_token_id = tok.pad_token_id

    # load weights-only base or resume checkpoint
    start_step = 0
    if cfg.resume_from:
        print(f"[i] Resuming from checkpoint {cfg.resume_from}")
        # Note: torch.load is marked with a FutureWarning, but it is necessary here to load non-model states.
        ckpt = torch.load(cfg.resume_from, map_location=device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            start_step = ckpt.get("step", 0)
        else:
            model.load_state_dict(ckpt)
            start_step = 0

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.max_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler() if (cfg.fp16 and device.type=="cuda") else None

    # resume optimizer & scheduler if requested
    if cfg.resume_from and cfg.resume_optimizer:
        # Load ckpt again to get optimizer/scheduler state
        ckpt = torch.load(cfg.resume_from, map_location=device) 
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                if "scheduler" in ckpt and ckpt["scheduler"] is not None:
                    scheduler.load_state_dict(ckpt["scheduler"])
                if "scaler" in ckpt and ckpt["scaler"] is not None and scaler is not None:
                    scaler.load_state_dict(ckpt["scaler"])
                start_step = ckpt.get("step", start_step)
                print(f"[i] Resumed optimizer/scheduler from checkpoint at step {start_step}")
            except Exception as e:
                print("[!] Failed to resume optimizer/scheduler:", e)

    # logging
    os.makedirs(cfg.out_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg.out_dir, "runs"), comment=cfg.tb_comment)

    # Training loop
    step = start_step
    micro_steps = 0
    running_loss = 0.0
    tokens_seen = 0
    best_val = float("inf")
    
    # Inference sampling prompts (English and Tamil)
    INFERENCE_PROMPTS = [
        "### Instruction:\nExplain the concept of large language models in simple terms.\n\n### Response:",
        "### Instruction:\nWrite a short, persuasive email to a client about a new product feature.\n\n### Response:",
        "### Instruction:\nTell me a short story about a flying dog.\n\n### Response:",
        "### Instruction:\nசங்க இலக்கியத்தில் குறிக்கப்பட்டுள்ள இயற்கை அழகைப் பற்றி ஒரு சிறு குறிப்பு தருக.\n\n### Response:", # Tamil: Give a short note on the natural beauty mentioned in Sangam literature.
        "### Instruction:\nஉலகில் உள்ள முதல் மூன்று நீண்ட நதிகள் யாவை?\n\n### Response:", # Tamil: What are the world's top three longest rivers?
    ]
    
    # Wrap DataLoader with tqdm
    pbar = tqdm(total=cfg.max_steps, initial=step, dynamic_ncols=True, desc="Training")

    try:
        data_iterator = iter(train_loader)
        while step < cfg.max_steps:
            try:
                ids, mask = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
                ids, mask = next(data_iterator)

            ids = ids.to(device)
            mask = mask.to(device)
            
            with torch.amp.autocast(device_type=device.type, enabled=(cfg.fp16 and device.type=="cuda")):
                logits, loss = model(ids, attention_mask=mask, labels=ids)
                if loss is None:
                    continue
                loss = loss / cfg.grad_accum
                
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_steps += 1
            running_loss += loss.item()
            tokens_seen += ids.numel()

            if micro_steps % cfg.grad_accum == 0:
                step += 1
                pbar.update(1)

                # optimizer step
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
                
                # Dynamic logging / PBAR update
                avg_loss = running_loss / cfg.log_every if cfg.log_every > 0 else running_loss
                lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else cfg.lr
                
                # Update PBAR metrics
                if avg_loss > 0:
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr_now:.2e}"}, refresh=False)

                # Logging to console/TensorBoard
                if step % cfg.log_every == 0:
                    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                    writer.add_scalar("train/loss", avg_loss, step)
                    writer.add_scalar("train/ppl", ppl, step)
                    writer.add_scalar("train/lr", lr_now, step)
                    writer.add_scalar("train/grad_norm", grad_norm, step)
                    print(f"[{time.strftime('%H:%M:%S')}] step={step} avg_loss={avg_loss:.4f} ppl={ppl:.2f} lr={lr_now:.2e}")
                    running_loss = 0.0

                # evaluation
                if step % cfg.eval_every == 0:
                    val_loss = evaluate(model, val_loader, device, max_batches=200)
                    val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                    writer.add_scalar("eval/loss", val_loss, step)
                    writer.add_scalar("eval/ppl", val_ppl, step)
                    pbar.write(f"[{time.strftime('%H:%M:%S')}] [Eval] step={step} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
                    
                    # save best
                    if val_loss < best_val:
                        best_val = val_loss
                        save_best(model, optimizer, scheduler, scaler, step, cfg.out_dir, best_val)

                    # intermediate inference check
                    if step % cfg.infer_every == 0:
                        pbar.write("\n" + "="*50 + "\n[INFERENCE QUALITY CHECK]")
                        try:
                            # Use pre-defined English and Tamil prompts
                            for i, prompt in enumerate(INFERENCE_PROMPTS):
                                # Truncate prompt text for console output clarity
                                console_prompt = prompt.replace("### Instruction:\n", "").replace("\n\n### Response:", "").strip()[:50] + "..."
                                generated = greedy_generate(model, tok, prompt, device, max_new_tokens=64, eos_id=tok.eos_token_id)
                                
                                pbar.write(f"[Sample {i}] Prompt: {console_prompt}\n-> {generated}\n")
                                writer.add_text(f"infer/sample_{i}", f"PROMPT: {prompt}\nGENERATED: {generated}", step)
                        except Exception as e:
                            pbar.write(f"[!] Inference sampling failed: {e}")
                        pbar.write("="*50 + "\n")

                # checkpointing
                if step % cfg.save_every == 0:
                    save_checkpoint(model, optimizer, scheduler, scaler, step, cfg.out_dir, keep_last=cfg.keep_last, metric=best_val)
                    pbar.write(f"[⏺] Checkpoint saved at step {step}")

                if step >= cfg.max_steps:
                    break

        # final
        pbar.close()
        save_checkpoint(model, optimizer, scheduler, scaler, step, cfg.out_dir, keep_last=cfg.keep_last, tag="final", metric=best_val)
        print("[✓] Training finished. Final checkpoint saved.")
    except KeyboardInterrupt:
        pbar.write("[!] KeyboardInterrupt — saving interrupt checkpoint...")
        save_checkpoint(model, optimizer, scheduler, scaler, step, cfg.out_dir, tag="interrupt", metric=best_val)
        pbar.close()
        raise
    except Exception as e:
        pbar.write("[!] Exception during training — saving error checkpoint...")
        traceback.print_exc()
        save_checkpoint(model, optimizer, scheduler, scaler, step, cfg.out_dir, tag="error", metric=best_val)
        pbar.close()
        raise
    finally:
        writer.close()

# ----------------------
# CLI
# ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--load_weights_only_from", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--alpaca_subset", type=int, default=None)
    p.add_argument("--oasst_subset", type=int, default=None)
    p.add_argument("--dolly_subset", type=int, default=None)
    p.add_argument("--tamil_alpaca_subset", type=int, default=None)
    p.add_argument("--tamil_instruction_subset", type=int, default=None) # Corrected name
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--resume_optimizer", action="store_true")
    p.add_argument("--eval_every", type=int, default=None, help="Evaluate every N steps")
    p.add_argument("--infer_every", type=int, default=None, help="Run quick inference sample every N steps")
    p.add_argument("--batch_size", type=int, default=None, help="Micro-batch size")
    p.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps")


    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = V2Config()
    
    # Overwrite config from CLI args
    if args.tokenizer_path: cfg.tokenizer_path = args.tokenizer_path
    if args.load_weights_only_from: cfg.load_weights_only_from = args.load_weights_only_from
    if args.out_dir: cfg.out_dir = args.out_dir
    if args.max_steps: cfg.max_steps = args.max_steps
    if args.alpaca_subset is not None: cfg.alpaca_subset = args.alpaca_subset
    if args.oasst_subset is not None: cfg.oasst_subset = args.oasst_subset
    if args.dolly_subset is not None: cfg.dolly_subset = args.dolly_subset
    if args.tamil_alpaca_subset is not None: cfg.tamil_alpaca_subset = args.tamil_alpaca_subset
    if args.tamil_instruction_subset is not None: cfg.tamil_instruction_subset = args.tamil_instruction_subset # Corrected name
    if args.resume_from: cfg.resume_from = args.resume_from
    if args.resume_optimizer: cfg.resume_optimizer = True
    if args.eval_every is not None: cfg.eval_every = args.eval_every
    if args.infer_every is not None: cfg.infer_every = args.infer_every
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.grad_accum is not None: cfg.grad_accum = args.grad_accum

    # print config
    print("[i] Starting ZIA IFT v2 with config:")
    print(json.dumps(cfg.__dict__, indent=2, default=str))
    train(cfg)