#!/usr/bin/env python3
"""
prepare_superift_v2_balanced.py

Balanced SuperIFT builder for ZIA (Option B ~400k examples).
- Uses local tokenizer at artifacts/zia_tokenizer_60k
- Streams each HF dataset (loads only needed splits, handles missing splits)
- Formats + cleans per-source
- Deduplicates (SHA1 on final text) in-memory (safe for ~400k)
- Tokenizes in batches with max_length=4096 (padding=max_length)
- Writes Arrow shards for train/val (train: 98%, val: 2%)
- Windows-friendly (avoid huge in-memory dataset concatenation)
- Adjustable per-dataset target counts to reach ~400k total

Run from project root (where artifacts/zia_tokenizer_60k exists).
"""
import os
import sys
import math
import json
import hashlib
import random
import time
from typing import Optional, List, Dict, Any

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ----------------------- CONFIG -----------------------
SEED = 42
random.seed(SEED)

TOKENIZER_DIR = "artifacts/zia_tokenizer_60k"   # your tokenizer
SAVE_ROOT = "datasets/processed/zia_superift_v2_balanced"
TRAIN_DIR = os.path.join(SAVE_ROOT, "train")
VAL_DIR = os.path.join(SAVE_ROOT, "val")
INFO_PATH = os.path.join(SAVE_ROOT, "info.json")

MAX_LEN = 4096           # 4k context
VAL_RATIO = 0.02         # 2% val
SHARD_SIZE = 5000        # number of examples per shard folder
NUM_PROC_TOKENIZE = 1    # safe on Windows; set >1 on Linux if desired
BATCH_TOKENIZE = 64      # batch size for tokenizer.map

# ~400k balanced target distribution (you can tune)
PER_DATASET_TARGETS = {
    "yahma/alpaca-cleaned": 30000,
    "OpenAssistant/oasst1": 90000,
    "HuggingFaceH4/ultrachat_200k": 70000,
    "databricks/databricks-dolly-15k": 20000,
    "Open-Orca/OpenOrca": 120000,
    "lvwerra/stack-exchange-paired": 20000,
    # fallback/extra sources can be added
}

# safe minimums for an example
MIN_TOKENS_IN_RESPONSE = 3
MIN_CHARS_INPUT = 3
MIN_CHARS_OUTPUT = 3

# HF load kwargs (keep small memory footprint)
LOAD_KWARGS = {
    # "cache_dir": "datasets/raw/hf_cache",  # optional if you want to enforce cache location
    # "use_auth_token": True,  # add if required for gated datasets
}

# -------------------- HELPERS --------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def clean_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    # normalize newlines/spaces, remove excessive whitespace
    s = s.replace("\r", " ").replace("\t", " ")
    s = s.replace("\n\n", "\n").strip()
    # collapse multiple spaces
    s = " ".join(s.split())
    return s.strip()

# -------------------- FORMATTERS --------------------
def fmt_alpaca(ex: Dict[str, Any]) -> str:
    inst = ex.get("instruction", "") or ""
    inp = ex.get("input", "") or ""
    out = ex.get("output", "") or ""
    parts = []
    if inst: parts.append("### Instruction:\n" + inst.strip())
    if inp: parts.append("### Input:\n" + inp.strip())
    parts.append("### Response:\n" + out.strip())
    return "\n\n".join(parts)

def fmt_oasst(ex: Dict[str, Any]) -> str:
    # Try structured messages first
    if ex.get("messages"):
        msgs = ex["messages"]
        parts = []
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content") or m.get("text") or ""
            if content:
                parts.append(f"{role}: {content.strip()}")
        text = "\n".join(parts).strip()
        # Put entire conversation under instruction; empty response to avoid dropping
        return f"### Instruction:\n{text}\n\n### Response:\n"
    # fallback
    inst = ex.get("input") or ex.get("prompt") or ex.get("instruction") or ""
    out = ex.get("output") or ex.get("response") or ""
    return f"### Instruction:\n{inst.strip()}\n\n### Response:\n{out.strip()}"

def fmt_dolly(ex: Dict[str, Any]) -> str:
    instr = ex.get("instruction") or ex.get("title") or ""
    out = ex.get("text") or ex.get("response") or ex.get("output") or ""
    return f"### Instruction:\n{instr.strip()}\n\n### Response:\n{out.strip()}"

def fmt_ultrachat(ex: Dict[str, Any]) -> str:
    # ultrachat often contains 'text' or conversation structure
    txt = ex.get("text") or ex.get("dialogue") or ex.get("conversations") or ""
    if isinstance(txt, list):
        txt = "\n".join(str(x) for x in txt)
    return f"### Instruction:\n{txt}\n\n### Response:\n"

def fmt_openorca(ex: Dict[str, Any]) -> str:
    sys_prompt = ex.get("system_prompt") or ""
    question = ex.get("question") or ex.get("prompt") or ""
    response = ex.get("response") or ex.get("answer") or ""
    prompt = (("[SYSTEM] " + sys_prompt + "\n") if sys_prompt else "") + question
    return f"### Instruction:\n{prompt.strip()}\n\n### Response:\n{response.strip()}"

def fmt_stack(ex: Dict[str, Any]) -> str:
    q = ex.get("question") or ex.get("prompt") or ""
    r = ex.get("response_j") or ex.get("response_k") or ex.get("answer") or ""
    return f"### Instruction:\n{q.strip()}\n\n### Response:\n{r.strip()}"

# map dataset -> formatter
FORMATTERS = {
    "yahma/alpaca-cleaned": fmt_alpaca,
    "OpenAssistant/oasst1": fmt_oasst,
    "databricks/databricks-dolly-15k": fmt_dolly,
    "HuggingFaceH4/ultrachat_200k": fmt_ultrachat,
    "Open-Orca/OpenOrca": fmt_openorca,
    "lvwerra/stack-exchange-paired": fmt_stack,
}

# ------------------ PROCESSING PIPELINE ------------------
def process_dataset(
    repo_name: str,
    target: int,
    tokenizer,
    seen_hashes: set,
    output_accumulators: Dict[str, List[Dict[str, Any]]],
    counters: Dict[str, int],
):
    """
    Streams and processes one HF dataset until 'target' accepted examples are gathered (or dataset exhausted).
    Appends tokenized examples (dict with input_ids, attention_mask, labels) to output_accumulators['train'/'val'].
    """
    print(f"\n[→] Processing {repo_name}  target={target:,}")
    formatter = FORMATTERS.get(repo_name, None)

    # try common splits heuristics
    candidate_splits = ["train", "train[:]", "train[:]", "train[::1]"]
    # Some datasets have different named splits; try a small set of likely names
    try_splits = ["train", "train_sft", "train_gen", "all", "validation"]
    loaded = None
    # Attempt to load dataset; allow fallback if split names differ
    for sp in try_splits:
        try:
            ds = load_dataset(repo_name, split=sp, **LOAD_KWARGS)
            loaded = ds
            break
        except Exception:
            continue
    if loaded is None:
        # final try: load without specifying split (builder decides)
        try:
            ds = load_dataset(repo_name, **LOAD_KWARGS)
            # if dataset returns dict-like with splits, pick first split
            if isinstance(ds, dict):
                # pick common split keys
                for k in ("train", "train_sft", "train_gen", "validation"):
                    if k in ds:
                        loaded = ds[k]
                        break
                if loaded is None:
                    # pick first split available
                    first = list(ds.keys())[0]
                    loaded = ds[first]
            else:
                loaded = ds
        except Exception as e:
            print(f"[!] Failed to load {repo_name}: {e}")
            return

    ds = loaded
    total_in_source = len(ds) if hasattr(ds, "__len__") else None
    if total_in_source is not None:
        print(f"[i] Source size: {total_in_source:,}")

    # We'll iterate through the dataset in streaming fashion via `.select` windows to avoid materializing huge lists.
    # But HuggingFace `Dataset` supports indexing and iteration; this should be fine on local caches.
    accepted = 0
    processed = 0

    # Use iterator over dataset (this will use available caching)
    iterator = iter(ds)

    while accepted < target:
        try:
            ex = next(iterator)
        except StopIteration:
            break
        processed += 1

        # format and clean
        try:
            if formatter:
                text = formatter(ex)
            else:
                # generic fallback - try to join common fields
                inst = ex.get("instruction") or ex.get("input") or ex.get("prompt") or ex.get("question") or ""
                out = ex.get("output") or ex.get("response") or ex.get("answer") or ""
                text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
        except Exception:
            # skip malformed entry
            continue

        text = clean_text(text)
        if not text:
            continue
        # Quick heuristics: ensure "### Response:" exists
        if "### Response:" not in text:
            continue

        # ensure response is long enough
        resp = text.split("### Response:")[-1].strip()
        instr_part = text.split("### Response:")[0].strip()
        if len(resp.split()) < MIN_TOKENS_IN_RESPONSE:
            continue
        if len(instr_part) < MIN_CHARS_INPUT:
            continue

        # dedupe by hash of entire text
        h = sha1_hex(text)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # Decide train/val split (deterministic by hash to avoid drifting)
        is_val = (int(h[:8], 16) % 100) < int(VAL_RATIO * 100)

        target_split = "val" if is_val else "train"
        # respect per-dataset target only for train+val combined; if val chosen but we are already saturated in val we can place into train
        if counters["total_written"] >= counters["target_total"]:
            # already reached global target; stop
            break

        # Append raw text and source meta (we will tokenize in batches later)
        output_accumulators[target_split].append({"text": text, "source": repo_name})
        counters["written_per_source"].setdefault(repo_name, 0)
        counters["written_per_source"][repo_name] += 1
        counters["total_written"] += 1
        accepted += 1

        # occasional progress print
        if processed % 5000 == 0:
            print(f"[{repo_name}] processed {processed:,} -> accepted {accepted:,} (global {counters['total_written']:,})")

        # global safety break
        if counters["total_written"] >= counters["target_total"]:
            break

    print(f"[✓] Finished {repo_name}: scanned {processed:,}, accepted {accepted:,}")
    return

# ------------------ TOKENIZE & SHARD SAVE ------------------
def tokenize_and_save_accumulators(tokenizer, accumulators, save_root, shard_size=SHARD_SIZE):
    """
    Tokenizes accumulators['train'] and ['val'] and writes shards to disk in save_root/train/* and save_root/val/*
    Expects accumulators to contain lists of dicts with keys 'text' and 'source'.
    """
    ensure_dir(save_root)
    train_list = accumulators["train"]
    val_list = accumulators["val"]

    def _tokenize_list(lst, split_name):
        out_shard_paths = []
        total = len(lst)
        print(f"[i] Tokenizing {split_name}: {total:,} examples")
        # process in chunks to avoid huge memory use
        for i in range(0, total, shard_size):
            chunk = lst[i:i+shard_size]
            texts = [x["text"] for x in chunk]
            # batch tokenize
            enc = tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors=None)
            # build examples with labels (-100 shift for CLM)
            examples = []
            for j, tid in enumerate(enc["input_ids"]):
                # create labels: copy input_ids but keep -100 for first token to match the training scripts that expect labels shifted by 1
                labels = [-100] + tid[1:]
                examples.append({
                    "input_ids": tid,
                    "attention_mask": enc["attention_mask"][j],
                    "labels": labels,
                    "source": chunk[j]["source"],
                    "text": chunk[j]["text"],
                })
            # save shard to disk as a dataset
            shard_idx = i // shard_size
            shard_folder = os.path.join(save_root, split_name, f"shard_{shard_idx:03d}")
            ensure_dir(shard_folder)
            ds = Dataset.from_list(examples)
            ds.save_to_disk(shard_folder)
            out_shard_paths.append(shard_folder)
            print(f"[saved] {split_name} shard {shard_idx:03d} -> {shard_folder} ({len(examples):,})")
        return out_shard_paths

    train_paths = _tokenize_list(train_list, "train")
    val_paths = _tokenize_list(val_list, "val")
    return train_paths, val_paths

# ---------------------- MAIN ----------------------
def main():
    ensure_dir(SAVE_ROOT)
    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)

    # compute global target total
    target_total = sum(PER_DATASET_TARGETS.values())
    # safety clamp: aim for ~400k; if user config differs it's respected
    print(f"[i] Balanced target_total = {target_total:,} (VAL ratio={VAL_RATIO*100:.1f}%)")

    # load tokenizer
    print("[i] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True, local_files_only=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    if tok.eos_token is None:
        tok.add_special_tokens({"eos_token": "</s>"})
    print(f"[i] Tokenizer loaded: vocab={len(tok)} pad={tok.pad_token_id} eos={tok.eos_token_id}")

    # accumulators kept in memory but limited by target_total (400k -> OK)
    accumulators = {"train": [], "val": []}
    seen_hashes = set()
    counters = {
        "total_written": 0,
        "target_total": target_total,
        "written_per_source": {}
    }

    # Iterate datasets in deterministic order
    for repo_name, tgt in PER_DATASET_TARGETS.items():
        # if already reached global target, break
        if counters["total_written"] >= counters["target_total"]:
            break
        remaining_global = counters["target_total"] - counters["total_written"]
        take = min(tgt, remaining_global)
        process_dataset(repo_name, take, tok, seen_hashes, accumulators, counters)

    total_collected = counters["total_written"]
    print(f"\n[i] Collection complete. Total collected: {total_collected:,}")

    # quick shuffle
    random.Random(SEED).shuffle(accumulators["train"])
    random.Random(SEED + 1).shuffle(accumulators["val"])

    # If val is empty (rare) take random 2% from train
    if len(accumulators["val"]) == 0 and len(accumulators["train"]) > 0:
        n_val = max(1, int(len(accumulators["train"]) * VAL_RATIO))
        acc = accumulators["train"]
        accumulators["val"] = acc[:n_val]
        accumulators["train"] = acc[n_val:]
        print(f"[i] Created val from train: val={len(accumulators['val']):,} train={len(accumulators['train']):,}")

    # Tokenize and save shards (this will write multiple shard folders)
    print("\n[i] Tokenizing and saving shards (this may take a while)...")
    train_paths, val_paths = tokenize_and_save_accumulators(tok, accumulators, SAVE_ROOT, shard_size=SHARD_SIZE)

    # Write info.json
    info = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "target_total": target_total,
        "collected": total_collected,
        "val_ratio": VAL_RATIO,
        "max_len": MAX_LEN,
        "tokenizer": TOKENIZER_DIR,
        "shard_size": SHARD_SIZE,
        "train_shards": train_paths,
        "val_shards": val_paths,
        "written_per_source": counters["written_per_source"],
    }
    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"\n[✓] Done. Prepared dataset saved to: {SAVE_ROOT}")
    print("info:", info)

if __name__ == "__main__":
    main()
