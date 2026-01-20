#!/usr/bin/env python3
"""
prepare_ift_option_c.py

Memory-safe, streaming-friendly IFT dataset builder for ZIA (Option C: compact ~100k samples).
- Uses HF datasets but only pulls limited slices per source to keep total ~100k.
- Per-dataset streaming + formatting -> light dedupe -> batched tokenization -> incremental shard write.
- Tokenizes to MAX_LEN=4096 using your tokenizer at artifacts/zia_tokenizer_60k.
- Output: datasets/processed/zia_ift_compact_4k/{train, val}/shard_XX folders (Arrow).
- Windows-safe and designed to run on modest machines.

Run from project root (where artifacts/zia_tokenizer_60k exists).
"""
import os
import math
import json
import hashlib
import random
from typing import List, Dict, Iterable

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ---------------- CONFIG ----------------
OUT_BASE = "datasets/processed/zia_ift_compact_4k"
TOKENIZER_DIR = "artifacts/zia_tokenizer_60k"
MAX_LEN = 4096
SEED = 42

# target total (approx)
TARGET_TOTAL = 100_000

# per-dataset target allocation (you can tune)
ALLOC = {
    "yahma/alpaca-cleaned": 25_000,
    "OpenAssistant/oasst1": 30_000,
    "HuggingFaceH4/ultrachat_200k": 20_000,  # use train_sft split
    "databricks/databricks-dolly-15k": 8_000,
    "Open-Orca/OpenOrca": 17_000,           # small portion
}

# splits to use (some datasets need special split names)
SPLIT_OVERRIDE = {
    "HuggingFaceH4/ultrachat_200k": "train_sft",
    # others default to "train"
}

# tokenization / batching
TOKENIZE_BATCH = 16           # smaller on Windows; increase on Linux
MAP_BATCH = 512               # number of examples fetched per map/iteration
SHARD_SIZE = 5000             # number of tokenized examples per shard (adjustable)
VAL_RATIO = 0.02              # 2% validation

# minimal response length filter
MIN_RESPONSE_TOKENS = 3

# deterministic
random.seed(SEED)

# ---------------- HELPERS ----------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()

def format_example(dataset_name: str, ex: Dict) -> str:
    """
    Normalize to a single text field with '### Instruction:' and '### Response:'.
    Conservative: only include fields that commonly exist.
    """
    if dataset_name == "yahma/alpaca-cleaned":
        inst = ex.get("instruction", "") or ""
        inp = ex.get("input", "") or ""
        out = ex.get("output", "") or ex.get("response", "") or ""
        prompt = inst if not inp else f"{inst}\n{inp}"
        return f"### Instruction:\n{prompt.strip()}\n\n### Response:\n{out.strip()}"
    if dataset_name == "OpenAssistant/oasst1":
        # many entries have 'messages' list
        if "messages" in ex and isinstance(ex["messages"], list):
            parts = []
            for m in ex["messages"]:
                role = m.get("role", "")
                content = m.get("content") or m.get("text") or ""
                parts.append(f"{role}: {content}")
            txt = "\n".join(parts).strip()
            return f"### Instruction:\n{txt}\n\n### Response:\n"
        prompt = ex.get("input") or ex.get("prompt") or ex.get("instruction") or ""
        out = ex.get("response") or ex.get("text") or ex.get("output") or ""
        return f"### Instruction:\n{prompt.strip()}\n\n### Response:\n{out.strip()}"
    if dataset_name == "HuggingFaceH4/ultrachat_200k":
        # choose possible fields conservatively
        if "text" in ex:
            txt = ex.get("text") or ""
            return f"### Instruction:\n{txt.strip()}\n\n### Response:\n"
        # fallback
        q = ex.get("question") or ex.get("prompt") or ""
        r = ex.get("response") or ""
        return f"### Instruction:\n{q}\n\n### Response:\n{r}"
    if dataset_name == "databricks/databricks-dolly-15k":
        inst = ex.get("instruction") or ex.get("title") or ""
        out = ex.get("text") or ex.get("response") or ex.get("output") or ""
        return f"### Instruction:\n{inst.strip()}\n\n### Response:\n{out.strip()}"
    if dataset_name == "Open-Orca/OpenOrca":
        # many OpenOrca entries have 'question' and 'response' or 'system_prompt'
        q = ex.get("question") or ex.get("prompt") or ex.get("instruction") or ""
        r = ex.get("response") or ex.get("answer") or ex.get("completion") or ""
        return f"### Instruction:\n{q.strip()}\n\n### Response:\n{r.strip()}"
    # generic fallback
    txt = ex.get("text") or ex.get("input") or ex.get("instruction") or ""
    out = ex.get("output") or ex.get("response") or ""
    if out:
        return f"### Instruction:\n{txt.strip()}\n\n### Response:\n{out.strip()}"
    return txt.strip()

# ---------------- STREAMING PIPELINE ----------------
def process_and_write(tokenizer, dataset_name: str, split: str, target_n: int,
                      global_seen_hashes: set, train_dir: str, val_dir: str):
    """
    Load dataset slice (split), stream format -> dedupe -> batch tokenize -> write shards incrementally.
    Returns count written.
    """
    written = 0
    examples_for_shard = []
    shard_idx = 0

    # use a limited split to avoid pulling entire dataset when not needed
    # we will attempt to load train[:limit] where limit = target_n * 2 (to allow filtering/dedup)
    probe_limit = target_n * 3  # over-fetch factor; small enough for Option C
    split_expr = f"{split}[:{probe_limit}]" if probe_limit else split

    ds = load_dataset(dataset_name, split=split_expr)
    # iterate in small batches (using .select by ranges is simpler on Windows)
    total = len(ds)
    idx = 0

    pbar = tqdm(total=min(total, probe_limit), desc=f"proc {dataset_name}", unit="ex")
    while idx < total and written < target_n:
        # fetch a small chunk
        end = min(total, idx + MAP_BATCH)
        batch = ds.select(range(idx, end))
        idx = end

        # format
        txts = []
        for ex in batch:
            t = format_example(dataset_name, ex)
            if not t:
                continue
            # basic check: must contain response marker
            if "### Response:" not in t:
                continue
            resp = t.split("### Response:")[-1].strip()
            if len(resp.split()) < MIN_RESPONSE_TOKENS:
                continue
            h = sha1(t)
            if h in global_seen_hashes:
                continue
            global_seen_hashes.add(h)
            txts.append({"text": t})
        pbar.update(len(batch))

        if not txts:
            continue

        # tokenize in smaller batches
        for i in range(0, len(txts), TOKENIZE_BATCH):
            sub = txts[i:i+TOKENIZE_BATCH]
            texts = [s["text"] for s in sub]
            enc = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN)
            input_ids = enc["input_ids"]
            # labels for causal LM: shift by 1 and pad first token with -100
            labels = [[-100] + ids[1:] for ids in input_ids]

            for ids, lbl, txt in zip(input_ids, labels, texts):
                examples_for_shard.append({"input_ids": ids, "labels": lbl, "text": txt})
                # when shard buffer full, flush to disk
                if len(examples_for_shard) >= SHARD_SIZE:
                    # split to train/val using simple VAL_RATIO per example
                    train_bucket, val_bucket = [], []
                    for ex in examples_for_shard:
                        if random.random() < VAL_RATIO:
                            val_bucket.append({"input_ids": ex["input_ids"], "labels": ex["labels"], "text": ex["text"]})
                        else:
                            train_bucket.append({"input_ids": ex["input_ids"], "labels": ex["labels"], "text": ex["text"]})
                    if train_bucket:
                        shard_folder = os.path.join(train_dir, f"{dataset_name.replace('/','_')}_shard_{shard_idx:03d}")
                        Dataset.from_list(train_bucket).save_to_disk(shard_folder)
                    if val_bucket:
                        shard_folder = os.path.join(val_dir, f"{dataset_name.replace('/','_')}_shard_{shard_idx:03d}")
                        Dataset.from_list(val_bucket).save_to_disk(shard_folder)
                    shard_idx += 1
                    written += len(examples_for_shard)
                    examples_for_shard = []
                    # early stop if reached target
                    if written >= target_n:
                        break
            if written >= target_n:
                break
        if written >= target_n:
            break

    # final flush remaining buffer
    if examples_for_shard and written < target_n:
        # limit to remaining needed
        remaining = max(0, target_n - written)
        to_flush = examples_for_shard[:remaining]
        train_bucket, val_bucket = [], []
        for ex in to_flush:
            if random.random() < VAL_RATIO:
                val_bucket.append({"input_ids": ex["input_ids"], "labels": ex["labels"], "text": ex["text"]})
            else:
                train_bucket.append({"input_ids": ex["input_ids"], "labels": ex["labels"], "text": ex["text"]})
        if train_bucket:
            shard_folder = os.path.join(train_dir, f"{dataset_name.replace('/','_')}_shard_{shard_idx:03d}")
            Dataset.from_list(train_bucket).save_to_disk(shard_folder)
        if val_bucket:
            shard_folder = os.path.join(val_dir, f"{dataset_name.replace('/','_')}_shard_{shard_idx:03d}")
            Dataset.from_list(val_bucket).save_to_disk(shard_folder)
        written += len(to_flush)
        examples_for_shard = []

    pbar.close()
    return written

# ---------------- ORCHESTRATOR ----------------
def main():
    ensure_dir(OUT_BASE)
    train_dir = os.path.join(OUT_BASE, "train")
    val_dir = os.path.join(OUT_BASE, "val")
    ensure_dir(train_dir); ensure_dir(val_dir)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    total_written = 0
    global_seen = set()

    # compute per-dataset targets scaled so sum ~ TARGET_TOTAL
    alloc_sum = sum(ALLOC.values())
    scale = TARGET_TOTAL / alloc_sum if alloc_sum > 0 else 1.0
    per_dataset_targets = {k: max(100, int(v * scale)) for k, v in ALLOC.items()}

    print(f"[i] Target total ~{TARGET_TOTAL}. Per-dataset targets: {per_dataset_targets}")

    for dataset_name, target in per_dataset_targets.items():
        split = SPLIT_OVERRIDE.get(dataset_name, "train")
        print(f"\n[i] Processing {dataset_name} split={split} target={target} ...")
        written = process_and_write(tokenizer, dataset_name, split, target, global_seen, train_dir, val_dir)
        total_written += written
        print(f"[i] -> written {written} for {dataset_name} (cumulative {total_written})")

    # final metadata summary
    meta = {
        "approx_total_written": total_written,
        "target_total": TARGET_TOTAL,
        "per_dataset_targets": per_dataset_targets,
        "max_len": MAX_LEN,
        "tokenizer": TOKENIZER_DIR,
        "shard_size": SHARD_SIZE,
        "val_ratio": VAL_RATIO,
        "seed": SEED
    }
    with open(os.path.join(OUT_BASE, "info.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("\n[✓] Done. Output saved to:", OUT_BASE)
    print("[✓] Info:", meta)

if __name__ == "__main__":
    main()
