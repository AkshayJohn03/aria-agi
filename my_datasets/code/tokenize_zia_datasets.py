#!/usr/bin/env python3
# tokenize_zia_datasets.py
"""
Tokenize cleaned ZIA datasets (sharded Arrow dirs) into a tokenized DatasetDict
suitable for train_zia_ift_v3_rope_alibi.py's `--tokenized_dataset_path`.

Usage (example):
python tokenize_zia_datasets.py \
  --raw_data_dir datasets/processed/zia_ift_v3_clean_v2 \
  --tokenizer_path artifacts/zia_tokenizer_60k \
  --out_path artifacts/tokenized_dataset/zia_ift_v3 \
  --max_len 512 \
  --num_proc 4
"""
import os
import argparse
import math
from datasets import concatenate_datasets, load_from_disk, DatasetDict
from transformers import AutoTokenizer
from tqdm.auto import tqdm

def list_shard_dirs(parent_dir):
    """Return a sorted list of shard directories inside parent_dir (shard_00, shard_01...)."""
    if not os.path.isdir(parent_dir):
        return []
    items = sorted(
        [os.path.join(parent_dir, d) for d in os.listdir(parent_dir)],
        key=lambda p: p.lower()
    )
    # keep only directories that look like shard folders (or dataset dirs)
    dirs = [p for p in items if os.path.isdir(p)]
    return dirs

def load_shards_concat(path_or_dir):
    """
    Load a dataset or concatenate shards.
    Accepts either:
      - a direct dataset folder saved by datasets.save_to_disk (single dataset)
      - a directory with shard_xx subfolders (each created via save_to_disk)
    """
    if os.path.isdir(path_or_dir):
        # If this folder itself looks like a dataset (has dataset_info.json), load directly
        try:
            if os.path.exists(os.path.join(path_or_dir, "dataset_info.json")):
                return load_from_disk(path_or_dir)
        except Exception:
            pass
        # otherwise try to load sub-shards
        shard_dirs = list_shard_dirs(path_or_dir)
        if shard_dirs:
            datasets = []
            for sd in shard_dirs:
                try:
                    d = load_from_disk(sd)
                    # If shard saved as a DatasetDict, attempt to extract train/validation
                    if isinstance(d, dict) or "train" in getattr(d, "keys", lambda: [])():
                        # if user accidentally saved a DatasetDict inside each shard, handle
                        if "train" in d:
                            datasets.append(d["train"])
                        else:
                            # fallback: convert to list
                            datasets.append(d)
                    else:
                        datasets.append(d)
                except Exception as e:
                    print(f"[!] Could not load shard {sd}: {e}")
            if not datasets:
                raise RuntimeError(f"No shards loaded from {path_or_dir}")
            return concatenate_datasets(datasets)
    # fallback: try loading directly (may raise)
    return load_from_disk(path_or_dir)

def make_tokenize_fn(tokenizer, max_len, template):
    """
    Returns a map function usable in datasets.map. The function expects examples
    to have 'input' and 'output' columns (strings).
    """
    IGNORE_INDEX = -100

    def tokenize_batch(examples):
        inputs = examples["input"]
        outputs = examples["output"]

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for inp, out in zip(inputs, outputs):
            text = template.format(instruction=inp, response=out)
            # Tokenize full sequence
            enc = tokenizer(
                text,
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )
            # `enc` returns lists when return_tensors=None
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            # Determine how many tokens belong to instruction prefix, so we can mask labels
            instr_prefix = template.format(instruction=inp, response="")
            instr_enc = tokenizer(instr_prefix, truncation=True, padding=False, return_tensors=None)
            cutoff = len(instr_enc["input_ids"])

            # Build labels and mask prefix + padding with IGNORE_INDEX
            labels = list(input_ids)[:]  # shallow copy
            # mask instruction tokens
            for i in range(min(cutoff, len(labels))):
                labels[i] = IGNORE_INDEX
            # mask padding tokens
            for i, m in enumerate(attention_mask):
                if m == 0:
                    labels[i] = IGNORE_INDEX

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    return tokenize_batch

def main():
    p = argparse.ArgumentParser(description="Tokenize ZIA cleaned dataset shards")
    p.add_argument("--raw_data_dir", type=str, required=True,
                   help="Path to cleaned dataset folder (containing train/val shard dirs).")
    p.add_argument("--tokenizer_path", type=str, default="artifacts/zia_tokenizer_60k")
    p.add_argument("--out_path", type=str, default="artifacts/tokenized_dataset/zia_ift_v3")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--num_proc", type=int, default=None,
                   help="Number of processes for datasets.map (use 0 on Windows).")
    p.add_argument("--template", type=str, default="Instruction: {instruction}\nResponse: {response}",
                   help="Template used to join instruction+response for tokenization.")
    args = p.parse_args()

    # Determine num_proc: default 0 on Windows to avoid multiprocessing pickling issues
    if args.num_proc is None:
        args.num_proc = 0 if os.name == "nt" else max(1, (os.cpu_count() or 1) - 1)

    os.makedirs(args.out_path, exist_ok=True)

    print(f"[i] Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        # Ensure pad token exists for masking
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # Load train/val shards
    train_src = os.path.join(args.raw_data_dir, "train")
    val_src = os.path.join(args.raw_data_dir, "val")

    print(f"[i] Loading train shards from: {train_src}")
    train_ds = load_shards_concat(train_src)
    print(f"[i] Loaded train examples: {len(train_ds)}")

    print(f"[i] Loading val shards from: {val_src}")
    val_ds = load_shards_concat(val_src)
    print(f"[i] Loaded val examples: {len(val_ds)}")

    # Map function
    tokenize_fn = make_tokenize_fn(tokenizer, args.max_len, args.template)

    # Remove columns after tokenization. Expect original columns to be at least 'input' and 'output'
    remove_cols = train_ds.column_names

    print(f"[i] Tokenizing train dataset with num_proc={args.num_proc} ...")
    train_tok = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=remove_cols,
        desc="Tokenizing train",
        num_proc=(args.num_proc if args.num_proc > 0 else None),
    )

    print(f"[i] Tokenizing validation dataset with num_proc={args.num_proc} ...")
    val_tok = val_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=val_ds.column_names,
        desc="Tokenizing val",
        num_proc=(args.num_proc if args.num_proc > 0 else None),
    )

    ds_dict = DatasetDict({"train": train_tok, "validation": val_tok})
    print(f"[i] Saving tokenized dataset to: {args.out_path}")
    ds_dict.save_to_disk(args.out_path)

    print(f"[✅] Tokenization complete. Saved to {args.out_path}")
    print(f"     train -> {len(ds_dict['train'])} examples")
    print(f"     val   -> {len(ds_dict['validation'])} examples")

if __name__ == "__main__":
    main()
