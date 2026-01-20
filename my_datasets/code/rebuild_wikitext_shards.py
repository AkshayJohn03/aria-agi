#!/usr/bin/env python3
"""
Stream-tokenize WikiText -> per-shard tensors (input_ids.pt + labels.pt).
Labels are shifted (next-token targets) and pad tokens -> -100.
Produces folders like:
  datasets/processed/wikitext_60k_4096/train_shard_0000/{input_ids.pt, labels.pt}
"""
import os, gc, random
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

# CONFIG
TOKENIZER_DIR = "artifacts/zia_tokenizer_60k_clean"
OUT_DIR = "datasets/processed/wikitext_60k_4096"
SEQ_LEN = 4096
SHARD_SIZE = 1000     # sequences per shard folder
FLUSH_TOKENS = SEQ_LEN * SHARD_SIZE
VAL_RATIO = 0.005     # small val split if desired
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)

def save_shard(tensor_seqs, pad_id, out_dir, prefix, idx):
    path = Path(out_dir) / f"{prefix}_shard_{idx:04d}"
    path.mkdir(parents=True, exist_ok=True)
    input_tensor = torch.tensor(tensor_seqs, dtype=torch.long)
    # build labels = next-token (shifted)
    labels = input_tensor.clone()
    labels[:, :-1] = input_tensor[:, 1:]
    # last token has no next token -> mark as -100 (ignore)
    labels[:, -1] = -100
    # mask pads as -100
    if pad_id is not None:
        labels[labels == pad_id] = -100
    torch.save(input_tensor, path / "input_ids.pt")
    torch.save(labels, path / "labels.pt")
    return path

def encode_and_flush(token_buffer, tok, pad_id, out_dir, train_idx, val_idx):
    shards_written = 0
    while len(token_buffer) >= FLUSH_TOKENS:
        chunk = token_buffer[:FLUSH_TOKENS]
        token_buffer = token_buffer[FLUSH_TOKENS:]
        # reshape into sequences
        tensor = torch.tensor(chunk, dtype=torch.long).view(SHARD_SIZE, SEQ_LEN)
        if random.random() < VAL_RATIO:
            save_shard(tensor, pad_id, out_dir, "val", val_idx)
            val_idx += 1
        else:
            save_shard(tensor, pad_id, out_dir, "train", train_idx)
            train_idx += 1
        shards_written += 1
        # free
        del tensor
        gc.collect()
    return token_buffer, train_idx, val_idx, shards_written

def main():
    print("[i] Loading tokenizer:", TOKENIZER_DIR)
    tok = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else None
    eos_id = tok.eos_token_id

    print("[i] Loading WikiText-103 (raw)...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    token_buffer = []
    train_idx = 0
    val_idx = 0
    total_shards = 0

    for row in tqdm(ds, desc="Tokenizing", unit="rows", dynamic_ncols=True):
        text = row.get("text", "")
        if not text or len(text.strip()) < 10:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        if not ids:
            continue
        token_buffer.extend(ids)
        # optionally separate articles
        if eos_id is not None:
            token_buffer.append(eos_id)

        token_buffer, train_idx, val_idx, shards_written = encode_and_flush(
            token_buffer, tok, pad_id, OUT_DIR, train_idx, val_idx
        )
        total_shards += shards_written

    # flush leftovers if enough for at least one sequence
    while len(token_buffer) >= SEQ_LEN:
        take = (len(token_buffer) // SEQ_LEN) * SEQ_LEN
        chunk = token_buffer[:take]
        token_buffer = token_buffer[take:]
        tensor = torch.tensor(chunk, dtype=torch.long).view(-1, SEQ_LEN)
        for i in range(tensor.size(0)):
            seq = tensor[i : i+1]  # shape [1, SEQ_LEN]
            if random.random() < VAL_RATIO:
                save_shard(seq, pad_id, OUT_DIR, "val", val_idx)
                val_idx += 1
            else:
                save_shard(seq, pad_id, OUT_DIR, "train", train_idx)
                train_idx += 1
            total_shards += 1
        del tensor
        gc.collect()

    print(f"\n✅ Done. Train shards: {train_idx}, Val shards: {val_idx}, Saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
