#!/usr/bin/env python3
"""
Stream-retokenize Arrow -> per-shard tensors.
Creates folders: datasets/processed/chatml_60k_4096_stream/train_shard_0000/
  each folder contains input_ids.pt and labels.pt (torch.LongTensor: [SHARD_SIZE, SEQ_LEN])
Memory-safe: token_buffer is flushed regularly.
"""
import os, random, gc
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

# CONFIG (edit paths if needed)
ARROW_DIR = "datasets/local/openorca"
OUT_DIR = "datasets/processed/chatml_60k_4096_stream"
TOKENIZER_DIR = "artifacts/zia_tokenizer_60k_clean"
SEQ_LEN = 4096
SHARD_SIZE = 1000         # sequences per shard folder
FLUSH_TOKENS = SEQ_LEN * SHARD_SIZE
VAL_RATIO = 0.005         # ~0.5% shards become val
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)

def iter_arrow_file(path):
    try:
        ds = Dataset.from_file(str(path)).with_format("python")
        for row in ds:
            yield row
    except Exception as e:
        print(f"⚠️ error reading {path}: {e}")

def extract_text(row):
    sys = row.get("system_prompt") or row.get("system") or row.get("instruction") or ""
    q = row.get("question") or row.get("prompt") or row.get("input") or row.get("text") or ""
    a = row.get("response") or row.get("output") or row.get("answer") or ""
    def s(x): return str(x) if x is not None else ""
    return s(sys).strip(), s(q).strip(), s(a).strip()

def format_chatml(sys_t, q_t, a_t):
    parts = []
    if sys_t: parts.append(f"<|im_start|>system\n{sys_t}<|im_end|>")
    if q_t:   parts.append(f"<|im_start|>user\n{q_t}<|im_end|>")
    if a_t:   parts.append(f"<|im_start|>assistant\n{a_t}<|im_end|>")
    return "\n".join(parts)

def save_shard(tensor_data, pad_id, out_dir, prefix, idx):
    path = Path(out_dir) / f"{prefix}_shard_{idx:04d}"
    path.mkdir(parents=True, exist_ok=True)
    labels = tensor_data.clone()
    labels[labels == pad_id] = -100
    torch.save(tensor_data, path / "input_ids.pt")
    torch.save(labels, path / "labels.pt")
    return path

def main():
    tok = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    files = sorted(Path(ARROW_DIR).glob("*.arrow"))
    token_buffer = []
    train_idx = 0
    val_idx = 0

    for fp in files:
        iterator = iter_arrow_file(fp)
        pbar = tqdm(iterator, desc=f"Tokenizing {fp.name}", unit="rows", dynamic_ncols=True)
        for row in pbar:
            sys_t, q_t, a_t = extract_text(row)
            if not q_t and not a_t:
                continue
            text = format_chatml(sys_t, q_t, a_t)
            ids = tok.encode(text, add_special_tokens=False)
            if not ids:
                continue
            token_buffer.extend(ids)

            # flush while loop
            while len(token_buffer) >= FLUSH_TOKENS:
                chunk = token_buffer[:FLUSH_TOKENS]
                token_buffer = token_buffer[FLUSH_TOKENS:]
                tensor = torch.tensor(chunk, dtype=torch.long).view(SHARD_SIZE, SEQ_LEN)
                if random.random() < VAL_RATIO:
                    save_shard(tensor, pad_id, OUT_DIR, "val", val_idx)
                    val_idx += 1
                else:
                    save_shard(tensor, pad_id, OUT_DIR, "train", train_idx)
                    train_idx += 1
                del tensor
                gc.collect()

    # leftover handling
    if len(token_buffer) >= SEQ_LEN:
        num_seqs = len(token_buffer) // SEQ_LEN
        token_buffer = token_buffer[:num_seqs * SEQ_LEN]
        tensor = torch.tensor(token_buffer, dtype=torch.long).view(num_seqs, SEQ_LEN)
        save_shard(tensor, pad_id, OUT_DIR, "train", train_idx)
        train_idx += 1
        del tensor
        gc.collect()

    print(f"Done. Train shards: {train_idx}, Val shards: {val_idx}, Saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
