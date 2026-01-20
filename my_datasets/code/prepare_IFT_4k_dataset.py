#!/usr/bin/env python3
import os, time, glob, json, torch
from datasets import load_from_disk
from tqdm import tqdm

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DATASET_DIR = "datasets/processed/zia_superift_v2_balanced"
OUT_DIR = "datasets/processed/zia_superift_4k_torch"
TOKENIZER_DIR = "artifacts/zia_tokenizer_60k"
FIXED_LEN = 4096

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def read_json_utf8(path):
    """Read JSON files safely on Windows."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_pad_id():
    """Resolve PAD token ID robustly across HF tokenizer formats."""
    tok_json_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    tok_cfg_path = os.path.join(TOKENIZER_DIR, "tokenizer_config.json")
    spm_map_path = os.path.join(TOKENIZER_DIR, "special_tokens_map.json")

    pad_id = None

    # 1) Check special_tokens_map.json
    if os.path.exists(spm_map_path):
        try:
            data = read_json_utf8(spm_map_path)
            if "pad_token" in data:
                pad_id = data["pad_token"].get("id", None)
        except:
            pass

    # 2) Check tokenizer_config.json
    if pad_id is None and os.path.exists(tok_cfg_path):
        try:
            data = read_json_utf8(tok_cfg_path)
            if "pad_token_id" in data:
                pad_id = data["pad_token_id"]
        except:
            pass

    # 3) Check tokenizer.json
    if pad_id is None and os.path.exists(tok_json_path):
        try:
            data = read_json_utf8(tok_json_path)
            # Try model.vocab first
            if "added_tokens" in data:
                for tok in data["added_tokens"]:
                    if tok.get("content") == "<pad>":
                        pad_id = tok["id"]
                        break
        except:
            pass

    # Fallback (your tokenizer uses 60003)
    return 60003 if pad_id is None else pad_id

def pad_or_trunc(seq, pad_id, max_len):
    """Pad or truncate sequence."""
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_id] * (max_len - len(seq))

def convert_shard(shard_path, out_path, pad_id):
    """Convert dataset shard to fixed-length .pt tensors."""
    print(f"\n[→] Processing shard: {shard_path}")
    ds = load_from_disk(shard_path)

    ids_list = []
    lbl_list = []

    for row in tqdm(ds, desc="shard"):
        ids = row["input_ids"]
        lbl = row["labels"]

        ids = pad_or_trunc(ids, pad_id, FIXED_LEN)
        lbl = pad_or_trunc(lbl, -100, FIXED_LEN)

        ids_list.append(torch.tensor(ids, dtype=torch.long))
        lbl_list.append(torch.tensor(lbl, dtype=torch.long))

    ids_tensor = torch.stack(ids_list)
    lbl_tensor = torch.stack(lbl_list)

    os.makedirs(out_path, exist_ok=True)
    torch.save(ids_tensor, os.path.join(out_path, "input_ids.pt"))
    torch.save(lbl_tensor, os.path.join(out_path, "labels.pt"))

    print(f"[✓] Saved: {out_path}/input_ids.pt  shape={ids_tensor.shape}")
    print(f"[✓] Saved: {out_path}/labels.pt     shape={lbl_tensor.shape}")

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    print("\n=== PREPARING FIXED 4K TORCH DATASET ===")

    pad_id = get_pad_id()
    print(f"[i] PAD ID detected: {pad_id}")

    train_shards = sorted(glob.glob(os.path.join(DATASET_DIR, "train", "*")))
    train_shards = [s for s in train_shards if os.path.isdir(s)]

    print(f"[i] Found {len(train_shards)} train shards")

    for shard in train_shards:
        name = os.path.basename(shard)
        out_path = os.path.join(OUT_DIR, "train", name)
        convert_shard(shard, out_path, pad_id)

    val_shards = sorted(glob.glob(os.path.join(DATASET_DIR, "val", "*")))
    val_shards = [s for s in val_shards if os.path.isdir(s)]

    print(f"\n[i] Found {len(val_shards)} val shards")

    for shard in val_shards:
        name = os.path.basename(shard)
        out_path = os.path.join(OUT_DIR, "val", name)
        convert_shard(shard, out_path, pad_id)

    print("\n=== DONE: FIXED 4K TORCH DATASET READY ===")

if __name__ == "__main__":
    main()
