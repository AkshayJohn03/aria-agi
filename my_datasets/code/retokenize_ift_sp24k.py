#!/usr/bin/env python3
"""
prepare_all_sources_to_4k_sp24k.py

Discover, load, clean, tokenise and shard many dataset formats into fixed-length
4096-token torch shards for training with the SP-24k tokenizer.

Usage:
    python prepare_all_sources_to_4k_sp24k.py \
        --src . \
        --out datasets/processed/zia_superift_sp24k_4k \
        --spm artifacts/zia_tokenizer_v2_clean/zia_spm.model \
        --shard_size 5000

Requirements:
    pip install sentencepiece datasets tqdm pandas torch pyarrow
"""

import os
import re
import json
import glob
import math
import argparse
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict, Tuple, Iterable

import sentencepiece as spm
import torch
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk, Dataset as HFDataset

# -----------------------
# Utils & cleaning
# -----------------------
HTML_TAG_RE = re.compile(r"<[^>]+>")
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
MULTI_WS_RE = re.compile(r"\s+")
# remove unwanted special chars commonly seen in your outputs
WEIRD_TOK_RE = re.compile(r"[ĊĠĻŁŢŊÞðłĝć…‹›“”‘’•—–·†‡ºª]")

def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return ""

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # Normalization pass
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # remove HTML
    s = HTML_TAG_RE.sub(" ", s)
    # remove weird control characters
    s = CONTROL_RE.sub(" ", s)
    # remove weird token artefacts we saw (Ċ, Ġ, etc)
    s = WEIRD_TOK_RE.sub(" ", s)
    # collapse whitespace
    s = MULTI_WS_RE.sub(" ", s).strip()
    return s

def to_chatml(user_text: str, assistant_text: str = None) -> str:
    # Minimal ChatML wrapper consistent with your earlier format
    if assistant_text is None:
        assistant_text = ""
    # use clear tokens; ensure no leftover special chars
    user_text = clean_text(user_text)
    assistant_text = clean_text(assistant_text)
    return f"[BOS] <|user|> {user_text} <|assistant|> {assistant_text} [EOS]"

# -----------------------
# File discovery & loaders
# -----------------------
def find_source_files(root: str) -> List[Path]:
    rootp = Path(root)
    exts = ["**/*.json", "**/*.jsonl", "**/*.txt", "**/*.csv", "**/*.parquet", "**/*.arrow"]
    found = []
    for pat in exts:
        found.extend(list(rootp.glob(pat)))
    # also detect huggingface dataset folders (containing dataset_info.json or arrow files)
    for d in rootp.rglob("*"):
        if d.is_dir():
            if (d / "dataset_info.json").exists() or any((d / f).suffix == ".arrow" for f in d.iterdir() if f.is_file()):
                found.append(d)
    # unique and sort
    unique = sorted(set(found), key=lambda p: str(p))
    return unique

def load_jsonl(path: Path) -> List[Dict]:
    out = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    # sometimes lines contain trailing commas - be permissive
                    try:
                        out.append(json.loads(line.rstrip(",")))
                    except Exception:
                        continue
    except Exception:
        # fallback reading binary
        txt = safe_read_text(path)
        for line in txt.splitlines():
            line = line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def load_json(path: Path) -> List[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # support list of objects or dict with common keys
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # try common fields: 'data', 'examples'
                for k in ("data", "examples", "rows"):
                    if k in data and isinstance(data[k], list):
                        return data[k]
                # else wrap
                return [data]
    except Exception:
        return []

def load_csv(path: Path) -> List[Dict]:
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        try:
            df = pd.read_csv(path, dtype=str, engine="python", encoding="latin-1")
        except Exception:
            return []
    df = df.fillna("")
    return df.to_dict(orient="records")

def load_parquet(path: Path) -> List[Dict]:
    try:
        df = pd.read_parquet(path)
        df = df.fillna("")
        return df.to_dict(orient="records")
    except Exception:
        return []

def load_arrow(path: Path) -> List[Dict]:
    try:
        ds = load_from_disk(str(path)) if (path / "dataset_info.json").exists() else load_from_disk(str(path.parent))
        # if ds is DatasetDict, pick first split; else convert
        if isinstance(ds, dict) or isinstance(ds, DatasetDict):
            # flatten first split found
            for k in ds:
                return [dict(item) for item in ds[k]]
        if isinstance(ds, HFDataset):
            return [dict(x) for x in ds]
    except Exception:
        # try reading .arrow file by pyarrow
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(str(path))
            df = table.to_pandas()
            df = df.fillna("")
            return df.to_dict(orient="records")
        except Exception:
            return []
    return []

def load_hf_dataset_dir(path: Path) -> List[Dict]:
    try:
        ds = load_from_disk(str(path))
        # if datasetdict
        if isinstance(ds, DatasetDict):
            out = []
            for k in ds:
                out.extend([dict(x) for x in ds[k]])
            return out
        elif isinstance(ds, HFDataset):
            return [dict(x) for x in ds]
    except Exception:
        return []

# -----------------------
# Schema extraction heuristics
# -----------------------
def extract_text_pairs(record: dict) -> Tuple[str, str]:
    """
    Try to extract (user, assistant) text from a record with many possible schemas.
    Return (user_text, assistant_text). If only one side found, assistant_text may be empty.
    """
    # common keys
    possible_user_keys = ["instruction", "prompt", "input", "user", "question", "context", "text"]
    possible_assistant_keys = ["response", "output", "answer", "completion", "assistant", "reply"]
    # if it's already chatml content
    for k in list(record.keys()):
        v = record.get(k)
        if isinstance(v, str) and "[BOS]" in v and "<|user|>" in v:
            s = clean_text(v)
            # try split
            m = re.split(r"<\|assistant\|>", s)
            if len(m) >= 2:
                u = m[0].split("<|user|>")[-1].strip()
                a = m[1].replace("[EOS]", "").strip()
                return u, a
    # prefer explicit pair fields
    for uk in possible_user_keys:
        for ak in possible_assistant_keys:
            if uk in record and ak in record:
                return clean_text(record.get(uk, "")), clean_text(record.get(ak, ""))
    # try single-field styles
    for uk in possible_user_keys:
        if uk in record:
            return clean_text(record.get(uk, "")), ""
    for ak in possible_assistant_keys:
        if ak in record:
            return "", clean_text(record.get(ak, ""))
    # as fallback, try combining 'text' or first string field
    for k, v in record.items():
        if isinstance(v, str) and len(v) > 20:
            return clean_text(v), ""
    return "", ""

# -----------------------
# Tokenizer wrapper
# -----------------------
class SPTokenizer:
    def __init__(self, sp_path: str):
        self.sp = spm.SentencePieceProcessor()
        ok = self.sp.load(sp_path)
        if not ok:
            raise RuntimeError(f"Failed to load sentencepiece model: {sp_path}")
    def encode_ids(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)
    def decode_ids(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

# -----------------------
# Main pipeline
# -----------------------
def process_sources(
    src_root: str,
    sp_model_path: str,
    out_dir: str,
    shard_size: int = 5000,
    max_len: int = 4096,
    save_arrow: bool = False,
    verbose: bool = True
):
    src_root = Path(src_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_items: List[Tuple[str, str]] = []  # list of (user, assistant) strings
    seen_hash = set()

    sp_tok = SPTokenizer(sp_model_path)

    files = find_source_files(src_root)
    if verbose:
        print(f"[i] Discovered {len(files)} source files/dirs under {src_root}")

    for p in tqdm(files, desc="sources"):
        try:
            recs = []
            if p.is_dir():
                recs = load_hf_dataset_dir(p)
            else:
                ext = p.suffix.lower()
                if ext == ".jsonl":
                    recs = load_jsonl(p)
                elif ext == ".json":
                    recs = load_json(p)
                elif ext == ".csv":
                    recs = load_csv(p)
                elif ext == ".parquet":
                    recs = load_parquet(p)
                elif ext == ".arrow":
                    recs = load_arrow(p)
                elif ext == ".txt":
                    txt = safe_read_text(p)
                    # heuristics: each line may be a sample or whole file
                    lines = [l.strip() for l in txt.splitlines() if l.strip()]
                    # if many lines, treat as one sample per line into user side
                    if len(lines) > 1 and len(lines) < 200000:
                        for ln in lines:
                            recs.append({"text": ln})
                    else:
                        recs.append({"text": txt})
                else:
                    # unknown file type - try reading text
                    txt = safe_read_text(p)
                    if txt:
                        recs.append({"text": txt})
            if not recs:
                continue

            # heuristics to extract pairs
            for r in recs:
                if not isinstance(r, dict):
                    continue
                u, a = extract_text_pairs(r)
                # if both blank skip
                if not u and not a:
                    continue
                # build chatml
                chatml = to_chatml(u, a)
                # dedupe by hashed chatml
                h = hash(chatml)
                if h in seen_hash:
                    continue
                seen_hash.add(h)
                tmp_items.append((u, a))
        except Exception as e:
            # continue on errors
            print(f"[warn] error loading {p}: {e}")
            continue

    if verbose:
        print(f"[i] Total extracted pairs (unique): {len(tmp_items)}")

    # optional: shuffle
    import random
    random.shuffle(tmp_items)

    # convert to token ids and create fixed-length tensors
    input_ids_shards = []
    labels_shards = []
    cur_input_ids = []
    cur_labels = []

    def make_label_from_input(ids: List[int], seq_len: int):
        # labels: next-token prediction, we will align by leaving -100 for pad
        # For fixed-length we will produce labels same length with -100 padded
        lab = ids[:]  # same length as ids
        # shift not applied here because model loss expects logits[:, :-1] vs labels[:, 1:]
        # we'll store labels as ids (already aligned), training code uses labels[:,1:] .
        return lab

    total = len(tmp_items)
    pbar = tqdm(total=total, desc="tokenizing")
    for (u, a) in tmp_items:
        # create ChatML sample (user + assistant text)
        chat = to_chatml(u, a)
        ids = sp_tok.encode_ids(chat)
        if len(ids) == 0:
            pbar.update(1)
            continue
        # truncate to max_len
        if len(ids) > max_len:
            ids = ids[:max_len]
        # pad to fixed-length now (pad id assumed to be 0 in many sp models; we will check)
        pad_id = 0
        pad_len = max_len - len(ids)
        input_ids = ids + [pad_id] * pad_len
        labels = make_label_from_input(ids, max_len) + [-100] * pad_len

        cur_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
        cur_labels.append(torch.tensor(labels, dtype=torch.long))

        if len(cur_input_ids) >= shard_size:
            # stack and save
            input_tensor = torch.stack(cur_input_ids)
            label_tensor = torch.stack(cur_labels)
            shard_idx = len(input_ids_shards)
            shard_dir = out_dir / "train" / f"shard_{shard_idx:03d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            torch.save(input_tensor, shard_dir / "input_ids.pt")
            torch.save(label_tensor, shard_dir / "labels.pt")
            input_ids_shards.append(shard_dir)
            cur_input_ids = []
            cur_labels = []
        pbar.update(1)
    pbar.close()

    # flush remainder
    if cur_input_ids:
        shard_idx = len(input_ids_shards)
        shard_dir = out_dir / "train" / f"shard_{shard_idx:03d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.stack(cur_input_ids), shard_dir / "input_ids.pt")
        torch.save(torch.stack(cur_labels), shard_dir / "labels.pt")
        input_ids_shards.append(shard_dir)

    # split off validation set: take last shard N samples -> create val folder
    # Simple rule: 1,000 samples to validation if enough
    val_n = 1000
    total_samples = sum(torch.load(p / "input_ids.pt").shape[0] for p in input_ids_shards)
    if total_samples >= (val_n + 1):
        # collect last val_n from last shards
        taken = 0
        val_list_inputs = []
        val_list_labels = []
        # iterate shards from last to first popping samples
        for shard_dir in reversed(input_ids_shards):
            arr_in = torch.load(shard_dir / "input_ids.pt")
            arr_lbl = torch.load(shard_dir / "labels.pt")
            n = arr_in.shape[0]
            need = min(n, val_n - taken)
            if need <= 0:
                break
            # take tail samples
            val_list_inputs.insert(0, arr_in[-need:])
            val_list_labels.insert(0, arr_lbl[-need:])
            # keep the rest back
            keep_idx = n - need
            if keep_idx > 0:
                torch.save(arr_in[:keep_idx], shard_dir / "input_ids.pt")
                torch.save(arr_lbl[:keep_idx], shard_dir / "labels.pt")
            else:
                # remove shard entirely
                try:
                    os.remove(shard_dir / "input_ids.pt")
                    os.remove(shard_dir / "labels.pt")
                    os.rmdir(shard_dir)
                    input_ids_shards.remove(shard_dir)
                except Exception:
                    pass
            taken += need
            if taken >= val_n:
                break
        if taken > 0:
            # concat val and save as single shard val/shard_000
            val_dir = out_dir / "val" / "shard_000"
            val_dir.mkdir(parents=True, exist_ok=True)
            vi = torch.cat(val_list_inputs, dim=0)
            vl = torch.cat(val_list_labels, dim=0)
            torch.save(vi, val_dir / "input_ids.pt")
            torch.save(vl, val_dir / "labels.pt")

    # optionally create huggingface arrow dataset for inspection (small index)
    if save_arrow:
        all_records = []
        for shard_dir in sorted(list((out_dir / "train").glob("shard_*"))):
            ids_tensor = torch.load(shard_dir / "input_ids.pt")  # [N, L]
            lbl_tensor = torch.load(shard_dir / "labels.pt")
            N, L = ids_tensor.shape
            for i in range(N):
                all_records.append({"input_ids": ids_tensor[i].tolist(), "labels": lbl_tensor[i].tolist()})
        if all_records:
            ds = Dataset.from_list(all_records)
            # save
            ds.save_to_disk(str(out_dir / "arrow_dataset"))
            print(f"[i] Saved HF arrow dataset → {out_dir / 'arrow_dataset'}")

    print(f"[DONE] Saved training shards under: {out_dir / 'train'}")
    print(f"[DONE] Saved validation shards under: {out_dir / 'val'}")

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Aggregate many source files → fixed-length 4k token shards (SP tokenizer)")
    parser.add_argument("--src", type=str, default=".", help="root to scan (recursively)")
    parser.add_argument("--spm", type=str, required=True, help="SentencePiece model (.model) path")
    parser.add_argument("--out", type=str, required=True, help="output folder for shards")
    parser.add_argument("--shard_size", type=int, default=5000, help="samples per shard")
    parser.add_argument("--max_len", type=int, default=4096, help="fixed sequence length (tokens)")
    parser.add_argument("--save_arrow", action="store_true", help="also save HF arrow dataset for inspection")
    args = parser.parse_args()

    process_sources(
        src_root=args.src,
        sp_model_path=args.spm,
        out_dir=args.out,
        shard_size=args.shard_size,
        max_len=args.max_len,
        save_arrow=args.save_arrow,
        verbose=True
    )

if __name__ == "__main__":
    main()
