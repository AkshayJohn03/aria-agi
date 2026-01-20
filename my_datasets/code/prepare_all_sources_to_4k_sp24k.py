#!/usr/bin/env python3
"""
prepare_all_sources_to_4k_sp24k_v4_torch.py

Torch-only, Windows-safe dataset preparation (Option B: per-shard meta.json).
Saves shards as:
  out/train/shard_000/input_ids.pt
  out/train/shard_000/labels.pt
  out/train/shard_000/meta.json

Requirements:
  pip install sentencepiece torch datasets tqdm

Usage example:
python prepare_all_sources_to_4k_sp24k_v4_torch.py \
  --src datasets/local/dolly datasets/local/openorca \
  --spm artifacts/zia_tokenizer_v2_clean/zia_spm.model \
  --out datasets/processed/zia_superift_sp24k_4k_v4 \
  --max_len 4096 --shard_size 1000 --flush_size 512 --val_frac 0.02 \
  --batch_tok 32 --workers 6 --seed 42 --resume
"""
import argparse
import json
import os
import re
import random
import tempfile
import time
from pathlib import Path
from functools import partial
import concurrent.futures
from hashlib import sha1

import torch
import sentencepiece as spm

try:
    from datasets import load_from_disk, load_dataset
    _HAS_HF = True
except Exception:
    _HAS_HF = False

from tqdm import tqdm

# ---------------- cleaning ----------------
HTML = re.compile(r"<[^>]+>")
SPECIAL = re.compile(r"[ĊĠĻŁŢŊÞðłĝć…‹›“”‘’•—–·†‡ºª]")
CTRL = re.compile(r"[\x00-\x1f\x7f]")
MULTI_WS = re.compile(r"\s+")

def clean_text(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    s = HTML.sub(" ", s)
    s = SPECIAL.sub(" ", s)
    s = CTRL.sub(" ", s)
    s = MULTI_WS.sub(" ", s)
    s = s.strip()
    s = ''.join(ch if (32 <= ord(ch) <= 126 or ch in "—–“”’‘…•") else " " for ch in s)
    s = MULTI_WS.sub(" ", s).strip()
    return s

# ---------- record extraction heuristic ----------
def extract_pair_from_record(rec):
    if not isinstance(rec, dict):
        return None, None
    user_keys = ["user", "prompt", "instruction", "input", "question", "context"]
    assistant_keys = ["assistant", "response", "output", "answer", "completion"]
    for uk in user_keys:
        for ak in assistant_keys:
            if uk in rec and ak in rec and isinstance(rec[uk], str) and isinstance(rec[ak], str):
                return clean_text(rec[uk]), clean_text(rec[ak])
    if "text" in rec and isinstance(rec["text"], str):
        return clean_text(rec["text"]), ""
    if "conversations" in rec and isinstance(rec["conversations"], list):
        conv = rec["conversations"]
        if len(conv) >= 2 and isinstance(conv[-2], dict) and isinstance(conv[-1], dict):
            u = conv[-2].get("value")
            a = conv[-1].get("value")
            if isinstance(u, str):
                return clean_text(u), clean_text(a or "")
    str_fields = [v for v in rec.values() if isinstance(v, str) and len(v) > 10]
    if len(str_fields) >= 2:
        return clean_text(str_fields[0]), clean_text(str_fields[1])
    if len(str_fields) == 1:
        return clean_text(str_fields[0]), ""
    return None, None

# ---------- iterators ----------
def iter_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return
    if isinstance(data, list):
        for rec in data:
            u,a = extract_pair_from_record(rec)
            if u is not None: yield u,a
    elif isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            for rec in data["data"]:
                u,a = extract_pair_from_record(rec)
                if u is not None: yield u,a
        else:
            u,a = extract_pair_from_record(data)
            if u is not None: yield u,a

def iter_jsonl(path: Path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln: continue
                try: rec = json.loads(ln)
                except Exception: continue
                u,a = extract_pair_from_record(rec)
                if u is not None: yield u,a
    except Exception:
        return

def iter_txt(path: Path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln: continue
                txt = clean_text(ln)
                if len(txt) > 5: yield txt, ""
    except Exception:
        return

def iter_hf_dataset_dir(path: Path):
    if not _HAS_HF: return
    try:
        ds = load_from_disk(str(path))
    except Exception:
        return
    # Dataset or DatasetDict
    if hasattr(ds, "__len__") and hasattr(ds, "__getitem__"):
        for rec in ds:
            u,a = extract_pair_from_record(rec)
            if u is not None: yield u,a
    else:
        for k in ds.keys():
            for rec in ds[k]:
                u,a = extract_pair_from_record(rec)
                if u is not None: yield u,a

def iter_arrow_or_parquet(path: Path):
    if not _HAS_HF: return
    try:
        ds = load_dataset("arrow", data_files=str(path))
    except Exception:
        try:
            ds = load_dataset("parquet", data_files=str(path))
        except Exception:
            return
    if hasattr(ds, "__len__") and hasattr(ds, "__getitem__"):
        for rec in ds:
            u,a = extract_pair_from_record(rec)
            if u is not None: yield u,a
    else:
        for k in ds.keys():
            for rec in ds[k]:
                u,a = extract_pair_from_record(rec)
                if u is not None: yield u,a

def file_iterator(path: Path):
    if not path.exists(): return
    if path.is_dir():
        if (path/"dataset_info.json").exists() or any(p.suffix==".arrow" for p in path.iterdir()):
            yield from iter_hf_dataset_dir(path)
        else:
            for f in sorted(path.iterdir()):
                yield from file_iterator(f)
    else:
        sfx = path.suffix.lower()
        if sfx == ".json": yield from iter_json(path)
        elif sfx in (".jsonl", ".ndjson"): yield from iter_jsonl(path)
        elif sfx in (".txt", ".md"): yield from iter_txt(path)
        elif sfx in (".arrow", ".parquet"): yield from iter_arrow_or_parquet(path)
        else:
            # try fallbacks
            yield from iter_jsonl(path)
            yield from iter_json(path)
            yield from iter_txt(path)

# ---------- ChatML ----------
def to_chatml(user: str, assistant: str):
    return f"[BOS] <|user|> {user} <|assistant|> {assistant} [EOS]"

# ---------- tokenization ----------
def tokenize_single(sp, text, out_type=int):
    try:
        return sp.encode(text, out_type=out_type)
    except Exception:
        try: return sp.encode_as_ids(text)
        except Exception: return []

def tokenize_batch(sp, text_list, max_workers=6, out_type=int):
    if not text_list: return []
    if len(text_list) < 200 or max_workers <= 1:
        return [tokenize_single(sp, t, out_type) for t in text_list]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(lambda t: sp.encode(t, out_type=out_type), text_list))
    except Exception:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(partial(tokenize_single, sp, out_type=out_type), text_list))

# ---------- atomic torch save FIX ----------
def atomic_torch_save(obj, target_path: Path, tries=3):
    target_path = Path(target_path)
    tmp = None
    # Use a more reliable temporary naming convention on Windows
    # based on time, PID, and a random number to ensure uniqueness.
    unique_suffix = f"{os.getpid()}_{int(time.time() * 1000)}_{random.randint(100, 999)}.tmp"
    tmp_path = target_path.parent / (target_path.name + "." + unique_suffix)
    
    for attempt in range(tries):
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = tmp_path
            
            # 1. Write to temporary file
            # Note: We rely on Path.unlink() to clean up if this fails.
            torch.save(obj, tmp)
            
            # 2. Atomic replace (rename). This is usually faster/more reliable than mkstemp + os.close + os.replace
            os.replace(str(tmp), str(target_path))
            return
        except Exception as e:
            # Clean up failed temporary file
            try:
                if tmp and tmp.exists(): tmp.unlink()
            except Exception:
                pass
                
            if attempt + 1 < tries:
                wait = 1 + attempt * 2
                # Use tqdm.write if available, otherwise print
                tqdm.write(f"[WARN] atomic_torch_save failed (attempt {attempt+1}/{tries}) -> {e}. retrying in {wait}s")
                time.sleep(wait)
                continue
            
            # Final attempt failed
            raise

# ---------- meta write ----------
def write_meta_atomic(meta: dict, target_path: Path, tries=3):
    target_path = Path(target_path)
    tmp = None
    s = json.dumps(meta, indent=2, ensure_ascii=False)
    for attempt in range(tries):
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmpname = tempfile.mkstemp(prefix=target_path.name + ".", dir=str(target_path.parent))
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(s)
            os.replace(tmpname, str(target_path))
            return
        except Exception as e:
            try:
                if tmp and Path(tmp).exists(): Path(tmp).unlink()
            except Exception:
                pass
            if attempt + 1 < tries:
                time.sleep(1 + attempt)
                continue
            raise

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--spm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--shard_size", type=int, default=1000, help="samples per shard to write")
    ap.add_argument("--flush_size", type=int, default=512, help="internal buffer flush size (<= shard_size)")
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_tok", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--print_samples", type=int, default=5)
    ap.add_argument("--resume", action="store_true", help="skip shards that already exist")
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm)
    try:
        pad_id = sp.piece_to_id("<pad>")
    except Exception:
        pad_id = 0

    outp = Path(args.out)
    train_dir = outp / "train"
    val_dir = outp / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print("[i] Scanning sources for samples...")
    samples = []
    total_found = 0
    for p in args.src:
        pth = Path(p)
        if not pth.exists():
            print(f"[WARN] source not found: {p}")
            continue
        for u,a in file_iterator(pth):
            total_found += 1
            if (u and len(u) > 3) or (a and len(a) > 3):
                samples.append((u or "", a or ""))
            if total_found % 10000 == 0:
                print(f"[i] scanned {total_found} records, collected {len(samples)} usable samples")

    print(f"[i] Total scanned records: {total_found}; usable samples: {len(samples)}")
    if len(samples) == 0:
        print("[!] No usable samples found. Exiting.")
        return

    # light dedupe
    seen = set()
    uniq = []
    for u,a in samples:
        k = (u + "\n" + a)[:1000]
        if k in seen: continue
        seen.add(k)
        uniq.append((u,a))
    print(f"[i] After light deduplication: {len(uniq)} samples")

    random.Random(args.seed).shuffle(uniq)
    n_val = max(1, int(len(uniq) * args.val_frac))
    val_samples = uniq[:n_val]
    train_samples = uniq[n_val:]
    print(f"[i] Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    def shard_exists(target_dir: Path, idx: int):
        sdir = target_dir / f"shard_{idx:03d}"
        return (sdir / "input_ids.pt").exists() and (sdir / "labels.pt").exists() and (sdir / "meta.json").exists()

    def make_shards(samples_list, target_dir: Path, prefix="shard"):
        buf_in = []
        buf_lbl = []
        shard_idx = 0
        total = len(samples_list)

        # if resume, find next missing shard_idx
        if args.resume:
            while shard_exists(target_dir, shard_idx):
                shard_idx += 1

        it = range(0, total, args.batch_tok)
        pbar = tqdm(it, desc=f"Tokenizing -> {target_dir.name}", total=(total + args.batch_tok -1)//args.batch_tok)
        for i in pbar:
            batch = samples_list[i:i+args.batch_tok]
            texts = [to_chatml(u,a) for u,a in batch]
            ids_batch = tokenize_batch(sp, texts, max_workers=args.workers, out_type=int)
            for ids in ids_batch:
                if len(ids) > args.max_len:
                    ids = ids[-args.max_len:]
                pad_len = args.max_len - len(ids)
                in_ids = ids + [pad_id]*pad_len
                lbls = ids + [-100]*pad_len
                buf_in.append(torch.tensor(in_ids, dtype=torch.long))
                buf_lbl.append(torch.tensor(lbls, dtype=torch.long))

                # flush to disk in flush_size batches to keep files small/resumable
                if len(buf_in) >= args.flush_size:
                    # if overwrite existing shard (resume false) start writing to current shard index
                    # build or append to current shard arrays until shard_size reached
                    while len(buf_in) > 0:
                        take = min(len(buf_in), args.shard_size)
                        chunk_in = buf_in[:take]
                        chunk_lbl = buf_lbl[:take]
                        # prepare shard dir
                        sdir = target_dir / f"{prefix}_{shard_idx:03d}"
                        sdir.mkdir(parents=True, exist_ok=True)
                        input_path = sdir / "input_ids.pt"
                        label_path = sdir / "labels.pt"
                        meta_path = sdir / "meta.json"

                        # if resume and shard exists, increment index and continue
                        if args.resume and shard_exists(target_dir, shard_idx):
                            shard_idx += 1
                            continue

                        # save tensors (atomic)
                        tensor_in = torch.stack(chunk_in, dim=0)
                        tensor_lbl = torch.stack(chunk_lbl, dim=0)
                        try:
                            atomic_torch_save(tensor_in, input_path)
                            atomic_torch_save(tensor_lbl, label_path)
                        except Exception as e:
                            raise RuntimeError(f"Failed to save shard {shard_idx}: {e}")

                        # shard meta
                        meta = {
                            "shard_index": shard_idx,
                            "samples": tensor_in.size(0),
                            "max_len": args.max_len,
                            "spm": str(args.spm),
                            "seed": args.seed,
                            "created_ts": int(time.time()),
                            "sha1_input": sha1(torch.flatten(tensor_in).numpy().tobytes()).hexdigest()[:12]
                        }
                        write_meta_atomic(meta, meta_path)
                        # remove used from buffer
                        buf_in[:take] = []
                        buf_lbl[:take] = []
                        shard_idx += 1
        # final flush (whatever remains)
        if buf_in:
            # may require multiple final shards
            while buf_in:
                take = min(len(buf_in), args.shard_size)
                chunk_in = buf_in[:take]
                chunk_lbl = buf_lbl[:take]
                sdir = target_dir / f"{prefix}_{shard_idx:03d}"
                sdir.mkdir(parents=True, exist_ok=True)
                input_path = sdir / "input_ids.pt"
                label_path = sdir / "labels.pt"
                meta_path = sdir / "meta.json"
                if args.resume and shard_exists(target_dir, shard_idx):
                    shard_idx += 1
                    continue
                tensor_in = torch.stack(chunk_in, dim=0)
                tensor_lbl = torch.stack(chunk_lbl, dim=0)
                atomic_torch_save(tensor_in, input_path)
                atomic_torch_save(tensor_lbl, label_path)
                meta = {
                    "shard_index": shard_idx,
                    "samples": tensor_in.size(0),
                    "max_len": args.max_len,
                    "spm": str(args.spm),
                    "seed": args.seed,
                    "created_ts": int(time.time()),
                    "sha1_input": sha1(torch.flatten(tensor_in).numpy().tobytes()).hexdigest()[:12]
                }
                write_meta_atomic(meta, meta_path)
                buf_in[:take] = []
                buf_lbl[:take] = []
                shard_idx += 1

        print(f"[✓] Wrote {shard_idx} shards to {target_dir}")

    # run
    make_shards(train_samples, train_dir, prefix="shard")
    make_shards(val_samples, val_dir, prefix="shard")

    # print verification samples
    print("\n[INFO] Printing random samples (decoded with spm) ...")
    rand = random.Random(args.seed)
    for _ in range(min(args.print_samples, len(uniq))):
        idx = rand.randrange(len(uniq))
        u,a = uniq[idx]
        chat = to_chatml(u,a)
        ids = sp.encode(chat, out_type=int)
        text_roundtrip = sp.decode(ids)
        print("----- SAMPLE -----")
        print("RAW PROMPT:", u[:300])
        print("RAW RESP:", a[:300])
        print("ROUNDTRIP (decoded):", text_roundtrip[:400])
        print()

    manifest = {
        "total_samples": len(uniq),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "max_len": args.max_len,
        "shard_size": args.shard_size,
        "flush_size": args.flush_size,
        "seed": args.seed,
        "spm": str(args.spm)
    }
    with open(outp / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"[✓] Finished. Manifest written to {outp / 'manifest.json'}")

if __name__ == "__main__":
    main()
