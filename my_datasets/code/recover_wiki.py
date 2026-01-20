#!/usr/bin/env python3
"""
recover_wiki_v2.py
Load wiki tokenized shards (arrow / dataset folders), decode with tokenizer,
aggressively clean, and save a single newline-delimited clean corpus for tokenizer training.
"""

import os, glob, re, sys
from tqdm import tqdm
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer

# CONFIG - adjust paths if needed
SHARD_ROOT = "artifacts/processed/wiki_tokenized"
TOKENIZER_PATH = "artifacts/zia_tokenizer_60k"
OUT_DIR = "artifacts/wiki_recovered_clean"
OUT_FILE = os.path.join(OUT_DIR, "wiki_clean_corpus.txt")
os.makedirs(OUT_DIR, exist_ok=True)

# CLEANING regexes (aggressive)
_html = re.compile(r"<[^>]+>")
_wiki_braces = re.compile(r"\{\{.*?\}\}", re.DOTALL)
_wiki_links = re.compile(r"\[\[.*?\]\]")
_url = re.compile(r"http[s]?://\S+")
_non_eng = re.compile(r"[^A-Za-z0-9 ,.'\"\-?!;:()\n]")
_multi_space = re.compile(r"\s+")

def clean_text(t: str) -> str:
    t = _html.sub(" ", t)
    t = _wiki_braces.sub(" ", t)
    t = _wiki_links.sub(" ", t)
    t = _url.sub(" ", t)
    # remove weird unicode / control chars
    t = _non_eng.sub(" ", t)
    t = _multi_space.sub(" ", t)
    t = t.strip()
    return t

def find_shard_dirs(root):
    # pick directories matching wiki_chunk_*
    candidates = sorted(glob.glob(os.path.join(root, "*")))
    dirs = [p for p in candidates if os.path.isdir(p)]
    return dirs

def load_shard(path):
    """
    Try loading path with datasets.load_from_disk.
    If that fails, try to find arrow files inside and load the folder.
    """
    try:
        ds = load_from_disk(path)
        return ds
    except Exception as e:
        # fallback: if path contains .arrow files, try load_from_disk on parent
        arrow_files = glob.glob(os.path.join(path, "*.arrow"))
        if arrow_files:
            try:
                ds = load_from_disk(path)
                return ds
            except Exception:
                pass
    raise RuntimeError(f"Could not load shard as a dataset: {path}")

def main():
    print("=== Recovering + Cleaning Wikipedia Text (v2) ===")
    # tokenizer
    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer at {TOKENIZER_PATH}: {e}")
        sys.exit(1)

    pad_tok = None
    if tok.pad_token is None:
        try:
            tok.add_special_tokens({"pad_token": "<pad>"})
        except Exception:
            pass

    shard_dirs = find_shard_dirs(SHARD_ROOT)
    if not shard_dirs:
        print(f"[i] No shard directories found under {SHARD_ROOT}. Exiting.")
        # create empty file to keep downstream happy
        open(OUT_FILE, "w", encoding="utf-8").close()
        return

    print(f"[i] Found {len(shard_dirs)} shard dirs (processing each)...")
    written = 0
    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        for shard in tqdm(shard_dirs, desc="shards"):
            try:
                ds = load_shard(shard)
            except Exception as e:
                tqdm.write(f"[WARN] Skipping shard (load failed): {shard} | {e}")
                continue

            # Expect each dataset row to contain either 'input_ids' (list of ints) or 'text'
            if "input_ids" in ds.column_names:
                # iterate rows
                for row in ds["input_ids"]:
                    # row may be list or numpy array
                    try:
                        # decode with tokenizer; skip special tokens
                        txt = tok.decode(list(row), skip_special_tokens=True)
                    except Exception:
                        # if decode fails, skip
                        continue
                    cleaned = clean_text(txt)
                    if len(cleaned) < 60:  # discard tiny lines
                        continue
                    if cleaned.count(" ") < 5:
                        continue
                    fout.write(cleaned + "\n")
                    written += 1
            elif "text" in ds.column_names:
                for txt in ds["text"]:
                    cleaned = clean_text(txt)
                    if len(cleaned) < 60:
                        continue
                    if cleaned.count(" ") < 5:
                        continue
                    fout.write(cleaned + "\n")
                    written += 1
            else:
                tqdm.write(f"[WARN] shard missing expected columns (input_ids/text): {shard}")

            # free memory
            del ds

    print(f"[✓] Done. Wrote {written} cleaned lines → {OUT_FILE}")

if __name__ == "__main__":
    main()
