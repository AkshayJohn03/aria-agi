#!/usr/bin/env python3
"""
clean_ift_dataset.py

- Finds a tokenized HF dataset directory (or raw text folder) and builds a cleaned,
  re-tokenized IFT dataset using your exact tokenizer.
- Removes common provenance markers (e.g. "[ alp aca ]", "[ oasst ]"), normalizes
  "Instruction: ... Response: ..." format and re-tokenizes with the tokenizer at
  artifacts/zia_tokenizer_60k by default.
- Produces a new huggingface-dataset-on-disk (load_from_disk friendly) at output_dir.

Usage:
    python my_datasets/code/clean_ift_dataset.py `
        --input artifacts/tokenized_dataset/zia_ift_v3_verified `
        --tokenizer artifacts/zia_tokenizer_60k `
        --output artifacts/tokenized_dataset/zia_ift_v3_cleaned `
        --max_len 1024 `
        --chunk_stride 256

        

Notes:
- Does NOT overwrite the input dataset.
- If your dataset is already raw text files instead of HF dataset, pass --raw_text_dir
  and the script will read *.jsonl or *.txt lines (basic support).
"""

import os, re, argparse, math, json
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

# --- environment
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --- cleaning regexes & helpers ---
PROVENANCE_PATTERNS = [
    r"\[\s*alpaca\s*\]", r"\[\s*oasst\s*\]", r"\[\s*openorca\s*\]",
    r"\[ *alp *aca *\]", r"\[ *alp_aca *\]", r"\[ *alpaca\]", r"\[ *alpaca *\]",
    r"\{ *alpaca *\}", r"\[ *human\]", r"\[\s*helpful\s*\]"
]
PROV_RE = re.compile("|".join(PROVENANCE_PATTERNS), flags=re.IGNORECASE)

# Normalization helpers
def normalize_markers(s: str) -> str:
    # Replace weird 'Ċ' (sometimes tokenized newline) with newline
    s = s.replace("Ċ", "\n").replace("\\n", "\n")
    s = PROV_RE.sub("", s)
    # unify spacing and punctuation for Instruction/Response label
    s = re.sub(r"Instruction\s*[:\-]+\s*", "Instruction: ", s, flags=re.IGNORECASE)
    s = re.sub(r"Response\s*[:\-]+\s*", "Response: ", s, flags=re.IGNORECASE)
    # sometimes "Instruction :" variants
    s = re.sub(r"Instruction\s* :", "Instruction:", s)
    s = re.sub(r"Response\s* :", "Response:", s)
    # collapse multiple spaces
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n\s+", "\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = s.strip()
    return s

def enforce_structure(s: str):
    """
    Ensure exactly one 'Instruction:' and one 'Response:'.
    If the text doesn't contain the canonical structure, we try:
      1. Fixing casing issues (e.g., 'instruction' to 'Instruction').
      2. Split on '\n\n' as a final heuristic.

    Returns (instr, resp) or (None, None) if not recoverable.
    """
    
    # 1. Try canonical structure (Instruction: ... Response: ...)
    def try_split(text):
        if "Instruction:" in text and "Response:" in text:
            try:
                left, right = text.split("Instruction:", 1)[1].split("Response:", 1)
                instr = left.strip()
                resp = right.strip()
                if instr and resp:
                    return instr, resp
            except Exception:
                pass
        return None, None

    instr, resp = try_split(s)
    if instr is not None:
        return instr, resp

    # 2. Try fixing casing and re-running the check (Non-recursive fix for the bug)
    s_fixed = s.replace("response", "Response").replace("instruction", "Instruction")
    
    if s_fixed != s:
        instr, resp = try_split(s_fixed)
        if instr is not None:
            return instr, resp
            
    # 3. Fallback Heuristic: Split on first double newline
    if "\n\n" in s:
        first, rest = s.split("\n\n", 1)
        if first.strip() and rest.strip():
            # Only return if both parts are non-empty
            return first.strip(), rest.strip()

    # 4. Final fallback: no structure
    return None, None

# chunking helper: split tokens into sliding windows while ensuring we don't cut Instruction label
def chunk_token_ids(ids, max_len, stride):
    chunks = []
    L = len(ids)
    if L <= max_len:
        chunks.append(ids)
        return chunks
    start = 0
    while start < L:
        end = min(start + max_len, L)
        chunks.append(ids[start:end])
        if end == L:
            break
        start = max(0, end - stride)
    return chunks

# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, default="artifacts/tokenized_dataset/zia_ift_v3_verified",
                        help="HuggingFace dataset dir OR file pattern containing tokenized examples.")
    parser.add_argument("--raw_text_dir", type=str, default=None,
                        help="Optional: raw text dir with .jsonl/.txt lines. If provided, script will read these instead.")
    parser.add_argument("--tokenizer", type=str, default="artifacts/zia_tokenizer_60k")
    parser.add_argument("--output", type=str, default="artifacts/tokenized_dataset/zia_ift_v3_cleaned")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--chunk_stride", type=int, default=256,
                        help="When chunking long examples, how much stride to overlap (sliding window).")
    parser.add_argument("--sample_limit", type=int, default=500,
                        help="How many examples to decode & inspect for reports (per split).")
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    print(f"[i] Tokenizer loaded | vocab={len(tok)}, pad={tok.pad_token_id}, eos={tok.eos_token_id}")

    # load source dataset if available
    if args.raw_text_dir:
        # read raw .jsonl or .txt files
        raw_dir = Path(args.raw_text_dir)
        texts = []
        for p in raw_dir.glob("**/*"):
            if p.suffix.lower() in [".jsonl", ".txt", ".json"]:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                # try common fields
                                s = obj.get("text") or obj.get("content") or obj.get("instruction") or obj.get("input")
                                if s:
                                    texts.append(s)
                                else:
                                    texts.append(line)
                            else:
                                texts.append(line)
                        except Exception:
                            texts.append(line)
        print(f"[i] Found {len(texts)} raw text lines.")
        raw_ds = Dataset.from_dict({"text": texts})
        splits = {"train": raw_ds}
    else:
        # assume HF dataset saved with load_from_disk
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input dataset path not found: {args.input}")
        ds = load_from_disk(args.input)
        # ds may be DatasetDict or Dataset
        if isinstance(ds, dict) or "train" in ds:
            splits = {}
            for k in ds.keys():
                splits[k] = ds[k]
        else:
            # single dataset
            splits = {"train": ds}

    print("[i] Splits found:", list(splits.keys()))
    cleaned_splits = {}

    for split_name, split_ds in splits.items():
        print(f"\n--- Processing split: {split_name} | examples={len(split_ds)} ---")
        cleaned_records = []
        malformed = 0
        # We will iterate, decode if tokenized, else assume text field 'text' or 'input_text'
        for i, ex in enumerate(tqdm(split_ds, desc=f"decoding {split_name}")):
            # discover text content
            text = None
            # common fields -> adapt to your dataset variant
            for key in ("text", "input", "instruction", "prompt"):
                if key in ex and ex[key] is not None:
                    text = ex[key]
                    break
            # some tokenized datasets store input_ids already
            if text is None:
                if "input_ids" in ex:
                    # decode token ids to text
                    try:
                        text = tok.decode(ex["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    except Exception:
                        # if input_ids is bytes or list
                        try:
                            ids = ex["input_ids"]
                            if isinstance(ids, (list, tuple)):
                                text = tok.decode(ids)
                        except Exception:
                            text = None
            if text is None:
                # fallback try "labels" decode
                if "labels" in ex:
                    try:
                        text = tok.decode(ex["labels"])
                    except Exception:
                        text = None

            if text is None:
                # skip
                malformed += 1
                continue

            # clean and normalize
            text = normalize_markers(text)
            instr, resp = enforce_structure(text)
            if instr is None or resp is None:
                malformed += 1
                continue

            full = f"Instruction: {instr}\nResponse: {resp}"
            # basic length sanity
            if len(full.strip()) == 0:
                malformed += 1
                continue

            # tokenize to ids (do not truncate here; we'll chunk if needed)
            enc = tok(full, add_special_tokens=True, return_attention_mask=False)
            ids = enc["input_ids"]
            # chunk long examples into sliding windows so we can train on long contexts safely
            chunks = chunk_token_ids(ids, args.max_len, args.chunk_stride)
            for c in chunks:
                cleaned_records.append({"input_ids": c, "text": full})  # keep text for debugging

        print(f"[i] {split_name}: malformed/skipped={malformed} / kept={len(cleaned_records)}")

        # build HF dataset (variable-length token lists)
        # to save space we store input_ids and text; labels will be produced at collate time or here as copy
        ds_out = Dataset.from_dict({
            "input_ids": [r["input_ids"] for r in cleaned_records],
            "text": [r["text"] for r in cleaned_records]
        })
        cleaned_splits[split_name] = ds_out

        # quick sample decode
        for i in range(min(args.sample_limit, len(ds_out))):
            if i >= 5:
                break
            row = ds_out[i]
            # Fixed: Extract replacement operation to a separate variable to resolve SyntaxError
            preview_text = row['text'][:240].replace('\n', ' / ')
            print(f"[sample-{split_name}-{i}] len={len(row['input_ids'])} text_preview={preview_text}")

    # package into DatasetDict and save
    dsdict = DatasetDict(cleaned_splits)
    output_dir = args.output
    if os.path.exists(output_dir):
        # avoid overwriting accidentally
        backup = output_dir + ".bak"
        print(f"[!] Output path {output_dir} exists. Will save to {output_dir} (overwrite disabled).")
    print(f"[i] Saving cleaned dataset -> {output_dir}")
    dsdict.save_to_disk(output_dir)
    print("[✓] Saved cleaned dataset.")

    # Final verification: token bounds and structure sampled
    print("\n--- Post-save quick verification (samples) ---")
    example_issues = 0
    for split_name, ds_out in dsdict.items():
        print(f"[verify] split {split_name} len={len(ds_out)}")
        for i in range(min(20, len(ds_out))):
            ids = ds_out[i]["input_ids"]
            if max(ids) >= len(tok):
                print(f"[ERROR] token index out of bounds in split {split_name} sample {i}: max={max(ids)} >= vocab={len(tok)}")
                example_issues += 1
    if example_issues == 0:
        print("[✓] Quick verification passed. Proceed to training with this new cleaned dataset.")
    else:
        print("[✗] Found issues. Please inspect the reported samples above.")

if __name__ == "__main__":
    main()
