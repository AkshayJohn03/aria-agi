#!/usr/bin/env python3
"""
prepare_dataset_v3_long_resume.py

Resumable Wikipedia + instruction-chunk builder.

Features:
- Resume exactly from last saved page using resume_state.json
- Periodically saves resume state so powercuts are recoverable
- Keeps both `wiki` and `merge` modes (instruction merging preserved)
- Writes Arrow chunks (HF Dataset.save_to_disk) like the prior script
- Minimal console output (progress only). No extra logging files.

Usage examples (wiki):
  python my_datasets/code/prepare_dataset_v3_long_resume.py `
    --mode wiki `
    --input_raw my_datasets/raw/enwiki-latest-pages-articles/enwiki-latest-pages-articles.xml `
    --out_dir artifacts/processed/wiki_chunks 
    --tokenizer artifacts/zia_tokenizer_60k --max_len 1024 --items_per_chunk 200000

Usage examples (merge):
  python my_datasets/code/prepare_dataset_v3_long_resume.py \
    --mode merge \
    --sources datasets/raw/hf_cache/yahma___alpaca-cleaned \
            datasets/raw/hf_cache/OpenAssistant___oasst1 \
    --out_dir artifacts/processed/ift_chunks \
    --tokenizer artifacts/zia_tokenizer_60k --max_len 8192 --items_per_chunk 100000
"""
import os
import sys
import argparse
import json
import bz2
import xml.etree.ElementTree as ET
import re
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer

RE_SPACES = re.compile(r"[^\S\r\n]+")
RE_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f]")

RESUME_STATE_FILENAME = "resume_state.json"
RESUME_SAVE_EVERY_PAGES = 500  # update resume file every N pages

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = RE_CTRL.sub(" ", s)
    s = RE_SPACES.sub(" ", s)
    return s.strip()

def load_resume_state(out_dir):
    path = os.path.join(out_dir, RESUME_STATE_FILENAME)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}

def save_resume_state(out_dir, state):
    path = os.path.join(out_dir, RESUME_STATE_FILENAME)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[!] Failed to write resume state: {e}", file=sys.stderr)

# ---------------------------
# Wiki stream with resume
# ---------------------------
def stream_wiki_xml_resume(input_path, tokenizer, out_dir, max_len=1024, items_per_chunk=200000):
    """
    Stream parse Wikipedia XML and save in chunks.
    Resume behavior:
      - checks resume_state.json in out_dir for keys:
          last_page_index (int)  -> index (0-based) of last processed <page>
          last_chunk (int)       -> last chunk index written
          saved_samples (int)    -> total samples saved so far
      - when resuming, it will skip pages until it reaches last_page_index+1 and continue.
    Note: For .bz2 inputs we must iterate from start (can't seek easily); resume will still skip pages,
    but we periodically save resume_state so interruption cost is limited.
    """
    os.makedirs(out_dir, exist_ok=True)
    resume = load_resume_state(out_dir)
    last_page_index = int(resume.get("last_page_index", -1))
    last_chunk = int(resume.get("last_chunk", -1))
    saved_samples = int(resume.get("saved_samples", 0))

    # find next chunk index by scanning out_dir for existing chunks if resume file absent
    if last_chunk < 0:
        existing = sorted([f for f in os.listdir(out_dir) if f.startswith("wiki_chunk_") and f.endswith(".arrow")])
        if existing:
            # take max index present
            try:
                last_chunk = int(existing[-1].split("_")[-1].split(".")[0])
            except Exception:
                last_chunk = -1
        else:
            last_chunk = -1

    # open xml (support .bz2)
    if input_path.endswith(".bz2"):
        fh = bz2.open(input_path, "rb")
    else:
        fh = open(input_path, "rb")

    # iterparse
    context = ET.iterparse(fh, events=("end",))
    buffer = []
    chunk_idx = last_chunk + 1
    page_counter = -1  # will be incremented when seeing a <page>
    saved = saved_samples

    # quick progress print
    print(f"[ℹ] Starting wiki stream. Resuming from page {last_page_index+1} (last_chunk={last_chunk}, saved={saved}).")

    pbar = tqdm(unit="pages")
    for event, elem in context:
        if elem.tag.endswith("page"):
            page_counter += 1
            # skip pages until we reach last_page_index+1
            if page_counter <= last_page_index:
                # still need to update the progress bar for visual feedback
                pbar.update(1)
                elem.clear()
                continue

            title_elem = elem.find("./{*}title")
            text_elem = elem.find("./{*}revision/{*}text")
            title = title_elem.text if title_elem is not None else ""
            text = text_elem.text if text_elem is not None else ""
            text = normalize_text(text)
            if len(text) >= 200:
                prompt = f"Write an encyclopedic article about {title.strip()}:"
                completion = text
                # check token length; if too long, truncate completion portion
                try:
                    toks = tokenizer.encode(prompt + " " + completion, truncation=False)
                except Exception:
                    # fallback: naive truncation by characters if tokenizer fails
                    completion = completion[: max(1000, max_len * 3)]
                    toks = tokenizer.encode(prompt + " " + completion, truncation=False)
                if len(toks) > max_len:
                    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                    prompt_len = len(prompt_ids)
                    max_comp = max(4, max_len - prompt_len)
                    comp_ids = tokenizer.encode(completion, add_special_tokens=False)[:max_comp]
                    toks = prompt_ids + comp_ids
                    # decode completion back safely
                    completion = tokenizer.decode(toks[prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                buffer.append({"instruction": prompt, "response": completion})
                saved += 1

            pbar.update(1)
            elem.clear()

            # periodically save chunk
            if len(buffer) >= items_per_chunk:
                chunk_file = os.path.join(out_dir, f"wiki_chunk_{chunk_idx:04d}.arrow")
                Dataset.from_list(buffer).save_to_disk(chunk_file)
                # update resume state: last_page_index is current page_counter
                last_chunk = chunk_idx
                state = {"last_page_index": page_counter, "last_chunk": last_chunk, "saved_samples": saved}
                save_resume_state(out_dir, state)
                chunk_idx += 1
                buffer.clear()

            # periodically persist resume state even if buffer not full (helps with interruptions)
            if (page_counter - last_page_index) % RESUME_SAVE_EVERY_PAGES == 0:
                state = {"last_page_index": page_counter, "last_chunk": last_chunk, "saved_samples": saved}
                save_resume_state(out_dir, state)

    # final flush
    if buffer:
        chunk_file = os.path.join(out_dir, f"wiki_chunk_{chunk_idx:04d}.arrow")
        Dataset.from_list(buffer).save_to_disk(chunk_file)
        last_chunk = chunk_idx
        state = {"last_page_index": page_counter, "last_chunk": last_chunk, "saved_samples": saved}
        save_resume_state(out_dir, state)
        buffer.clear()

    pbar.close()
    print(f"[✓] Done. Total saved wiki samples (this run cumulative): {saved} -> saved to {out_dir}")

# ---------------------------
# Merge instruction/chat datasets (unchanged)
# ---------------------------
def merge_instruction_datasets(source_dirs, out_dir, tokenizer, max_len=8192, items_per_chunk=100000):
    os.makedirs(out_dir, exist_ok=True)
    buffer = []
    chunk_idx = 0
    total = 0

    def try_fields(obj, keys):
        for k in keys:
            if k in obj and isinstance(obj[k], str):
                return obj[k]
        return ""

    print(f"[ℹ] Merging sources: {len(source_dirs)} -> {out_dir}")
    for src in source_dirs:
        # walk src to find .json/.jsonl files (handles HF cache directories)
        jsonl_files = []
        for root, _, files in os.walk(src):
            for f in files:
                if f.endswith(".json") or f.endswith(".jsonl"):
                    jsonl_files.append(os.path.join(root, f))
        for path in jsonl_files:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            j = json.loads(line)
                        except Exception:
                            continue
                        instr = normalize_text(try_fields(j, ["instruction","prompt","question","query","input"]))
                        resp  = normalize_text(try_fields(j, ["output","response","answer","completion"]))
                        if not resp or len(resp.split()) < 5:
                            continue
                        full = f"Instruction: {instr}\nResponse: {resp}"
                        toks = tokenizer.encode(full, add_special_tokens=False)
                        if len(toks) > max_len:
                            toks = toks[:max_len]
                            full = tokenizer.decode(toks, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        buffer.append({"text": full})
                        total += 1
                        if len(buffer) >= items_per_chunk:
                            Dataset.from_list(buffer).save_to_disk(os.path.join(out_dir, f"instruct_chunk_{chunk_idx:04d}.arrow"))
                            buffer.clear()
                            chunk_idx += 1
            except Exception as e:
                print(f"[!] Skipping file (error): {path} -> {e}", file=sys.stderr)
    if buffer:
        Dataset.from_list(buffer).save_to_disk(os.path.join(out_dir, f"instruct_chunk_{chunk_idx:04d}.arrow"))
    print(f"[✓] Merged {total} instruction/chat samples into {out_dir}")

# ---------------------------
# Entry
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["wiki","merge"], default="wiki")
    ap.add_argument("--input_raw", type=str, help="Wikipedia XML path (.xml or .bz2)")
    ap.add_argument("--sources", nargs="*", help="Instruction dataset directories (for merge mode)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--items_per_chunk", type=int, default=200000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})

    if args.mode == "wiki":
        assert args.input_raw, "Provide --input_raw for wiki mode"
        # Create out_dir and print resume info
        os.makedirs(args.out_dir, exist_ok=True)
        resume = load_resume_state(args.out_dir)
        if resume:
            # auto-resume silently per your choice
            print(f"[ℹ] Found resume state: last_page_index={resume.get('last_page_index')} last_chunk={resume.get('last_chunk')} saved_samples={resume.get('saved_samples')}")
        else:
            print("[ℹ] No resume state found — starting fresh.")
        stream_wiki_xml_resume(args.input_raw, tok, args.out_dir, max_len=args.max_len, items_per_chunk=args.items_per_chunk)
    else:
        assert args.sources, "Provide --sources for merge mode"
        merge_instruction_datasets(args.sources, args.out_dir, tok, max_len=args.max_len, items_per_chunk=args.items_per_chunk)

if __name__ == "__main__":
    main()
