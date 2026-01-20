# ==========================
# 0) ENV & PATH SETUP
# ==========================
# Unified Dataset Acquisition for Aria-X
# Goal: One blended dataset that teaches both general conversation & finance/trading knowledge naturally.

import os
import io
import sys
import json
import time
import shutil
import tempfile
import logging
from typing import Optional, Iterable, Dict, Any
from datasets import load_dataset, Dataset
from datasets.download.download_manager import DownloadConfig
from huggingface_hub import HfApi
from tqdm import tqdm

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# --- Paths ---
RAW_DIR_D = r"D:\aria\aria_ai\aria_ai_assistant\datasets\raw"
RAW_DIR_F = r"F:\aria\aria_ai\aria_ai_assistant\datasets\raw"
HF_CACHE = os.path.join(RAW_DIR_D, "hf_cache")
TEMP_DIR = r"F:\temp"

for p in [RAW_DIR_D, RAW_DIR_F, HF_CACHE, TEMP_DIR]:
    os.makedirs(p, exist_ok=True)

# --- Force Hugging Face to use our paths ---
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE, "datasets")
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE, "transformers")
os.environ["HF_HUB_TEMP_DIR"] = TEMP_DIR
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR
tempfile.tempdir = TEMP_DIR
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

log.info("✅ Environment variables set.")
log.info(f"✅ HF cache set to: {HF_CACHE}")
log.info(f"✅ Temp files set to: {TEMP_DIR}")

# ==========================
# 1) GLOBAL BUDGET
# ==========================
MAX_TOTAL_BYTES = 70 * (1024 ** 3)  # ~70 GB
written_bytes_global = 0

def human(nbytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}PB"

def remaining_budget_bytes() -> int:
    return max(0, MAX_TOTAL_BYTES - written_bytes_global)

def append_size(b: int):
    global written_bytes_global
    written_bytes_global += b

# ==========================
# 2) IO HELPERS
# ==========================
def jsonl_writer(path: str) -> io.TextIOWrapper:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "w", encoding="utf-8")

# ==========================
# 3) STREAM TO JSONL
# ==========================
def stream_to_jsonl(ds_stream: Iterable[Dict[str, Any]], out_path: str,
                    text_keys: Optional[list] = None,
                    max_samples: Optional[int] = None,
                    max_bytes: Optional[int] = None) -> int:
    written = 0
    taken = 0
    budget_left = remaining_budget_bytes()
    if max_bytes is None or max_bytes > budget_left:
        max_bytes = budget_left
    if max_bytes <= 0:
        log.info("⏹ Global budget exhausted. Skipping.")
        return 0

    log.info(f"💾 Starting stream to JSONL, max size: {human(max_bytes)}")
    
    with jsonl_writer(out_path) as fout:
        for ex in tqdm(ds_stream, desc="Streaming samples", unit="samples", mininterval=1.0):
            if max_samples is not None and taken >= max_samples:
                break
            text = ""
            if text_keys:
                for k in text_keys:
                    if k in ex and ex[k]:
                        if text: text += "\n"
                        text += str(ex[k])
            else:
                for k in ["text", "content", "body", "title", "selftext"]:
                    if k in ex and ex[k]:
                        text = str(ex[k])
                        break
            if not text.strip():
                continue
            
            payload = json.dumps({"text": text.strip()}, ensure_ascii=False) + "\n"
            b = len(payload.encode("utf-8"))

            if written + b > max_bytes or remaining_budget_bytes() - b < 0:
                log.warning("Budget limit reached, stopping.")
                break
                
            fout.write(payload)
            written += b
            append_size(b)
            taken += 1
            
    log.info(f"✅ Wrote {taken} samples, total size {human(written)} to {out_path}")
    return written

# ==========================
# 4) DATASET DEFINITIONS
# ==========================
# Corrected and updated dataset list.
DATASETS = [
    # --- General Chat ---
    ("openhermes", "rombodawg/OpenHermes-2.5-Uncensored", "train", "F", ["text"], 3*(1024**3), 300_000),
    ("fineweb", "HuggingFaceFW/fineweb-edu", "train", "F", ["text"], 3*(1024**3), 300_000),
    ("ultrachat", "HuggingFaceH4/ultrachat_200k", "train_gen", "D", ["text"], 512*(1024**2), 100_000),

    # --- Empathy & reasoning ---
    ("empathetic_dialogues", "empathetic_dialogues", "train", "D", ["utterance"], 256*(1024**2), 50_000),
    ("stackexchange", "HuggingFaceM4/stack-exchange-paired", "train", "D", ["text"], 512*(1024**2), 100_000),

    # --- Finance / Trading ---
    ("financial_phrasebank", "financial_phrasebank", "train", "D", ["sentence"], 64*(1024**2), 100_000),
    ("fiqa", "financial_sentiment", "train", "F", ["sentence"], 64*(1024**2), 50_000),
    ("hedge_letters", "pszemraj/hedge-fund-letters", "train", "F", ["text"], 256*(1024**2), 50_000),
    ("fin_news", "zeroshot/twitter-financial-news-sentiment", "train", "F", ["text"], 256*(1024**2), 50_000),
    ("stocknet", "fin-sentiment/stocknet-sentiment", "train", "F", ["text"], 256*(1024**2), 50_000),
]

# ==========================
# 5) INGEST LOCAL CORPUS
# ==========================
ALTER_EGO_IN_DIR = os.path.join(RAW_DIR_D, "alter_ego_incoming")
ALTER_EGO_OUT = os.path.join(RAW_DIR_D, "cleaned", "alter_ego.jsonl")

def ingest_local_corpora(in_dir: str, out_path: str) -> int:
    written = 0
    if not os.path.isdir(in_dir):
        log.info("🧠 No local corpora found.")
        return 0

    log.info("🧠 Ingesting local corpora...")
    with jsonl_writer(out_path) as fout:
        for fname in os.listdir(in_dir):
            fpath = os.path.join(in_dir, fname)
            if not os.path.isfile(fpath):
                continue

            try:
                log.debug(f" - Processing {fname}")
                if fname.endswith(".jsonl"):
                    with open(fpath, "r", encoding="utf-8") as fin:
                        for ln in fin:
                            obj = json.loads(ln)
                            text = obj.get("text") or obj.get("content") or json.dumps(obj)
                            payload = json.dumps({"text": text.strip()}, ensure_ascii=False) + "\n"
                            b = len(payload.encode("utf-8"))
                            if remaining_budget_bytes() - b < 0:
                                log.warning(f"Global budget exhausted, stopping local ingestion.")
                                return written
                            fout.write(payload)
                            written += b
                            append_size(b)
                else:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fin:
                        buf = fin.read()
                    payload = json.dumps({"text": buf.strip()}, ensure_ascii=False) + "\n"
                    b = len(payload.encode("utf-8"))
                    if remaining_budget_bytes() - b < 0:
                        log.warning(f"Global budget exhausted, stopping local ingestion.")
                        return written
                    fout.write(payload)
                    written += b
                    append_size(b)
            except Exception as e:
                log.error(f"❌ Failed to process {fname}: {e}")

    log.info(f"✅ Ingested {human(written)} from local files to {out_path}")
    return written

# ==========================
# 6) PROCESS ONE DATASET
# ==========================
def try_load_stream(repo: str, split: str) -> Optional[Dataset]:
    # --- Robust Fallback ---
    # 1. Try standard streaming load with trust_remote_code
    try:
        log.info(f"Attempting standard streaming load for '{repo}'...")
        # Use a timeout to prevent hanging on slow connections or LFS issues
        return load_dataset(repo, split=split, streaming=True, trust_remote_code=True, download_mode='force_redownload', download_timeout=60)
    except Exception as e:
        log.warning(f"Standard streaming failed: {e}. Falling back to non-streaming.")
    
    # 2. Fallback to non-streaming with local cache and trust_remote_code
    try:
        log.info("Attempting non-streaming download with local cache...")
        dc = DownloadConfig(cache_dir=HF_CACHE)
        # Add trust_remote_code here as well
        return load_dataset(repo, split=split, download_config=dc, trust_remote_code=True)
    except Exception as e:
        log.error(f"Non-streaming download failed: {e}. Checking Hub for repository.")
    
    # 3. Final fallback: check for existence and suggest alternatives
    try:
        api = HfApi()
        if not api.dataset_info(repo):
            log.critical(f"❌ Dataset '{repo}' does not exist on the Hub.")
    except Exception as e:
        log.critical(f"❌ Dataset '{repo}' could not be accessed. Reason: {e}")
        
    return None

def process_one(name, repo, split, side, text_keys, cap_bytes, cap_samples):
    root = RAW_DIR_F if side.upper() == "F" else RAW_DIR_D
    raw_out = os.path.join(root, f"{name}.jsonl")
    clean_out = os.path.join(root, "cleaned", f"{name}.jsonl")

    if os.path.isfile(raw_out):
        log.info(f"⚠️ Skipped {name} (already exists)")
        return
        
    if remaining_budget_bytes() <= 0:
        log.warning(f"⏹ Skipping {name} — global budget exhausted.")
        return

    log.info(f"⬇ Downloading {name}: {repo} [{split}] -> {raw_out}")
    ds = try_load_stream(repo, split)
    
    if ds is None:
        log.error(f"Skipping {name} due to load failure.")
        return

    written_raw = stream_to_jsonl(ds, raw_out, text_keys, cap_samples, cap_bytes)
    
    if written_raw == 0:
        log.warning(f"Did not write any data for {name}.")
        return

    os.makedirs(os.path.dirname(clean_out), exist_ok=True)
    shutil.copy2(raw_out, clean_out)
    log.info(f"🧹 Cleaned {name}: copied raw data to {clean_out}")

# ==========================
# 7) MAIN
# ==========================
def download_all():
    start = time.time()
    log.info(f"🔧 Starting download process. Total budget: {human(MAX_TOTAL_BYTES)}")
    for name, repo, split, side, text_keys, cap_b, cap_n in DATASETS:
        process_one(name, repo, split, side, text_keys, cap_b, cap_n)
        if remaining_budget_bytes() <= 0:
            log.warning("Global budget exhausted. Stopping data acquisition.")
            break
            
    ingest_local_corpora(ALTER_EGO_IN_DIR, ALTER_EGO_OUT)
    
    end = time.time()
    log.info(f"✅ All tasks completed. Total data ingested: {human(written_bytes_global)} in {end-start:.1f}s")

if __name__ == "__main__":
    download_all()