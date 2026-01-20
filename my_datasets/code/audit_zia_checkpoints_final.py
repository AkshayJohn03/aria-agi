#!/usr/bin/env python3
"""
audit_zia_checkpoints_final.py

Unified audit script combining robustness (v2) with speed/caching (v1).

Features:
 - Finds .pt files under artifacts/
 - Logs detailed checkpoint stats (params, shapes, key sample).
 - ✅ Instantiates local model classes if found (MockGPT/ZiaModel).
 - 🧩 Registers V3Config for safe loading of pickled checkpoints.
 - 💾 Implements aggressive, persistent disk caching for datasets.
 - ⏩ Runs quick validation (32 samples) on available datasets.
 - 🧱 Robustly loads sharded, single-file, or HF-dumped datasets.
 - 🗃️ Writes complete results to a detailed log and a summary CSV.
"""
import os
import sys
import csv
import random
import json
import math
import glob
import traceback
from datetime import datetime

import torch
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset, Dataset

# --- FIX: Ensure the directory containing V3Config is on the path ---
# This resolves the ModuleNotFoundError by explicitly adding the current script's directory (my_datasets/code) 
# to the Python search path (sys.path) before the import is attempted.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Now import V3Config using the direct module name (file name) since its directory is in sys.path
# Assuming train_zia_ift_v3_rope_alibi_new.py is in the same directory or accessible via the paths below.
try:
    from train_zia_ift_v3_rope_alibi_new import V3Config
except ImportError as e:
    # Handle case where the dependency file might not be found immediately
    print(f"[!] Warning: Could not import V3Config initially. Will try again in try_instantiate_local_model. Error: {e}")
    V3Config = None


# --- Path Setup ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# Assuming the script runs from my_datasets/code/ or similar, adjust ROOT to point to the project root.
if not os.path.isdir(os.path.join(ROOT, "artifacts")):
    ROOT = os.path.abspath(".") # Fallback to current working directory if initial guess is wrong

sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "my_datasets", "code"))

ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
LOG_DIR = os.path.join(ROOT, "logs")
DATASET_CACHE_DIR = os.path.join(ROOT, "datasets", "audit_cache") # New dedicated cache directory

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

TOKENIZER_PATH = os.path.join(ARTIFACTS_DIR, "zia_tokenizer_60k")
CSV_LOG = os.path.join(LOG_DIR, "model_audit_results.csv")
DETAIL_LOG = os.path.join(LOG_DIR, "model_audit_details.txt")

# Datasets to evaluate (update these paths if needed)
DATASETS_TO_TEST = {
    "zia_ift_v3_clean": os.path.join(ROOT, "datasets/processed/zia_ift_v3_clean"),
    "arrow_cleaned_v1": os.path.join(ROOT, "my_datasets/processed/arrow_cleaned_v1"),
    # Add other dataset paths here
}

# --- Logging helpers ---
def log(msg):
    """Logs a message to both stdout and the detailed log file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(DETAIL_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --- Model Importer & Safe Globals ---

def try_instantiate_local_model():
    """Tries to find local ZIA model/config classes from common training files."""
    candidates = [
        ("train_zia_ift_v3_rope_alibi_new", "MockGPT", "V3Config"),
        ("train_zia_ift_v3_rope_alibi", "ZiaModel", "V3Config"),
        ("train_zia_ift_v3_rope_alibi_new", "ZiaModel", "V3Config"),
    ]
    for module_name, model_name, cfg_name in candidates:
        try:
            # Note: module_name is now the simple file name due to the sys.path fix above
            module = __import__(module_name, fromlist=[model_name, cfg_name])
            ModelClass = getattr(module, model_name, None)
            ConfigClass = getattr(module, cfg_name, None)
            if ModelClass and ConfigClass:
                log(f"[✅] Found local model class {model_name} / config {cfg_name} in {module_name}")
                return ModelClass, ConfigClass, module_name
        except Exception:
            continue
    log("[i] No local model class found. Evaluation will be skipped.")
    return None, None, None

ModelClass, ConfigClass, _ = try_instantiate_local_model()

# 🧩 Registering V3Config for safe checkpoint loading (addresses the user's main error)
ConfigClass_for_globals = ConfigClass
if ConfigClass_for_globals is None:
    # Try to find V3Config explicitly from a known path just for registration
    try:
        # Changed import path to simple module name
        from train_zia_ift_v3_rope_alibi_new import V3Config
        ConfigClass_for_globals = V3Config
    except ImportError:
        pass

if ConfigClass_for_globals is not None:
    try:
        # Check if it has a __name__ attribute which indicates a class
        if hasattr(ConfigClass_for_globals, '__name__'):
            # This is the crucial step to prevent 'GLOBAL __main__.V3Config' errors
            torch.serialization.add_safe_globals([ConfigClass_for_globals])
            log(f"[🧩] Registered {ConfigClass_for_globals.__name__} for safe checkpoint loading.")
    except Exception as e:
        log(f"[i] Could not register safe global {ConfigClass_for_globals.__name__}: {e}")


# --- Checkpoint Loading & Analysis Helpers ---

def safe_load_ckpt(path):
    """Loads checkpoint, handling weights_only=True failures and wrapper dicts."""
    ck = None
    try:
        # Try weights_only=True first (safer)
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Fallback for older PyTorch versions that don't support weights_only
        log(f"[i] torch.load failed with weights_only. Falling back to default load for {os.path.basename(path)}")
        try:
            ck = torch.load(path, map_location="cpu")
        except Exception as e:
            log(f"[❌] Failed to torch.load (fallback) checkpoint {path}: {e}")
            return None

    except Exception as e:
        # Handle the V3Config pickling error (should be fixed by safe globals, but here for robustness)
        log(f"[❌] Failed to torch.load checkpoint {path}: {e}")
        
        # Try a final time without weights_only if safe globals failed (dangerous, but gets the job done)
        if "WeightsUnpickler error" in str(e):
            try:
                log(f"[⚠️] Retrying load without weights_only (DANGEROUS).")
                ck = torch.load(path, map_location="cpu")
            except Exception as final_e:
                log(f"[❌] Final load attempt failed: {final_e}")
                return None
        else:
            return None

    if ck is None:
        return None

    # Extract state_dict from common wrappers
    if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
        return ck["model"]
    # If the dictionary itself looks like a state dict (contains tensors)
    if isinstance(ck, dict) and any(isinstance(v, torch.Tensor) for v in ck.values()):
        return ck
    # If it's a raw object/tensor, it's not the state dict we need
    return None


def analyze_state_dict(state):
    """Analyzes a state dictionary for parameter counts, shapes, and features."""
    stats = {
        "total_keys": 0,
        "param_count": 0,
        "emb_shape": None,
        "lm_head_shape": None,
        "rope_present": False,
        "alibi_present": False,
        "key_sample": [],
    }
    if not isinstance(state, dict):
        return stats

    keys = list(state.keys())
    stats["total_keys"] = len(keys)
    stats["key_sample"] = keys[:12]

    for k in keys:
        v = state[k]
        stats["param_count"] += int(v.numel()) if hasattr(v, "numel") else 0

        kl = k.lower()
        if stats["emb_shape"] is None and any(x in kl for x in ("tok.weight", "wte.weight", "tok_embeddings")):
            stats["emb_shape"] = tuple(v.shape)
        if stats["lm_head_shape"] is None and any(x in kl for x in ("lm_head.weight", "head.weight")):
            stats["lm_head_shape"] = tuple(v.shape)
        if "freqs" in kl or "freqs_cis" in kl or "rotary" in kl:
            stats["rope_present"] = True
        if "alibi" in kl:
            stats["alibi_present"] = True

    return stats

# --- Robust Dataset Loader (Internal Helper) ---

def _load_uncached_dataset_split(path):
    """Internal helper to load a dataset split robustly from disk (no cache logic)."""
    # 1) If path is dataset saved via save_to_disk:
    if os.path.isdir(path):
        try:
            dsd = load_from_disk(path)
            # Prefer 'validation' > 'val' > 'test' > 'train'
            for split in ["validation", "val", "test", "train"]:
                if split in dsd:
                    log(f"[i] Found split '{split}' in DatasetDict.")
                    return dsd[split]
            if isinstance(dsd, Dataset):
                return dsd # It was a single Dataset object saved to disk
        except Exception as e:
            log(f"[i] load_from_disk failed: {e}")

    # 2) Find arrow files within the path (for sharded datasets)
    arrow_files = []
    # Check current directory and one level deep (e.g. `path/shard_000.arrow` or `path/train/shard_000.arrow`)
    search_paths = [path, os.path.join(path, "train"), os.path.join(path, "val"), os.path.join(path, "test")]
    for sp in search_paths:
        if os.path.isdir(sp):
            arrow_files.extend(glob.glob(os.path.join(sp, "*.arrow")))
    # Also check if path itself is an arrow file
    if os.path.isfile(path) and path.endswith(".arrow"):
        arrow_files.append(path)

    if arrow_files:
        log(f"[i] Found {len(arrow_files)} arrow files. Loading via load_dataset('arrow').")
        ds = load_dataset("arrow", data_files=arrow_files, split="train") # Loads all shards into 'train' split
        return ds

    raise FileNotFoundError(f"No valid dataset split, HF dump, or arrow files found at {path}")

# 💾 Dataset Loader with Caching (Fusion of v1 cache and v2 loader)
def load_dataset_cached_and_split(name, path):
    """Loads dataset from cache or disk, and saves to cache."""
    cache_path = os.path.join(DATASET_CACHE_DIR, f"{name}_audit_cache")

    try:
        if os.path.exists(cache_path):
            ds = load_from_disk(cache_path)
            log(f"[💾] Loaded dataset {name} from cache ({len(ds)} rows).")
            return ds
        else:
            log(f"[i] Cache miss for {name}. Loading from disk at {path}...")
            # Use the robust loader logic
            ds = _load_uncached_dataset_split(path)
            ds.save_to_disk(cache_path)
            log(f"[💾] Cached dataset {name} → {cache_path} ({len(ds)} rows).")
            return ds
    except Exception as e:
        log(f"[❌] Failed to load/cache dataset {name}: {e}")
        log(traceback.format_exc())
        return None

# --- Small-sample validation routine ---
def eval_on_dataset_with_model(model, ds, device="cuda", max_samples=32, max_len=512):
    """Evaluates the model on a small sample of the dataset."""
    import torch.nn.functional as F
    model.to(device)
    model.eval()
    total_loss = 0.0
    count = 0
    
    # Ensure we don't try to sample more than available
    if len(ds) == 0:
        return None, "dataset is empty"

    sample_indices = random.sample(range(len(ds)), min(len(ds), max_samples))

    with torch.no_grad():
        for i in sample_indices:
            ex = ds[i]

            if "input_ids" not in ex or "labels" not in ex:
                return None, "dataset missing 'input_ids' or 'labels'"

            # convert to tensors and trim
            # Ensure input_ids and labels are lists/arrays of integers
            try:
                x = torch.tensor(ex["input_ids"][:max_len], dtype=torch.long).unsqueeze(0).to(device)
                y = torch.tensor(ex["labels"][:max_len], dtype=torch.long).unsqueeze(0).to(device)
            except Exception as e:
                log(f"[⚠️] Skipping sample {i} due to tensor conversion error: {e}")
                continue


            if x.numel() == 0 or y.numel() == 0:
                continue

            try:
                # Forward pass - tries to accommodate common model types
                out = model(x, labels=y)
                
                # Try to extract loss from output object/tuple
                if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                    loss = out[1]
                elif hasattr(out, "loss") and out.loss is not None:
                    loss = out.loss
                else:
                    # Compute loss manually from logits
                    logits = out[0] if isinstance(out, tuple) else out
                    # Reshape for cross_entropy: (B*L, C) and (B*L)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

            except Exception as e:
                return None, f"model forward failed: {e}"

            if not torch.isfinite(loss):
                continue

            total_loss += float(loss.item())
            count += 1

    if count == 0:
        return None, "no valid loss samples computed"
    avg_loss = total_loss / count
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return avg_loss, ppl


# ----------------------
# Main audit loop
# ----------------------
def main():
    # prepare logs
    if os.path.exists(CSV_LOG):
        # Only remove if starting fresh, otherwise append? Keeping the original remove logic.
        try:
            os.remove(CSV_LOG)
        except OSError:
            log(f"[⚠️] Could not remove existing CSV log: {CSV_LOG}")

    if os.path.exists(DETAIL_LOG):
        # Only remove if starting fresh, otherwise append? Keeping the original remove logic.
        try:
            os.remove(DETAIL_LOG)
        except OSError:
             log(f"[⚠️] Could not remove existing detail log: {DETAIL_LOG}")

    log("Starting ZIA model audit (Unified Script)")

    # Tokenizer info
    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        vocab_size = len(tok)
        log(f"Tokenizer loaded from {TOKENIZER_PATH} (vocab {vocab_size})")
    except Exception as e:
        log(f"[!] Could not load tokenizer at {TOKENIZER_PATH}: {e}")
        vocab_size = None

    # find .pt checkpoint files
    ckpt_paths = []
    for root, _, files in os.walk(ARTIFACTS_DIR):
        for f in files:
            if f.endswith(".pt"):
                ckpt_paths.append(os.path.join(root, f))
    ckpt_paths = sorted(ckpt_paths)
    log(f"Found {len(ckpt_paths)} checkpoint files under {ARTIFACTS_DIR}")

    # CSV header
    with open(CSV_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint_path", "param_count_M", "emb_shape", "lm_head_shape",
                         "rope", "alibi", "dataset", "val_loss", "ppl", "notes"])

    # Model and Config from globals
    global ModelClass, ConfigClass
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")

    # Pre-load datasets to save time inside the model loop
    loaded_datasets = {}
    for ds_name, ds_path in DATASETS_TO_TEST.items():
        loaded_datasets[ds_name] = load_dataset_cached_and_split(ds_name, ds_path)

    for ckpt in ckpt_paths:
        log(f"\nInspecting checkpoint: {ckpt}")
        state = safe_load_ckpt(ckpt)
        
        if state is None:
            log(f"[❌] Skipping {ckpt} (failed to load state dict)")
            continue

        stats = analyze_state_dict(state)
        param_count_m = stats["param_count"] / 1e6 if stats["param_count"] else None
        log(f" ↳ {stats['total_keys']} keys | params ~ {param_count_m:.2f}M")
        log(f" ↳ emb_shape={stats['emb_shape']} lm_head_shape={stats['lm_head_shape']} rope={stats['rope_present']} alibi={stats['alibi_present']}")
        log(f" ↳ key sample: {stats['key_sample']}")

        # Try instantiate and load model
        instantiated_model = None
        instantiate_note = ""
        if ModelClass and ConfigClass:
            try:
                # Create minimal config
                try:
                    cfg = ConfigClass()
                except Exception:
                    # Fallback for ConfigClass that requires arguments or has issues
                    class C: pass
                    cfg = C()
                    setattr(cfg, "tokenizer_path", TOKENIZER_PATH)
                    setattr(cfg, "vocab_size", vocab_size or 60004) # Defaulting to a safe vocab size
                    
                # Ensure vocab size matches state dict if available
                if hasattr(cfg, "vocab_size") and stats["emb_shape"] and cfg.vocab_size != stats["emb_shape"][0]:
                    setattr(cfg, "vocab_size", stats["emb_shape"][0])

                m = ModelClass(cfg)
                # load state dict (non-strict to handle naming differences)
                m.load_state_dict(state, strict=False)
                instantiated_model = m
                instantiate_note = "model_instantiated"
                log("  [✓] Instantiated local model and loaded state (strict=False).")
            except Exception as e:
                instantiate_note = f"instantiate_failed: {type(e).__name__}: {str(e)[:100]}..."
                log(f"  [⚠️] Could not instantiate/load model: {e}")
                log(traceback.format_exc())

        # Evaluate on datasets
        for ds_name, ds in loaded_datasets.items():
            val_loss = None
            ppl = None
            notes = ""

            if ds is None:
                notes = "dataset_load_failed"
            elif instantiated_model is None:
                notes = "no_local_model_class"
                log(f" [i] Skipping eval for {ds_name}: no runnable local model.")
            else:
                # Run eval on small sample
                try:
                    # NOTE: We pass the instantiated_model to eval, ensuring a fresh call per checkpoint
                    avg_loss, ppl_val = eval_on_dataset_with_model(instantiated_model, ds, device=device, max_samples=32, max_len=512)
                    
                    if avg_loss is None:
                        notes = ppl_val # error message from eval
                        log(f" [⚠️] Eval skipped/failed for {ds_name}: {notes}")
                    else:
                        val_loss = avg_loss
                        ppl = ppl_val
                        log(f" [🧮] {ds_name}: val_loss={val_loss:.6f}, ppl={ppl if math.isfinite(ppl) else 'inf'}")
                        notes = instantiate_note
                except Exception as e:
                    notes = f"eval_exception: {type(e).__name__}: {str(e)[:100]}..."
                    log(f" [❌] Exception during eval on {ds_name}: {e}")
                    log(traceback.format_exc())

            # Write results to CSV
            with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([ckpt, 
                                 f"{param_count_m:.2f}M" if param_count_m is not None else "", 
                                 str(stats['emb_shape']), str(stats['lm_head_shape']),
                                 stats['rope_present'], stats['alibi_present'], 
                                 ds_name,
                                 f"{val_loss:.6f}" if val_loss is not None else "", 
                                 f"{ppl:.4f}" if ppl is not None and math.isfinite(ppl) else "", 
                                 notes or instantiate_note])

    log("\n[✅] Audit complete! Results saved to logs/model_audit_results.csv and logs/model_audit_details.txt")


if __name__ == "__main__":
    main()
