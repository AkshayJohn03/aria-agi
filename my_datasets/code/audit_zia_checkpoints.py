#!/usr/bin/env python3
"""
audit_zia_checkpoints_v2.py

Improved audit of ZIA checkpoints:
 - finds .pt files under artifacts/
 - inspects state_dict keys/shapes
 - robust param counting (sums tensor.numel())
 - attempts small-sample validation on available datasets, supporting:
    * datasets saved via save_to_disk (load_from_disk)
    * Arrow shard folders (load_dataset("arrow", data_files=...))
 - tries to instantiate local model classes (MockGPT / ZiaModel) from common files;
   if not available, will still log shapes and stats.
 - writes CSV and verbose log.
"""
import os
import sys
import csv
import json
import math
import glob
import traceback
from datetime import datetime

import torch
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset

# --- Adjust paths if you run from different cwd ---
ROOT = os.path.abspath(".")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

CSV_LOG = os.path.join(LOG_DIR, "model_audit_results.csv")
DETAIL_LOG = os.path.join(LOG_DIR, "model_audit_details.txt")

# Datasets to evaluate (update these paths if needed)
DATASETS_TO_TEST = {
    "zia_ift_v3_clean": os.path.join(ROOT, "datasets/processed/zia_ift_v3_clean"),
    "arrow_cleaned_v1": os.path.join(ROOT, "my_datasets/processed/arrow_cleaned_v1"),
    # add other dataset paths if needed
}

# Tokenizer path
TOKENIZER_PATH = os.path.join(ROOT, "artifacts", "zia_tokenizer_60k")


# ----------------------
# Logging helpers
# ----------------------
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(DETAIL_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ----------------------
# Safe checkpoint loading
# ----------------------
def safe_load_ckpt(path):
    """
    Try to load checkpoint weights-only to avoid unpickling arbitrary python objects.
    Falls back to default torch.load if weights_only not supported.
    """
    try:
        # torch.load(..., weights_only=True) exists in newer torch versions
        ck = torch.load(path, map_location="cpu", weights_only=True)
        # If the checkpoint is a dict wrapping 'model', take it
        if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
            return ck["model"], ck
        if isinstance(ck, dict) and any(isinstance(v, torch.Tensor) for v in ck.values()):
            return ck, ck
        # Otherwise return ck as-is
        return ck, ck
    except TypeError:
        # weights_only not supported in this torch => fallback
        try:
            ck = torch.load(path, map_location="cpu")
            if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
                return ck["model"], ck
            if isinstance(ck, dict):
                return ck, ck
            return ck, ck
        except Exception as e:
            log(f"[❌] Failed to torch.load checkpoint {path}: {e}")
            return None, None
    except Exception as e:
        log(f"[❌] Failed to torch.load checkpoint {path}: {e}")
        return None, None


# ----------------------
# Analyze state dict keys / shapes
# ----------------------
def analyze_state_dict(state):
    """
    Returns a dict with:
      - total_keys
      - param_count (sum of numel)
      - emb_shape candidate
      - lm_head_shape candidate
      - rope_present, alibi_present
      - key_sample (first 8 keys)
    """
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
        # ensure it's a tensor-like object
        try:
            n = v.numel()
        except Exception:
            n = 0
        stats["param_count"] += int(n)

        kl = k.lower()
        # look for embeddings
        if stats["emb_shape"] is None and any(x in kl for x in ("tok.", "wte", "emb", "token_embedding", "tok_embeddings", "tok.weight")):
            try:
                stats["emb_shape"] = tuple(v.shape)
            except Exception:
                pass
        # look for lm_head / final head
        if stats["lm_head_shape"] is None and any(x in kl for x in ("lm_head", "head.", "lm_head.weight", "head.weight")):
            try:
                stats["lm_head_shape"] = tuple(v.shape)
            except Exception:
                pass

        # rope / rotary
        if "freqs" in kl or "freqs_cis" in kl or "rotary" in kl:
            stats["rope_present"] = True
        # alibi
        if "alibi" in kl:
            stats["alibi_present"] = True

    return stats


# ----------------------
# Dataset loader helpers
# ----------------------
def load_validation_split(path):
    """
    Try to load a validation split from a path. Supports:
      - datasets saved via save_to_disk (load_from_disk) (with 'validation' or 'val' keys)
      - a directory containing Arrow shards (search for .arrow files)
      - a single Arrow file or dataset directory
    Returns a Dataset object (or raises).
    """
    # 1) If path is dataset saved via save_to_disk:
    if os.path.isdir(path):
        # try huggingface load_from_disk
        try:
            ds = load_from_disk(path)
            # prefer 'validation' then 'val' then 'test'
            if isinstance(ds, dict) or hasattr(ds, 'keys'):
                if "validation" in ds:
                    return ds["validation"]
                if "val" in ds:
                    return ds["val"]
                if "test" in ds:
                    return ds["test"]
                # try 'train' as fallback but warn
                if "train" in ds:
                    return ds["train"]
            else:
                # ds could be a single Dataset
                return ds
        except Exception as e:
            # Not a save_to_disk dataset or failed; continue
            log(f"[i] load_from_disk failed for {path}: {e}")

    # 2) If path contains multiple arrow shard directories, find .arrow files
    arrow_files = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".arrow"):
                arrow_files.append(os.path.join(root, f))
    if arrow_files:
        # use load_dataset("arrow", data_files=...)
        try:
            # load as a single dataset
            ds = load_dataset("arrow", data_files=arrow_files, split="train")
            return ds
        except Exception as e:
            log(f"[i] load_dataset('arrow') failed for {len(arrow_files)} files under {path}: {e}")

    # 3) If path points to a single arrow file
    if os.path.isfile(path) and path.endswith(".arrow"):
        try:
            ds = load_dataset("arrow", data_files=[path], split="train")
            return ds
        except Exception as e:
            log(f"[i] load_dataset('arrow') single file failed for {path}: {e}")

    # If all attempts fail, raise
    raise FileNotFoundError(f"No valid dataset split found at {path}")


# ----------------------
# Try to instantiate a model from local training scripts
# ----------------------
def try_instantiate_local_model():
    """
    Attempts to import local model classes in common locations.
    Returns (model_class, config_class, import_path) or (None,None,None)
    """
    candidates = [
        ("my_datasets.code.train_zia_ift_v3_rope_alibi_new", "MockGPT", "V3Config"),
        ("my_datasets.code.train_zia_ift_v3_rope_alibi", "ZiaModel", "V3Config"),
        ("my_datasets.code.train_zia_ift_v3_rope_alibi_new", "MockGPT", "V3Config"),
        ("my_datasets.code.train_zia_ift_v3_rope_alibi", "MockGPT", "V3Config"),
        # add any other candidate module paths you use
    ]
    for module, cls_name, cfg_name in candidates:
        try:
            mod = __import__(module, fromlist=[cls_name, cfg_name])
            ModelClass = getattr(mod, cls_name, None)
            ConfigClass = getattr(mod, cfg_name, None)
            if ModelClass is not None:
                return ModelClass, ConfigClass, module
        except Exception:
            continue
    return None, None, None


# ----------------------
# Small-sample validation routine
# ----------------------
def eval_on_dataset_with_model(model, ds, device="cuda", max_samples=128, max_len=512):
    import torch.nn.functional as F
    model.to(device)
    model.eval()
    total_loss = 0.0
    count = 0
    # ds expected to have fields 'input_ids', 'labels' or similar
    # we'll try to be flexible and handle both list and dict formats
    with torch.no_grad():
        for i in range(min(len(ds), max_samples)):
            ex = ds[i]
            # try common keys
            if "input_ids" in ex:
                input_ids = ex["input_ids"]
            elif "input" in ex:
                # if raw text, we cannot evaluate
                return None, "dataset contains raw text"
            else:
                return None, "no input_ids in dataset example"

            if "labels" in ex:
                labels = ex["labels"]
            elif "label" in ex:
                labels = ex["label"]
            else:
                # can't evaluate if no labels
                return None, "no labels present"

            # convert to tensors and trim/pad
            x = torch.tensor(input_ids[:max_len], dtype=torch.long).unsqueeze(0).to(device)
            y = torch.tensor(labels[:max_len], dtype=torch.long).unsqueeze(0).to(device)

            # Forward: adapt to different model forward signatures
            try:
                out = model(x, labels=y)
                # out may be (logits, loss) in your mock code, or a transformers-like object
                if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                    loss = out[1]
                elif hasattr(out, "loss") and out.loss is not None:
                    loss = out.loss
                else:
                    # compute loss manually from logits if provided
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            except Exception as e:
                return None, f"model forward failed: {e}"

            if not torch.isfinite(loss):
                # skip non-finite
                continue
            total_loss += float(loss.item())
            count += 1

    if count == 0:
        return None, "no valid loss samples"
    avg_loss = total_loss / count
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return avg_loss, ppl


# ----------------------
# Main audit loop
# ----------------------
def main():
    # prepare logs
    if os.path.exists(CSV_LOG):
        os.remove(CSV_LOG)
    if os.path.exists(DETAIL_LOG):
        os.remove(DETAIL_LOG)

    log("Starting ZIA model audit")
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

    # attempt to get local model class
    ModelClass, ConfigClass, src_module = try_instantiate_local_model()
    if ModelClass:
        log(f"[i] Found local ModelClass {ModelClass.__name__} in module {src_module}")
    else:
        log("[i] No local model class found. Eval will be skipped unless you add a class.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")

    for ckpt in ckpt_paths:
        log(f"\nInspecting checkpoint: {ckpt}")
        state, full_ck = safe_load_ckpt(ckpt)
        if state is None:
            log(f"[❌] Skipping {ckpt} (failed to load)")
            continue

        stats = analyze_state_dict(state)
        param_count_m = stats["param_count"] / 1e6 if stats["param_count"] else None
        log(f"  ↳ {stats['total_keys']} keys | params ~ {param_count_m:.2f}M")
        log(f"  ↳ emb_shape={stats['emb_shape']} lm_head_shape={stats['lm_head_shape']} rope={stats['rope_present']} alibi={stats['alibi_present']}")
        log(f"  ↳ key sample: {stats['key_sample']}")

        # Try instantiate and load to model (best-effort)
        instantiated_model = None
        instantiate_note = ""
        if ModelClass and ConfigClass:
            try:
                # Create minimal config if ConfigClass is a dataclass or simple class
                try:
                    cfg = ConfigClass()
                except Exception:
                    # fallback: create empty simple namespace
                    class C: pass
                    cfg = C()
                    setattr(cfg, "tokenizer_path", TOKENIZER_PATH)
                    setattr(cfg, "vocab_size", vocab_size or 60004)
                # Set cfg attributes if present to avoid mismatch
                if hasattr(cfg, "vocab_size") and stats["emb_shape"] and cfg.vocab_size != stats["emb_shape"][0]:
                    try:
                        cfg.vocab_size = stats["emb_shape"][0]
                    except Exception:
                        pass

                m = ModelClass(cfg) if ConfigClass is not None else ModelClass()
                # load state dict (non-strict to handle naming differences)
                m.load_state_dict(state, strict=False)
                instantiated_model = m
                instantiate_note = "model_instantiated"
                log("  [✓] Instantiated local model and loaded state (strict=False).")
            except Exception as e:
                instantiate_note = f"instantiate_failed: {e}"
                log(f"  [⚠️] Could not instantiate/load model: {e}")
                log(traceback.format_exc())

        # Evaluate on datasets
        for ds_name, ds_path in DATASETS_TO_TEST.items():
            val_loss = None
            ppl = None
            notes = ""
            try:
                ds = load_validation_split(ds_path)
                log(f"  [i] Loaded dataset {ds_name} ({len(ds)} rows available)")
            except Exception as e:
                log(f"  [❌] Could not load dataset {ds_name} at {ds_path}: {e}")
                with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([ckpt, f"{param_count_m:.2f}M" if param_count_m else "", stats['emb_shape'], stats['lm_head_shape'],
                                     stats['rope_present'], stats['alibi_present'], ds_name, "", "", f"dataset_load_failed: {e}"])
                continue

            if instantiated_model is None:
                notes = "no_local_model_class"
                log(f"  [i] Skipping eval for {ds_name}: no runnable local model.")
                with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([ckpt, f"{param_count_m:.2f}M" if param_count_m else "", stats['emb_shape'], stats['lm_head_shape'],
                                     stats['rope_present'], stats['alibi_present'], ds_name, "", "", notes])
                continue

            # run eval on small sample (fast)
            try:
                avg_loss, ppl_val = eval_on_dataset_with_model(instantiated_model, ds, device=device, max_samples=128, max_len=512)
                if avg_loss is None:
                    notes = ppl_val  # error message
                    log(f"  [⚠️] Eval skipped/failed for {ds_name}: {notes}")
                else:
                    val_loss = avg_loss
                    ppl = ppl_val
                    log(f"  [🧮] {ds_name}: val_loss={val_loss:.6f}, ppl={ppl if math.isfinite(ppl) else 'inf'}")
            except Exception as e:
                notes = f"eval_exception: {e}"
                log(f"  [❌] Exception during eval on {ds_name}: {e}")
                log(traceback.format_exc())

            with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([ckpt, f"{param_count_m:.2f}M" if param_count_m else "", stats['emb_shape'], stats['lm_head_shape'],
                                 stats['rope_present'], stats['alibi_present'], ds_name,
                                 f"{val_loss:.6f}" if val_loss is not None else "", f"{ppl:.4f}" if ppl is not None else "", notes or instantiate_note])

    log("Audit complete. CSV and details written.")


if __name__ == "__main__":
    main()
