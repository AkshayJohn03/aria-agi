#!/usr/bin/env python3
"""
audit_zia_checkpoints_detailed.py

🧠 Stage 1: Deep Structural Audit for ZIA checkpoints.
Scans all checkpoints under `artifacts/`, extracts detailed model structure
and tensor statistics (without running inference).

Outputs:
 - logs/model_audit_detailed.csv
 - logs/model_audit_detailed.json
 - logs/model_audit_summary.txt
"""

import os, sys, json, csv, math, hashlib, traceback
from datetime import datetime
import torch
import numpy as np

# The script is located in '.../aria_ai_assistant/my_datasets/code'.
# To reach '.../aria_ai_assistant/artifacts', we go up two levels to 'aria_ai_assistant'.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")

LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

CSV_LOG = os.path.join(LOG_DIR, "model_audit_detailed.csv")
JSON_LOG = os.path.join(LOG_DIR, "model_audit_detailed.json")
TXT_LOG = os.path.join(LOG_DIR, "model_audit_summary.txt")

def log(msg):
    print(msg)
    with open(TXT_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def md5sum(filename, blocksize=65536):
    h = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            h.update(chunk)
    return h.hexdigest()

def analyze_state_dict(state):
    stats = {
        "total_params": 0,
        "tensor_types": {},
        "emb_shape": None,
        "lm_head_shape": None,
        "rope_present": False,
        "alibi_present": False,
        "key_sample": [],
        "d_model_guess": None,
        "n_layers_guess": None,
        "n_heads_guess": None,
        "mlp_ratio_guess": None,
    }

    keys = list(state.keys())
    stats["key_sample"] = keys[:12]
    layer_prefixes = set()

    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        stats["total_params"] += v.numel()

        dt = str(v.dtype)
        stats["tensor_types"][dt] = stats["tensor_types"].get(dt, 0) + 1

        kl = k.lower()
        if "tok" in kl and "weight" in kl and stats["emb_shape"] is None:
            stats["emb_shape"] = tuple(v.shape)
        if ("lm_head" in kl or "head.weight" in kl) and stats["lm_head_shape"] is None:
            stats["lm_head_shape"] = tuple(v.shape)
        if "rotary" in kl or "freqs" in kl:
            stats["rope_present"] = True
        if "alibi" in kl:
            stats["alibi_present"] = True

        # Track block/layer prefixes
        if "blocks." in k or "transformer.h." in k: # Added transformer.h. for GPT-like structure
            # Handle both 'blocks.0.ln1' (V2 dense) and 'transformer.h.0.ln_1' (V3 RoPE)
            parts = k.split(".")
            if len(parts) > 2 and parts[1].isdigit():
                 layer_prefixes.add(parts[1])

        # Guess hidden dimension, head count
        if "attn.in_proj_weight" in k:
            d_model = v.shape[1]
            stats["d_model_guess"] = d_model
            stats["n_heads_guess"] = 6 if d_model == 384 else (8 if d_model == 512 else None)
        # Assuming V3 uses a similar structure for the MLP first layer
        if "mlp.0.weight" in k or "ff.net.0.weight" in k:
            hidden_dim = v.shape[0]
            d_model = v.shape[1]
            if d_model > 0:
                stats["mlp_ratio_guess"] = round(hidden_dim / d_model, 2)
        
        # RoPE specific shape analysis for sequence length guess
        if "freqs_cis" in k:
            # Shape is usually (max_len/2, head_dim) or (max_len, head_dim).
            # If shape is [512, 32] as seen in the error, the max length is likely 512.
            # We don't save this to stats here, but it's noted.
            pass # FIX: Added 'pass' to avoid IndentationError

    stats["n_layers_guess"] = len(layer_prefixes)
    return stats

def safe_load_ckpt(path):
    try:
        # We try weights_only=True first for safety
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        log(f"[⚠️] Safe load failed for {os.path.basename(path)}: {e}")
        
        # Fallback to unsafe load, as seen in the previous log snippet
        log("[⚠️] Retrying load without weights_only (DANGEROUS).")
        try:
            ck = torch.load(path, map_location="cpu")
        except Exception as e2:
            log(f"[❌] Fallback load failed: {e2}")
            # If it failed to load even with weights_only=False, it's corrupted or unreadable
            return None

    if isinstance(ck, dict):
        if "model" in ck and isinstance(ck["model"], dict):
            return ck["model"]
        elif any(isinstance(v, torch.Tensor) for v in ck.values()):
            return ck
    return None

def main():
    if os.path.exists(CSV_LOG): os.remove(CSV_LOG)
    if os.path.exists(JSON_LOG): os.remove(JSON_LOG)
    if os.path.exists(TXT_LOG): os.remove(TXT_LOG)

    log(f"[🧠] Starting detailed model audit at {datetime.now()}")
    checkpoints = []
    
    # Check current ARTIFACTS_DIR path to confirm the fix
    log(f"[i] Checking for checkpoints in: {ARTIFACTS_DIR}")
    
    for root, _, files in os.walk(ARTIFACTS_DIR):
        for f in files:
            if f.endswith(".pt"):
                checkpoints.append(os.path.join(root, f))
    checkpoints = sorted(checkpoints)
    log(f"[📦] Found {len(checkpoints)} checkpoint files\n")

    all_data = []
    
    # Only proceed if files were found
    if not checkpoints:
         log("[i] No checkpoints found to analyze. Exiting.")
         return

    with open(CSV_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "checkpoint", "file_size_MB", "last_modified", "md5",
            "param_count_M", "n_layers", "d_model", "n_heads", "mlp_ratio",
            "emb_shape", "lm_head_shape",
            "rope", "alibi", "tensor_types", "key_sample"
        ])

        for ckpt in checkpoints:
            size_MB = os.path.getsize(ckpt) / (1024 * 1024)
            mtime = datetime.fromtimestamp(os.path.getmtime(ckpt)).strftime("%Y-%m-%d %H:%M")
            checksum = md5sum(ckpt)

            log(f"[🔍] Inspecting {os.path.basename(ckpt)} ({size_MB:.1f} MB)")
            state = safe_load_ckpt(ckpt)
            if state is None:
                log("   [❌] Could not load checkpoint.")
                continue

            stats = analyze_state_dict(state)
            param_M = stats["total_params"] / 1e6
            log(f"   ↳ Params: {param_M:.2f}M | Layers: {stats['n_layers_guess']} | d_model={stats['d_model_guess']}")
            log(f"   ↳ Rope={stats['rope_present']} Alibi={stats['alibi_present']} | Types={stats['tensor_types']}")
            log(f"   ↳ Emb: {stats['emb_shape']} | LM Head: {stats['lm_head_shape']}")
            log("")

            row = [
                ckpt, f"{size_MB:.2f}", mtime, checksum,
                f"{param_M:.2f}", stats["n_layers_guess"], stats["d_model_guess"],
                stats["n_heads_guess"], stats["mlp_ratio_guess"],
                str(stats["emb_shape"]), str(stats["lm_head_shape"]),
                stats["rope_present"], stats["alibi_present"],
                json.dumps(stats["tensor_types"]), str(stats["key_sample"])
            ]
            writer.writerow(row)
            all_data.append({
                "path": ckpt,
                "file_size_MB": size_MB,
                "last_modified": mtime,
                "md5": checksum,
                "params_M": param_M,
                "layers": stats["n_layers_guess"],
                "d_model": stats["d_model_guess"],
                "n_heads": stats["n_heads_guess"],
                "mlp_ratio": stats["mlp_ratio_guess"],
                "emb_shape": stats["emb_shape"],
                "lm_head_shape": stats["lm_head_shape"],
                "rope": stats["rope_present"],
                "alibi": stats["alibi_present"],
                "tensor_types": stats["tensor_types"],
                "key_sample": stats["key_sample"],
            })

    with open(JSON_LOG, "w", encoding="utf-8") as jf:
        json.dump(all_data, jf, indent=2)

    log(f"\n[✅] Detailed audit complete — saved to:")
    log(f" • {CSV_LOG}")
    log(f" • {JSON_LOG}")
    log(f" • {TXT_LOG}")

if __name__ == "__main__":
    main()
