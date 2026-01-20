#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A unified, robust validator for:
- Tokenizer health / special tokens / round-trip encode/decode
- TinyGPT model + checkpoint loading (shapes, missing/unexpected keys)
- Forward pass shape checks and safe sampling-based generation
- Arrow datasets integrity (2 random rows per shard + extended logs)
"""

import os
import sys
import re
import json
import math
import glob
import random
import argparse
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# ---- Optional deps (we guard imports to keep script resilient) ----
try:
    import datasets as hfd
except Exception:
    hfd = None

try:
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc
except Exception:
    pa, pa_ipc = None, None

from transformers import AutoTokenizer

# ---- Import your TinyGPT -----------------------------------------------------
# This import path assumes this file is executed from your repo root.
# Adjust if needed (you can also pass --model-import "my_datasets.code.train_zia_dense")
DEFAULT_MODEL_IMPORT = "my_datasets.code.train_zia_dense"

def dynamic_import(module_path: str, symbol: str):
    import importlib
    m = importlib.import_module(module_path)
    return getattr(m, symbol)

# ------------------------------ Utils -----------------------------------------

def log_header(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)

def warn(msg: str):
    print(f"[WARN] {msg}")

def info(msg: str):
    print(f"[i] {msg}")

def ok(msg: str):
    print(f"[OK] {msg}")

def err(msg: str):
    print(f"[ERR] {msg}")

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def has_control_chars(s: str) -> bool:
    return bool(CONTROL_CHARS_RE.search(s))

def try_torch_load(path: str, map_location: str):
    # Be resilient to torch versions with/without weights_only kwarg
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # torch>=2.3
    except TypeError:
        warn("Your torch version does not support weights_only=True; falling back to default torch.load.")
        return torch.load(path, map_location=map_location)

def find_token_embedding_module(model: nn.Module) -> Optional[nn.Embedding]:
    # Pick the largest nn.Embedding as token embedding
    emb_cands = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Embedding)]
    if not emb_cands:
        return None
    name, emb = max(emb_cands, key=lambda t: t[1].num_embeddings)
    info(f"Detected token embedding module: {name} (num_embeddings={emb.num_embeddings}, dim={emb.embedding_dim})")
    return emb

def find_output_head_module(model: nn.Module) -> Optional[nn.Linear]:
    # Prefer common names; else, pick the Linear with max out_features
    preferred_names = ["head", "lm_head", "output", "classifier"]
    named_linears = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    for pref in preferred_names:
        for name, m in named_linears:
            if name.endswith(pref):
                info(f"Detected output head module by name: {name} (out_features={m.out_features})")
                return m
    if named_linears:
        # Fallback: the linear with the largest out_features
        name, m = max(named_linears, key=lambda t: t[1].out_features)
        warn(f"Falling back to largest Linear as head: {name} (out_features={m.out_features})")
        return m
    return None

def extract_logits(output: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Normalize a model's forward output to [B, V] logits (last token per batch).
    Handles:
    - Tensor [B, T, V]
    - Tensor [B, V]
    - Tensor [T, V]  (we'll treat T as time and pick last, then unsqueeze to [1, V])
    - Tuple where the first element is logits tensor of any of the above shapes
    """
    if isinstance(output, tuple):
        output = output[0]

    if not isinstance(output, torch.Tensor):
        raise RuntimeError(f"Model forward returned unsupported type: {type(output)}")

    if output.dim() == 3:
        # [B, T, V]
        logits = output[:, -1, :]
    elif output.dim() == 2:
        # Could be [B, V] or [T, V]; if first dim matches batch size, assume [B, V]
        if output.size(0) == input_ids.size(0):
            logits = output
        else:
            # Treat as [T, V]
            logits = output[-1:, :]
    else:
        raise RuntimeError(f"Unexpected logits shape: {tuple(output.shape)}")

    return logits  # [B, V]

@torch.no_grad()
def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    logits: [B, V]
    """
    B, V = logits.shape
    # Top-K
    if top_k and top_k > 0 and top_k < V:
        kth_vals = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)

    # Top-P
    if top_p and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Remove tokens with cumulative prob above the threshold
        mask = cumulative_probs > top_p
        # Ensure at least one token is kept
        mask[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        # Scatter back to original indexing
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    return logits

@torch.no_grad()
def safe_generate(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Minimal, robust autoregressive generator that avoids the shape pitfalls.
    Returns full sequence ids including the prompt.
    """
    model.eval()
    gen = prompt_ids.to(device)
    if gen.dim() == 1:
        gen = gen.unsqueeze(0)  # [1, T]
    B = gen.size(0)
    if B != 1:
        warn(f"Generation expects batch size 1; got B={B}. Using only the first row.")
        gen = gen[:1, :]
        B = 1

    for _ in range(max_new_tokens):
        out = model(gen)
        logits = extract_logits(out, gen)  # [1, V]
        if temperature is not None and temperature > 0:
            logits = logits / max(1e-6, float(temperature))
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(logits, dim=-1)  # [1, V]
        next_tok = torch.multinomial(probs, num_samples=1)  # [1, 1]
        gen = torch.cat([gen, next_tok], dim=1)

        if tokenizer.eos_token_id is not None and next_tok.item() == tokenizer.eos_token_id:
            break

    return gen  # [1, T+new]

# -------------------------- Dataset Inspection --------------------------------

def is_hf_dataset_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    names = set(os.listdir(path))
    # load_from_disk artifacts typically have these files
    return any(n in names for n in ["dataset_info.json", "state.json", "data-00000-of-00001.arrow"])

def list_arrow_files(path: str) -> List[str]:
    if os.path.isfile(path) and path.endswith(".arrow"):
        return [path]
    return sorted(glob.glob(os.path.join(path, "*.arrow")))

def decode_ids(sample_ids: List[int], tok: AutoTokenizer) -> str:
    try:
        return tok.decode(sample_ids, skip_special_tokens=False)
    except Exception as e:
        return f"<decode_error: {e}>"

def sample_indices(n: int, k: int) -> List[int]:
    k = max(0, min(k, n))
    if n <= k:
        return list(range(n))
    # deterministic but varied
    random_indices = sorted(random.sample(range(n), k))
    return random_indices

def inspect_hf_dataset_dir(path: str, tok: AutoTokenizer, per_file_samples: int = 2):
    if hfd is None:
        err("🤖 `datasets` is not installed; cannot inspect HF dataset directories.")
        return

    try:
        ds = hfd.load_from_disk(path)
    except Exception as e:
        err(f"Failed to load dataset at {path} with load_from_disk: {e}")
        return

    info(f"[HF Dataset] {path}")
    info(f" - num_rows: {len(ds)} | columns: {list(ds.features.keys())}")

    rows = sample_indices(len(ds), per_file_samples)
    for idx in rows:
        row = ds[int(idx)]
        print(f"   • Row #{idx}: keys={list(row.keys())}")
        text_field = None
        ids_field = None

        for cand in ["input_ids", "ids", "token_ids"]:
            if cand in row and isinstance(row[cand], (list, tuple)):
                ids_field = cand
                break
        for cand in ["text", "content", "instruction", "prompt"]:
            if cand in row and isinstance(row[cand], str):
                text_field = cand
                break

        if ids_field:
            ids_list = row[ids_field]
            ids_str = str(ids_list[:32]) + (" ... " if len(ids_list) > 32 else "")
            print(f"     - {ids_field}[:32]: {ids_str}")
            print(f"     - decode({ids_field}): {decode_ids(ids_list, tok)}")
            # Basic checks
            oob = [i for i in ids_list if not (0 <= int(i) < len(tok))]
            if oob:
                warn(f"     - OOB token IDs detected (showing first 10): {oob[:10]}  (vocab={len(tok)})")
            if tok.bos_token_id is not None and ids_list and ids_list[0] != tok.bos_token_id:
                warn("     - First token is not BOS.")
            if tok.eos_token_id is not None and ids_list and ids_list[-1] != tok.eos_token_id:
                warn("     - Last token is not EOS.")

        if text_field:
            text_val = row[text_field]
            short = (text_val[:120] + "…") if len(text_val) > 120 else text_val
            print(f"     - {text_field}[:120]: {short}")
            if has_control_chars(text_val):
                warn("     - Text contains control characters.")

def inspect_arrow_file(path: str, tok: AutoTokenizer, per_file_samples: int = 2):
    if pa_ipc is None:
        err("🤖 `pyarrow` is not installed; cannot inspect raw .arrow shards.")
        return

    try:
        with pa.memory_map(path, "r") as source:
            reader = pa_ipc.RecordBatchFileReader(source)
            n_batches = reader.num_record_batches
            n_rows_total = 0
            for b in range(n_batches):
                n_rows_total += reader.get_record_batch(b).num_rows
    except Exception as e:
        err(f"Failed to open Arrow file: {path} | {e}")
        return

    info(f"[Arrow Shard] {path} | record_batches={n_batches} | approx_rows={n_rows_total}")
    if n_batches == 0 or n_rows_total == 0:
        warn("   - No data in this shard.")
        return

    # Sample a few rows by scanning batches until we accumulate enough
    want = per_file_samples
    grabbed = 0
    rnd_rows = set()
    while len(rnd_rows) < min(want, n_rows_total):
        rnd_rows.add(random.randint(0, max(0, n_rows_total - 1)))
    rnd_rows = sorted(rnd_rows)

    # Walk through batches and extract the global row indices we need
    base = 0
    for b in range(n_batches):
        batch = reader.get_record_batch(b)
        cols = [c for c in batch.schema.names]
        n = batch.num_rows
        needed = [i - base for i in rnd_rows if base <= i < base + n]
        base += n
        if not needed:
            continue

        table = pa.Table.from_batches([batch])
        df = table.to_pandas()  # light conversion for a couple rows

        for rel_idx in needed:
            row = df.iloc[int(rel_idx)].to_dict()
            print(f"   • Row(global={rel_idx}): keys={list(row.keys())}")

            # common field detection
            ids_list = None
            text_val = None
            for cand in ["input_ids", "ids", "token_ids"]:
                if cand in row and isinstance(row[cand], (list, tuple)):
                    ids_list = list(row[cand])  # ensure pyarrow list -> python list
                    break
            for cand in ["text", "content", "instruction", "prompt"]:
                if cand in row and isinstance(row[cand], str):
                    text_val = row[cand]
                    break

            if ids_list is not None:
                ids_str = str(ids_list[:32]) + (" ... " if len(ids_list) > 32 else "")
                print(f"     - input_ids[:32]: {ids_str}")
                print(f"     - decode(input_ids): {decode_ids(ids_list, tok)}")
                oob = [i for i in ids_list if not (0 <= int(i) < len(tok))]
                if oob:
                    warn(f"     - OOB token IDs detected (first 10): {oob[:10]}  (vocab={len(tok)})")
                if tok.bos_token_id is not None and ids_list and ids_list[0] != tok.bos_token_id:
                    warn("     - First token is not BOS.")
                if tok.eos_token_id is not None and ids_list and ids_list[-1] != tok.eos_token_id:
                    warn("     - Last token is not EOS.")

            if text_val is not None:
                short = (text_val[:120] + "…") if len(text_val) > 120 else text_val
                print(f"     - text[:120]: {short}")
                if has_control_chars(text_val):
                    warn("     - Text contains control characters.")

# --------------------------- Main checks --------------------------------------

def check_tokenizer(tokenizer_path: str) -> AutoTokenizer:
    log_header("1) Tokenizer Health Check")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    info(f"Loaded tokenizer from: {tokenizer_path}")
    info(f"Vocab size: len(tok)={len(tok)} | tokenizer.vocab_size={getattr(tok, 'vocab_size', 'N/A')}")
    info(f"Special tokens: {tok.special_tokens_map}")
    print(f" - pad_token={tok.pad_token} (id={tok.pad_token_id})")
    print(f" - bos_token={tok.bos_token} (id={tok.bos_token_id})")
    print(f" - eos_token={tok.eos_token} (id={tok.eos_token_id})")
    print(f" - unk_token={tok.unk_token} (id={tok.unk_token_id})")

    # Round-trip smoke tests
    samples = ["Hello world!", "The quick brown fox jumps over 13 lazy dogs."]
    for s in samples:
        ids = tok(s)["input_ids"]
        dec = tok.decode(ids)
        print(f" - encode('{s}') -> {ids}")
        print(f" - decode(ids)   -> {dec}")

    if tok.pad_token is None:
        warn("Tokenizer has no pad_token. Consider adding one for batching/inference.")
    if tok.eos_token is None:
        warn("Tokenizer has no eos_token. Generation may not know when to stop.")

    ok("Tokenizer check complete.")
    return tok

def check_model_and_ckpt(
    model_import: str,
    model_class: str,
    ckpt_path: str,
    tok: AutoTokenizer,
    device: str
) -> nn.Module:
    log_header("2) Model + Checkpoint Compatibility")

    # Import TinyGPT dynamically so you can change the import path via CLI if needed
    TinyGPT = dynamic_import(model_import, model_class)

    # Instantiate with tokenizer vocab
    vocab_size = len(tok)
    student = TinyGPT(
        vocab_size=vocab_size,
        d_model=384,
        n_layers=8,
        n_heads=6,
        mlp_ratio=4,
        max_len=256,
        dropout=0.1,
    ).to(device)

    info(f"Instantiated TinyGPT with vocab_size={vocab_size}.")
    info(f"Loading checkpoint: {ckpt_path}")
    ckpt = try_torch_load(ckpt_path, map_location=device)

    # Find a likely state_dict payload in the checkpoint
    sd_candidates = []
    for k in ["model_state_dict", "state_dict", "model", "module", "student", "net"]:
        if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            sd_candidates.append(k)

    if sd_candidates:
        chosen = sd_candidates[0]
        state = ckpt[chosen]
        info(f"Using state_dict from checkpoint key: '{chosen}'")
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        warn("Checkpoint looks like a raw state_dict (top level).")
        state = ckpt
    else:
        raise RuntimeError(
            "Could not find a state_dict in the checkpoint. Keys: "
            f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )

    # Try loading with strict=False to report missing/unexpected keys
    missing, unexpected = student.load_state_dict(state, strict=False)
    if missing:
        warn(f"Missing keys in model when loading: (showing up to 20)\n - " + "\n - ".join(missing[:20]))
    if unexpected:
        warn(f"Unexpected keys in checkpoint: (showing up to 20)\n - " + "\n - ".join(unexpected[:20]))
    ok("Checkpoint loaded (strict=False).")

    # Check embedding + head modules
    emb = find_token_embedding_module(student)
    head = find_output_head_module(student)

    if emb is not None and emb.num_embeddings != vocab_size:
        warn(
            f"Token embedding size ({emb.num_embeddings}) != tokenizer vocab ({vocab_size}). "
            "If this model was trained with a different tokenizer, you must retrain or resize embeddings."
        )
    if head is not None and head.out_features != vocab_size:
        warn(
            f"Output head out_features ({head.out_features}) != tokenizer vocab ({vocab_size}). "
            "This mismatch will cause gibberish or runtime errors."
        )

    # Forward shape smoke test
    student.eval()
    with torch.no_grad():
        dummy = torch.tensor([[tok.bos_token_id or 0, 10, 11, 12][: min(4, vocab_size)]], device=device)
        try:
            out = student(dummy)
            logits = extract_logits(out, dummy)
            print(f"[Forward] logits shape normalized to: {tuple(logits.shape)} (expect [1, {vocab_size}])")
        except Exception as e:
            err(f"Forward smoke test failed: {e}\n{traceback.format_exc()}")

    ok("Model check complete.")
    return student

def check_generation(student: nn.Module, tok: AutoTokenizer, device: str):
    log_header("3) Generation Smoke Test (no CUDA asserts)")
    prompt = "Hello Zia!"
    enc = tok(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    try:
        full = safe_generate(
            student, tok, ids,
            max_new_tokens=32,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            device=device,
        )
        decoded = tok.decode(full[0].tolist(), skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Generated IDs (first 64): {full[0].tolist()[:64]}")
        print(f"Decoded output: {decoded}")
        ok("Generation test complete.")
    except Exception as e:
        err(f"Generation failed: {e}\n{traceback.format_exc()}")

def check_datasets(paths: List[str], tok: AutoTokenizer, per_file_samples: int):
    log_header("4) Dataset Integrity (Arrow/HF dirs)")
    if not paths:
        info("No dataset paths provided. Skipping dataset checks.")
        return

    for p in paths:
        print("\n---")
        print(f"[Dataset Path] {p}")
        if os.path.isdir(p) and is_hf_dataset_dir(p):
            inspect_hf_dataset_dir(p, tok, per_file_samples=per_file_samples)
        else:
            arrow_files = list_arrow_files(p)
            if not arrow_files:
                warn("No .arrow files found here.")
            for af in arrow_files:
                inspect_arrow_file(af, tok, per_file_samples=per_file_samples)
    ok("Dataset checks complete.")

# ----------------------------- CLI --------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="End-to-end validator for tokenizer, model, and datasets.")
    parser.add_argument("--tokenizer_path", required=True, help="Path to tokenizer dir")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--datasets", nargs="*", default=[], help="List of dataset dirs or .arrow files")
    parser.add_argument("--model-import", default=DEFAULT_MODEL_IMPORT,
                        help=f"Python module containing TinyGPT (default: {DEFAULT_MODEL_IMPORT})")
    parser.add_argument("--model-class", default="TinyGPT", help="Class name for your TinyGPT (default: TinyGPT)")
    parser.add_argument("--per-file-samples", type=int, default=2, help="Rows to sample per dataset shard (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Environment info
    log_header("0) Environment")
    print(f"Python: {sys.version.split()[0]} | Torch: {torch.__version__} | Transformers: "
          f"{__import__('transformers').__version__}")
    print(f"CUDA available: {torch.cuda.is_available()} | Selected device: {args.device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        tok = check_tokenizer(args.tokenizer_path)
        student = check_model_and_ckpt(args.model_import, args.model_class, args.ckpt, tok, args.device)
        check_generation(student, tok, args.device)
        check_datasets(args.datasets, tok, per_file_samples=args.per_file_samples)
    except Exception as e:
        err(f"Fatal error: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    ok("All checks executed.")

if __name__ == "__main__":
    main()
