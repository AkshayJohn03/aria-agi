#!/usr/bin/env python3
"""
align_model_to_clean_tokenizer_safe.py

Safely align an existing checkpoint's embedding/head weights to a clean (ChatML) tokenizer.
- Backups original checkpoint (safe)
- Tries robust token name matching (several variants)
- Preserves all non-embedding params intact
- Reports mapping stats and writes a small JSON report
"""
import os
import json
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast

# --- CONFIG (edit paths if needed) ---
OLD_CKPT = "artifacts/zia_ift_fixed4k/checkpoint_step240.pt"
OLD_TOK_FILE = "artifacts/zia_tokenizer_60k/tokenizer.json"       # legacy tokenizer used by checkpoint
NEW_TOK_DIR = "artifacts/zia_tokenizer_60k_clean"                # clean ChatML tokenizer we want to adopt
OUT_CKPT = "artifacts/zia_ift_v5/zia_v5_aligned.pt"              # resulting aligned checkpoint
REPORT_JSON = "artifacts/zia_ift_v5/align_report.json"
D_MODEL = 384
RNG_SEED = 42
# ---------------------------------------

torch.manual_seed(RNG_SEED)

def safe_load_ckpt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ck = torch.load(path, map_location="cpu")
    # try to extract state dict
    if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
        state = ck["model"]
    else:
        state = ck
    return ck, state

def find_key_variant(state, base_key):
    # Some checkpoints/architectures may prefix module names; try variants
    if base_key in state:
        return base_key
    variants = [base_key, "model." + base_key, "module." + base_key, "model.module." + base_key]
    for v in variants:
        if v in state:
            return v
    return None

def try_token_names(tok, token):
    """Return token id if token exists otherwise None. Try variants for common differences."""
    vocab = tok.get_vocab()
    if token in vocab:
        return vocab[token]
    # variants
    variants = [
        token,
        token.replace("<|", "<").replace("|>", ">"),    # <|user|> -> <user>
        token.replace("|", ""),                         # <|user|> -> <user>
        token.strip(),
        token.lower(),
        token.replace("<|", "").replace("|>", "")       # remove pipe delimiters
    ]
    for v in variants:
        if v in vocab:
            return vocab[v]
    return None

def main():
    os.makedirs(os.path.dirname(OUT_CKPT), exist_ok=True)
    print("[i] Loading checkpoint...")
    ck_full, state = safe_load_ckpt(OLD_CKPT)

    # locate tok.weight and head.weight keys robustly
    tok_key = find_key_variant(state, "tok.weight")
    head_key = find_key_variant(state, "head.weight")
    if tok_key is None or head_key is None:
        raise KeyError("Could not find 'tok.weight' or 'head.weight' in checkpoint state dict. Keys available: "
                       + ", ".join(list(state.keys())[:50]))

    old_tok_weight = state[tok_key]
    old_head_weight = state[head_key]
    print(f"[i] Found embedding shapes: {tok_key} {tuple(old_tok_weight.shape)}, {head_key} {tuple(old_head_weight.shape)}")

    # load tokenizers
    print("[i] Loading tokenizers...")
    old_tok = PreTrainedTokenizerFast(tokenizer_file=OLD_TOK_FILE)
    new_tok = PreTrainedTokenizerFast.from_pretrained(NEW_TOK_DIR, local_files_only=True)

    # compute required target vocab size by checking max id used by new tokenizer
    new_vocab_map = new_tok.get_vocab()
    max_new_id = max(new_vocab_map.values())
    target_vocab_size = max_new_id + 1
    print(f"[i] New tokenizer max token id: {max_new_id} -> target_vocab_size = {target_vocab_size}")

    # If checkpoint embeddings have smaller size, expand; if larger, we'll keep only overlapping + init rest
    old_vocab_size, d_old = old_tok_weight.shape
    if d_old != D_MODEL:
        print(f"[!] Warning: checkpoint D_MODEL ({d_old}) != configured D_MODEL ({D_MODEL}). Using checkpoint D_MODEL.")
        DMODEL = d_old
    else:
        DMODEL = D_MODEL

    # prepare new embedding matrices
    new_tok_weight = torch.zeros(target_vocab_size, DMODEL)
    new_head_weight = torch.zeros(target_vocab_size, DMODEL)
    nn.init.normal_(new_tok_weight, mean=0.0, std=0.02)
    nn.init.normal_(new_head_weight, mean=0.0, std=0.02)

    # vocab maps
    old_vocab = old_tok.get_vocab()
    new_vocab = new_tok.get_vocab()

    mapped_tokens = []
    unmapped_tokens = []
    mapped_specials = []

    # 1) Direct string matches: token present in both vocabularies
    for token, new_id in new_vocab.items():
        if token in old_vocab:
            old_id = old_vocab[token]
            if old_id < old_vocab_size:
                new_tok_weight[new_id] = old_tok_weight[old_id]
                new_head_weight[new_id] = old_head_weight[old_id]
                mapped_tokens.append((token, old_id, new_id))
            else:
                unmapped_tokens.append((token, None, new_id))

    # 2) Special-case mapping table - ensure critical roles match
    special_map_candidates = {
        "<|user|>": "<user>",
        "<|assistant|>": "<assistant>",
        "<|system|>": "<system>",
        "<pad>": "[PAD]",
        "<s>": "[BOS]",
        "</s>": "[EOS]",
        "<unk>": "[UNK]",
    }

    print("\n[i] Running explicit special-token mapping (best-effort)...")
    for new_t, old_t in special_map_candidates.items():
        n_id = try_token_names(new_tok, new_t)
        o_id = try_token_names(old_tok, old_t)
        if n_id is not None and o_id is not None and o_id < old_vocab_size:
            new_tok_weight[n_id] = old_tok_weight[o_id]
            new_head_weight[n_id] = old_head_weight[o_id]
            mapped_specials.append((new_t, old_t, o_id, n_id))
        else:
            # log but do not throw
            print(f"  - could not map {new_t} <-> {old_t} (n_id={n_id} o_id={o_id})")

    # 3) Report summary
    print("\n[i] Mapping summary:")
    print(f"  total new vocab size: {target_vocab_size}")
    print(f"  old checkpoint vocab size: {old_vocab_size}")
    print(f"  mapped by direct match: {len(mapped_tokens)}")
    print(f"  mapped special tokens: {len(mapped_specials)}")
    print(f"  random-initialized tokens (count): {target_vocab_size - (len(mapped_tokens) + len(mapped_specials))}")

    # Safety: replace weights in a copy of the original state dict so all other params preserved
    new_state = dict(state)  # shallow copy
    new_state[tok_key] = new_tok_weight
    new_state[head_key] = new_head_weight

    # create output checkpoint packet (preserve optimizer if present? we'll drop it to be safe)
    out_ck = {
        "model": new_state,
        "step": ck_full.get("step", 0),
        "config": ck_full.get("config", {"d_model": DMODEL, "vocab_size": target_vocab_size})
    }

    # Backup original checkpoint first
    backup_path = OLD_CKPT + ".bak"
    if not os.path.exists(backup_path):
        try:
            print(f"[i] Backing up original checkpoint to: {backup_path}")
            torch.save(ck_full, backup_path)
        except Exception as e:
            print(f"[!] Could not write backup: {e}")

    # Save aligned checkpoint
    torch.save(out_ck, OUT_CKPT)
    print(f"\n✅ Aligned checkpoint written: {OUT_CKPT}")

    # Save report
    report = {
        "old_ckpt": OLD_CKPT,
        "out_ckpt": OUT_CKPT,
        "old_vocab_size": old_vocab_size,
        "target_vocab_size": target_vocab_size,
        "mapped_tokens_count": len(mapped_tokens),
        "mapped_specials": mapped_specials,
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[i] Report saved: {REPORT_JSON}")

    print("\n[i] NEXT: run a small inference test using the aligned checkpoint + clean tokenizer.")
    print("    e.g. python my_datasets/code/inference_new_Spm.py (pointing to OUT_CKPT)")

if __name__ == "__main__":
    main()
