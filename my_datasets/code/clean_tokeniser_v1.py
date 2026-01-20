#!/usr/bin/env python3
# clean_tokenizer_60k.py
# Loads your existing tokenizer.json (60k), fixes pad/bos/eos, and registers atomic special tokens
# Saves cleaned tokenizer to artifacts/zia_tokenizer_60k_clean/

import os
from transformers import PreTrainedTokenizerFast

SRC = "artifacts/zia_tokenizer_60k/tokenizer.json"
OUTDIR = "artifacts/zia_tokenizer_60k_clean"

# Role tokens we want atomic
ROLE_TOKENS = ["<|user|>", "<|assistant|>", "<|system|>", "<|im_start|>", "<|im_end|>", "<pad>", "<unk>"]

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("[i] Loading:", SRC)
    tok = PreTrainedTokenizerFast(tokenizer_file=SRC)
    # Inspect current mappings
    print("[i] vocab_size:", tok.vocab_size)
    print("[i] bos, eos, pad:", tok.bos_token, tok.eos_token, tok.pad_token)

    # Ensure pad/bos/eos are set
    add = {}
    if tok.pad_token is None:
        add["pad_token"] = "<pad>"
    if tok.bos_token is None:
        add["bos_token"] = "<s>"
    if tok.eos_token is None:
        add["eos_token"] = "</s>"

    # Add role tokens as additional_special_tokens but **do not** change existing token ids
    # We prefer to map the role text to existing token IDs if present; else add as new tokens.
    existing = set(tok.get_vocab().keys())
    new_specials = []
    for r in ROLE_TOKENS:
        if r in existing:
            new_specials.append(r)
        else:
            new_specials.append(r)

    if add or new_specials:
        added = {}
        if new_specials:
            added["additional_special_tokens"] = new_specials
        added.update(add)
        tok.add_special_tokens(added)
        print("[i] added special tokens:", added)

    # Save cleaned tokenizer dir
    tok.save_pretrained(OUTDIR)
    print("[✅] Saved cleaned tokenizer to:", OUTDIR)

    # Atomicity check
    print("\n[i] Atomicity check (encode with add_special_tokens=False):")
    for r in ["<|user|>", "<|assistant|>", "<|system|>", "<|im_start|>", "<|im_end|>"]:
        ids = tok.encode(r, add_special_tokens=False)
        print(f"  {r} -> {ids} | atomic: {len(ids)==1}")

    # Round-trip sanity
    sample = "<|im_start|>user\nHello world<|im_end|>"
    enc = tok.encode(sample, add_special_tokens=False)
    dec = tok.decode(enc, skip_special_tokens=False)
    print("\n[i] roundtrip check ok:", sample == dec)
    print("[i] original:", sample)
    print("[i] decoded :", dec)

if __name__ == "__main__":
    main()
