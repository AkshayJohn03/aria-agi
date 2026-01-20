#!/usr/bin/env python3
"""
clean_tokenizer_visual_check.py

Purpose:
- Inspect tokenizer vocabulary for unwanted characters (Ċ, Ġ, ĉ, č, ż, etc.)
- Optionally rewrite tokenizer by removing or replacing these tokens.
"""

import os
import re
from transformers import AutoTokenizer

TOKENIZER_DIR = "artifacts/zia_tokenizer_60k"

# Characters we SHOULD NOT HAVE
BAD_CHARS = r"[ĊĈČĠġżźžħıňŧĐđŁłťŉ]"   # Add anything you want flagged

def main():
    print("=== TOKENIZER INSPECTION ===")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)

    bad_tokens = []

    for token, idx in tok.get_vocab().items():
        if re.search(BAD_CHARS, token):
            bad_tokens.append((idx, token))

    print(f"\n[i] Total vocab size: {len(tok.get_vocab())}")
    print(f"[i] Found {len(bad_tokens)} suspicious tokens:\n")

    for idx, token in bad_tokens[:50]:
        print(f"  {idx:6d} | {repr(token)}")

    if len(bad_tokens) > 50:
        print(f"... ({len(bad_tokens)-50} more)")

    # -------------------------------------------------------------------
    # OPTIONAL CLEANING (Disabled by default)
    # -------------------------------------------------------------------
    SHOULD_CLEAN = False  # <<< change to True if you want to rewrite tokenizer

    if SHOULD_CLEAN:
        print("\n[!] Cleaning tokenizer...")
        new_vocab = {}

        for tok_str, tok_id in tok.get_vocab().items():
            if not re.search(BAD_CHARS, tok_str):
                new_vocab[tok_str] = tok_id

        print(f"[i] Clean vocab size → {len(new_vocab)}")

        tok.vocab = new_vocab  # rewrite internal dict
        tok.save_pretrained(TOKENIZER_DIR + "_cleaned")
        print(f"[✓] Saved cleaned tokenizer at {TOKENIZER_DIR}_cleaned")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
