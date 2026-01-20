#!/usr/bin/env python3
"""
build_tokeniser_v3.py
Automatically picks the correct vocab size based on corpus.
Retries vocab_size until SentencePiece accepts it.
"""

import os, sys, sentencepiece as spm

CORPUS = "artifacts/zia_tokenizer_v2_clean/merged_corpus.txt"
OUT_DIR = "artifacts/zia_tokenizer_v2_clean"
os.makedirs(OUT_DIR, exist_ok=True)

# Try vocab sizes from high → low
VOCAB_TRY = [32000, 28000, 24000]

USER_SYMBOLS = [
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "[BOS]",
    "[EOS]",
]

def try_train(vocab):
    print(f"\n=== Trying vocab_size = {vocab} ===")
    try:
        spm.SentencePieceTrainer.Train(
            input=CORPUS,
            model_prefix=os.path.join(OUT_DIR, "zia_spm"),
            vocab_size=vocab,
            model_type="unigram",
            character_coverage=1.0,
            input_sentence_size=200000,
            shuffle_input_sentence=True,
            num_threads=8,
            user_defined_symbols=",".join(USER_SYMBOLS),
            split_by_unicode_script=True,
            split_by_whitespace=True,
            hard_vocab_limit=True,
            max_sentence_length=4096,
        )
        print(f"[✓] SUCCESS: trained tokenizer with vocab_size={vocab}")
        return True
    except Exception as e:
        print(f"[!] Failed for vocab_size={vocab} → {str(e)}")
        return False

def main():
    if not os.path.exists(CORPUS):
        print(f"[ERROR] Corpus not found: {CORPUS}")
        sys.exit(1)

    for v in VOCAB_TRY:
        if try_train(v):
            print(f"[FINAL] Tokenizer available at: {OUT_DIR}")
            return

    print("[×] All attempts failed. Try reducing vocab_size further.")

if __name__ == "__main__":
    main()
