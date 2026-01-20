#!/usr/bin/env python3
# tokenizer_clean.py – rebuilds clean tokenizer for ZIA (no Ġ prefixes)

import os
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

# make sure this file exists – replace with your actual text corpus
input_files = ["datasets/train_corpus.txt"]

if not all(os.path.exists(p) for p in input_files):
    raise FileNotFoundError("Corpus file not found. Place a .txt dataset in datasets/train_corpus.txt")

out_dir = "artifacts/zia_tokenizer_clean"
os.makedirs(out_dir, exist_ok=True)

# train a new tokenizer (no Ġ, clean spacing)
tok = ByteLevelBPETokenizer()
tok.train(
    files=input_files,
    vocab_size=60000,
    min_frequency=2,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>"]
)

# disable GPT2 postprocessor
tok._tokenizer.post_processor = None

# save both JSON and legacy files
tok.save_model(out_dir)
tok.save(os.path.join(out_dir, "tokenizer.json"))

# wrap in HF fast tokenizer format
tokfast = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join(out_dir, "tokenizer.json"),
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    unk_token="<unk>"
)
tokfast.save_pretrained(out_dir)

print(f"[✓] Saved clean tokenizer at {out_dir}")
