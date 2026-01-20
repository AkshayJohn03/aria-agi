#!/usr/bin/env python3
# verify_pipeline_all.py
import os, json, torch
from pathlib import Path
from transformers import PreTrainedTokenizerFast

TOKEN_DIR = "artifacts/zia_tokenizer_60k_clean"
SHARD_DIR = "datasets/processed/chatml_60k_4096_stream"
ARROW_DIR = "datasets/local/openorca"
CHECKPOINTS = [
    "artifacts/zia_ift_fixed4k/checkpoint_step240.pt",
    "artifacts/zia_ift_v4_longctx/best_val_5255.pt"
]

def verify_tokenizer():
    print("== TOKENIZER ==")
    tok = PreTrainedTokenizerFast.from_pretrained(TOKEN_DIR, local_files_only=True)
    print("vocab_size:", tok.vocab_size, "pad:", tok.pad_token_id, "bos:", tok.bos_token_id, "eos:", tok.eos_token_id)
    for r in ["<|user|>","<|assistant|>","<|system|>","<|im_start|>","<|im_end|>"]:
        ids = tok.encode(r, add_special_tokens=False)
        print(f"'{r}' -> {ids} | atomic={len(ids)==1}")
    sample = "<|im_start|>user\nHello<|im_end|>"
    enc = tok.encode(sample, add_special_tokens=False)
    dec = tok.decode(enc, skip_special_tokens=False)
    print("roundtrip ok:", sample == dec)

def verify_shards():
    print("\n== SHARDS ==")
    p = Path(SHARD_DIR)
    if not p.exists():
        print("no shards at", SHARD_DIR); return
    found = list(p.glob("**/input_ids.pt"))
    print("found shards:", len(found))
    for s in found[:5]:
        x = torch.load(s, map_location="cpu")
        y = torch.load(str(s).replace("input_ids.pt","labels.pt"), map_location="cpu")
        print(s, "->", x.shape, y.shape)
        # print sample decode (first row)
        tok = PreTrainedTokenizerFast.from_pretrained(TOKEN_DIR, local_files_only=True)
        print("decoded sample (first 256 tokens):", tok.decode(x[0][:256].tolist(), skip_special_tokens=False))
        break

def verify_arrows():
    print("\n== ARROW FILES ==")
    files = list(Path(ARROW_DIR).glob("*.arrow"))
    print("arrow files:", len(files))
    for f in files[:6]:
        print(" ", f.name, f.stat().st_size)

def verify_checkpoints():
    print("\n== CHECKPOINTS ==")
    for ck in CHECKPOINTS:
        if not Path(ck).exists():
            print(" MISSING:", ck); continue
        print("Inspecting:", ck)
        ckobj = torch.load(ck, map_location="cpu")
        keys = list(ckobj.keys())
        print(" keys:", keys[:6], "... total:", len(keys))
        # inspect head shapes if present
        model_state = ckobj.get("model", ckobj)
        # find tok/head weights
        for k in model_state.keys():
            if "tok.weight" in k or "head.weight" in k:
                print("  ", k, "->", model_state[k].shape)
        # quick load attempt into dummy model dims (warn if mismatch)
    print("[✅] verification written to stdout")

if __name__ == "__main__":
    verify_tokenizer()
    verify_shards()
    verify_arrows()
    verify_checkpoints()
    print("\n[✅] verification completed.")
    