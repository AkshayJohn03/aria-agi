#!/usr/bin/env python3
# Encoder-safe inference for ZIA / TinyGPT
# Focus: sentence continuation, semantic sanity check

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from tinygpt_encoder import TinyGPT

# ---------------- CONFIG ----------------
CKPT_PATH = "artifacts/zia_ift_v22/ift_step5800.pt"  # change if needed
TOK_PATH  = "artifacts/zia_tokenizer_60k_clean"

MAX_NEW_TOKENS = 120
TEMPERATURE = 0.85
TOP_P = 0.92
TOP_K = 50
REPETITION_PENALTY = 1.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------


def load_model():
    print("[i] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(TOK_PATH, local_files_only=True)

    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})

    print("[i] Loading model...")
    model = TinyGPT(
        vocab_size=len(tok),
        d_model=384,
        n_layers=8,
        n_heads=6,
        max_len=4096,
        dropout=0.1
    ).to(DEVICE)

    ck = torch.load(CKPT_PATH, map_location="cpu")
    state = ck["model"] if "model" in ck else ck
    model.load_state_dict(state, strict=True)

    model.eval()
    print("[✓] Model ready\n")
    return model, tok


@torch.no_grad()
def generate(model, tok, prompt):
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    generated = input_ids.clone()

    for _ in range(MAX_NEW_TOKENS):
        if generated.size(1) > model.max_len:
            generated = generated[:, -model.max_len:]

        logits = model(generated)              # ✅ encoder returns logits only
        logits = logits[:, -1, :]

        # repetition penalty
        for t in set(generated[0].tolist()):
            logits[0, t] /= REPETITION_PENALTY

        logits = logits / TEMPERATURE
        probs = torch.softmax(logits, dim=-1)

        topk_probs, topk_idx = torch.topk(probs, TOP_K)
        cumulative = torch.cumsum(topk_probs, dim=-1)
        mask = cumulative <= TOP_P
        mask[..., 0] = True

        filtered = topk_probs * mask
        filtered /= filtered.sum()

        next_token = topk_idx[0, torch.multinomial(filtered, 1)]
        generated = torch.cat([generated, next_token.view(1, 1)], dim=1)

        if tok.eos_token_id and next_token.item() == tok.eos_token_id:
            break

    return tok.decode(generated[0], skip_special_tokens=True).strip()


# ---------------- MAIN ----------------
if __name__ == "__main__":
    model, tok = load_model()

    prompts = [
        "The theory of evolution explains that",
        "In physics, energy is defined as",
        "Once upon a time,",
        "The human brain works by",
        "India is a country that"
    ]

    for p in prompts:
        print("\n" + "=" * 60)
        print("PROMPT:", p)
        print("-" * 60)
        out = generate(model, tok, p)
        print(out)
