#!/usr/bin/env python3
# Encoder-safe forced inference for ZIA / TinyGPT
# Purpose: force continuation to probe grammar + semantics

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tinygpt_encoder import TinyGPT

# ---------------- CONFIG ----------------
CKPT_PATH = "artifacts/zia_ift_v22/ift_step5800.pt"
TOK_PATH  = "artifacts/zia_tokenizer_60k_clean"

MAX_NEW_TOKENS = 150
TEMPERATURE = 2.0          # 🔥 aggressive
TOP_P = 0.98
REPETITION_PENALTY = 1.25
MIN_NEW_TOKENS = 50

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
def generate_forced(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    generated = ids.clone()

    for step in range(MAX_NEW_TOKENS):
        logits = model(generated)
        logits = logits[:, -1, :]

        # 🔥 entropy boost
        logits = logits / TEMPERATURE

        # ❌ disable EOS early
        if tok.eos_token_id is not None and step < MIN_NEW_TOKENS:
            logits[:, tok.eos_token_id] = -1e9

        # 🔁 repetition penalty
        for token_id in set(generated[0].tolist()):
            logits[:, token_id] /= REPETITION_PENALTY

        # 🌊 SAFE nucleus sampling (no in-place ops, no aliasing)
        probs = F.softmax(logits, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)

            # keep tokens where cumulative prob <= TOP_P
        keep_mask = cumprobs <= TOP_P
        keep_mask[..., 0] = True  # always keep top token

        filtered_probs = sorted_probs * keep_mask
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(filtered_probs, 1)
        next_token = sorted_idx.gather(-1, next_token)

        generated = torch.cat([generated, next_token], dim=1)

    return tok.decode(generated[0], skip_special_tokens=True)


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
        print("\n" + "=" * 70)
        print("PROMPT:", p)
        print("-" * 70)
        print(generate_forced(model, tok, p))
