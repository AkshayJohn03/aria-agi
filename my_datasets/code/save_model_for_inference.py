#!/usr/bin/env python3
"""
save_model_for_inference.py
Unified inference + validation-compatible loader for ZIA TinyGPT models.
- Fully compatible with v3 dense & v4 cursor training checkpoints.
- Forward pass supports labels for evaluation.
- Handles vocab expansion, position embedding resizing.
- Includes compatible sample_generate() for diagnosis and inference.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# --- Silence TRANSFORMERS_CACHE warning & set cache dirs early ---
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------
# Model Definition
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, mlp_ratio, dropout):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mlp_ratio, dropout)

    def forward(self, x, key_padding_mask=None):
        h = self.ln1(x)
        T = h.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(h, h, h, attn_mask=causal_mask,
                           key_padding_mask=key_padding_mask, need_weights=False)
        x = x + out
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=8, n_heads=6,
                 mlp_ratio=4, max_len=1024, dropout=0.1, pad_token_id=0):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tie weights
        self.max_len = max_len
        self.pad_token_id = pad_token_id

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape
        if T > self.max_len:
            input_ids = input_ids[:, -self.max_len:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_len:]
            T = input_ids.size(1)

        pos_ids = torch.arange(0, T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok(input_ids) + self.pos(pos_ids)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        for blk in self.blocks:
            x = blk(x, key_padding_mask)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        return logits, loss


# -----------------------------
# Utility Functions
# -----------------------------
def expand_pos_and_vocab(state_dict, model):
    """Handle position and vocab expansion mismatches."""
    sd = model.state_dict()
    if "pos.weight" in state_dict and "pos.weight" in sd:
        src = state_dict["pos.weight"]
        dst = sd["pos.weight"]
        old_len, _ = src.size()
        new_len, _ = dst.size()
        if new_len > old_len:
            print(f"[i] Expanding pos.weight {old_len}→{new_len}")
            last = src[-1:].repeat(new_len - old_len, 1)
            state_dict["pos.weight"] = torch.cat([src, last], dim=0)
        elif new_len < old_len:
            print(f"[i] Truncating pos.weight {old_len}→{new_len}")
            state_dict["pos.weight"] = src[:new_len]
    for key in ("tok.weight", "head.weight"):
        if key in state_dict and key in sd:
            src = state_dict[key]
            dst = sd[key]
            if src.shape != dst.shape:
                print(f"[i] Adjusting {key} from {tuple(src.shape)}→{tuple(dst.shape)}")
                dst_copy = dst.clone()
                n_copy = min(src.size(0), dst.size(0))
                dst_copy[:n_copy] = src[:n_copy]
                if dst.size(0) > src.size(0):
                    nn.init.normal_(dst_copy[src.size(0):], std=0.02)
                state_dict[key] = dst_copy
    return state_dict


def load_student(ckpt_path, tokenizer_path, device="cpu"):
    """Load TinyGPT with tokenizer and flexible checkpoint handling."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    vocab_size = len(tokenizer)
    print(f"[i] Loaded tokenizer vocab_size={vocab_size}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)

    max_len = 1024
    if "pos.weight" in state_dict:
        max_len = state_dict["pos.weight"].size(0)
        print(f"[i] Detected max_len={max_len}")

    student = TinyGPT(vocab_size=vocab_size, max_len=max_len).to(device)
    state_dict = expand_pos_and_vocab(state_dict, student)
    student.load_state_dict(state_dict, strict=False)
    student.eval()

    test_ids = torch.randint(0, vocab_size, (1, 5)).to(device)
    with torch.no_grad():
        out, _ = student(test_ids)
    top_ids = torch.topk(out[0, -1], 3).indices.tolist()
    print("[i] Sanity check tokens:", test_ids.tolist())
    print("[i] Decoded input:", tokenizer.decode(test_ids[0]))
    print("[i] Decoded top-preds:", tokenizer.decode(top_ids))
    return student, tokenizer


# -----------------------------
# Generation utilities
# -----------------------------
@torch.no_grad()
def sample_generate(
    student,
    tokenizer,
    input_ids,
    device,
    max_length=100,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.1,
):
    """
    Safe and numerically stable text generator for TinyGPT.
    - Avoids CUDA asserts due to invalid tokens (id=0)
    - Clamps sampled tokens within valid vocab range
    - Recovers from NaN/inf probabilities
    """
    student.eval()
    generated = input_ids.clone().to(device)

    for step in range(max_length):
        try:
            logits, _ = student(generated)
        except Exception as e:
            print(f"[!] Forward failed at step {step}: {e}")
            break

        # Focus on last token logits
        next_token_logits = logits[:, -1, :]

        # Apply repetition penalty
        for token_id in set(generated.view(-1).tolist()):
            next_token_logits[:, token_id] /= repetition_penalty

        # Temperature scaling
        next_token_logits = next_token_logits / max(temperature, 1e-5)

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            min_v = v[:, -1].unsqueeze(-1)
            next_token_logits[next_token_logits < min_v] = -float("inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(probs, dim=-1)
            sorted_indices_to_remove = cum_probs > top_p
            if torch.any(sorted_indices_to_remove):
                sorted_logits[sorted_indices_to_remove] = -float("inf")
                next_token_logits.zero_().scatter_(1, sorted_indices, sorted_logits)

        # Convert logits to probabilities safely
        probs = F.softmax(next_token_logits, dim=-1)

        # Replace NaN/Inf with safe defaults
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # Sample next token safely
        next_token = torch.multinomial(probs, num_samples=1)

        # Clamp to valid range [1, vocab_size - 1]
        next_token = torch.clamp(next_token, 1, tokenizer.vocab_size - 1)

        generated = torch.cat([generated, next_token], dim=1)

        # Stop on EOS
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    return generated


# -----------------------------
# Chat Loop (Interactive)
# -----------------------------
def chat_loop(student, tokenizer, device="cpu"):
    print("\n=== Chat with ZIA (type 'exit' to quit) ===\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break
        formatted = f"Instruction: {user_input}\nResponse:"
        enc = tokenizer(formatted, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        outputs = sample_generate(
            student, tokenizer, input_ids, device,
            max_length=200, temperature=0.3, top_k=20, top_p=0.8, repetition_penalty=1.2
        )
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Response:" in reply:
            reply = reply.split("Response:")[-1].strip()
        reply = reply.replace("Ġ", " ").replace("  ", " ").strip()
        print(f"Zia: {reply}\n")
