import os
import torch
from torch import nn
from transformers import AutoTokenizer
from train_zia_dense import TinyGPT

def load_student(ckpt_path, tokenizer_path, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    vocab_size = len(tokenizer)
    model = TinyGPT(
        vocab_size=vocab_size,
        d_model=384,
        n_layers=8,
        n_heads=6,
        mlp_ratio=4,
        max_len=256,
        dropout=0.1
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    print(f"[i] Model loaded with vocab={vocab_size}")
    return model, tokenizer


def format_prompt(user_input):
    return f"User: {user_input}\nAssistant:"


@torch.no_grad()
def generate_response(model, tokenizer, prompt, device,
                      max_new_tokens=128, temperature=0.7, top_p=0.9, top_k=40, repetition_penalty=1.2):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Forward pass
        out = model(generated)
        # ✅ Ensure correct tensor extraction
        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        next_token_logits = logits / temperature

        # Repetition penalty
        for token_id in set(generated[0].tolist()):
            next_token_logits[0, token_id] /= repetition_penalty

        # Top-k filtering
        if top_k > 0:
            values, _ = torch.topk(next_token_logits, top_k)
            min_values = values[:, -1].unsqueeze(-1)
            next_token_logits[next_token_logits < min_values] = -float("Inf")

        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_logits[sorted_indices_to_remove] = -float("Inf")
            next_token_logits = torch.zeros_like(next_token_logits).scatter(1, sorted_indices, sorted_logits)

        probs = torch.softmax(next_token_logits, dim=-1)

        # 🔒 Handle NaNs or zero-sum probs
        if torch.isnan(probs).any() or probs.sum() == 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat((generated, next_token), dim=1)

        eos_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("</s>")
        if eos_id and next_token.item() == eos_id:
            break

    reply = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    reply = reply.replace("Ċ", "\n").replace("Ġ", " ").strip()
    return reply if reply else "(no output)"


def main():
    ckpt_path = "artifacts/zia_dense_ift/best_val/checkpoint.pt"
    tokenizer_path = "artifacts/zia_tokenizer_60k"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_student(ckpt_path, tokenizer_path, device)

    print("\n=== Chat with Zia (IFT-aligned) ===\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break

        prompt = format_prompt(user_input)
        reply = generate_response(model, tokenizer, prompt, device)
        print(f"Zia: {reply}\n")


if __name__ == "__main__":
    main()
