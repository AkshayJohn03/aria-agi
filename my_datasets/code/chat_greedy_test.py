import torch
from transformers import AutoTokenizer
from train_zia_dense import TinyGPT

ckpt = "recovered_best.pt"
tokenizer_path = "artifacts/zia_tokenizer_60k"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

model = TinyGPT(
    vocab_size=len(tokenizer),
    d_model=384,
    n_layers=8,
    n_heads=6,
    mlp_ratio=4,
    max_len=256,
    dropout=0.1,
).to(device)

ck = torch.load(ckpt, map_location=device)
state_dict = ck.get("model", ck.get("model_state_dict", ck))
model.load_state_dict(state_dict, strict=False)
model.eval()

prompt = "User: What is the capital of India?\nAssistant:"
enc = tokenizer(prompt, return_tensors="pt").to(device)
generated = enc["input_ids"]

with torch.no_grad():
    for _ in range(80):
        out = model(generated)
        logits = out if isinstance(out, torch.Tensor) else out[0]
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
print("\n[Generated Output]\n", decoded)
