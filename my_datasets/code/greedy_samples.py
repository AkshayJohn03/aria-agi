# checks/greedy_sample.py
import torch
from train_zia_dense import TinyGPT
from transformers import AutoTokenizer

ckpt = "artifacts/zia_dense_runs/best_val/checkpoint.pt"
tokenizer_path = "artifacts/zia_tokenizer_60k"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = TinyGPT(vocab_size=len(tokenizer), d_model=384, n_layers=8, n_heads=6, mlp_ratio=4, max_len=256, dropout=0.1).to(device)
ck = torch.load(ckpt, map_location=device)
model.load_state_dict(ck.get("model", ck.get("model_state_dict", ck)), strict=False)
model.eval()

prompt = "User: What is the capital of India?\nAssistant:"
enc = tokenizer(prompt, return_tensors="pt").to(device)
generated = enc["input_ids"]
with torch.no_grad():
    for _ in range(80):
        out = model(generated)
        logits = out[0] if isinstance(out, (tuple,list)) else out
        logits = logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
print("Decoded:", tokenizer.decode(generated[0], skip_special_tokens=True))
