# checks/check_vocab_head.py
import torch, os
from transformers import AutoTokenizer
from train_zia_dense import TinyGPT

ckpt = "artifacts/zia_dense_runs/best_val/checkpoint.pt"   # change if needed
tokenizer_path = "artifacts/zia_tokenizer_60k"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
print("Tokenizer vocab_size:", len(tokenizer), "pad:", tokenizer.pad_token, "eos:", tokenizer.eos_token)

model = TinyGPT(vocab_size=len(tokenizer), d_model=384, n_layers=8, n_heads=6, mlp_ratio=4, max_len=256, dropout=0.1).to(device)
ck = torch.load(ckpt, map_location=device)
if "model" in ck:
    model.load_state_dict(ck["model"], strict=False)
elif "model_state_dict" in ck:
    model.load_state_dict(ck["model_state_dict"], strict=False)
else:
    model.load_state_dict(ck, strict=False)

# head shape
head = getattr(model, "lm_head", None) or getattr(model, "head", None)
print("Model head out_features:", getattr(head, "out_features", None))
try:
    import numpy as np
    emb_norm = model.tok.weight.data.norm(dim=1).cpu().numpy()
    print("embed norm (min,mean,max):", emb_norm.min(), emb_norm.mean(), emb_norm.max())
    head_norm = head.weight.data.norm(dim=1).cpu().numpy()
    print("head norm (min,mean,max):", head_norm.min(), head_norm.mean(), head_norm.max())
except Exception as e:
    print("Norm calc error:", e)
