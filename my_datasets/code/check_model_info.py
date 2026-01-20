import torch
from transformers import AutoTokenizer

# --- Paths ---
tokenizer_path = "artifacts/zia_tokenizer_60k"
model_ckpt = "artifacts/zia_dense_base/recovered_best.pt"

# --- Tokenizer info ---
print("\n[Tokenizer Info]")
tok = AutoTokenizer.from_pretrained(tokenizer_path)
print(f"vocab_size = {tok.vocab_size}")
print(f"pad_token = {tok.pad_token} | eos_token = {tok.eos_token}")
print(f"special_tokens = {tok.special_tokens_map}")

# --- Model info ---
print("\n[Model Info]")
# The following line may trigger a FutureWarning, which is fine to ignore for now.
ckpt = torch.load(model_ckpt, map_location='cpu')

if "model_state_dict" in ckpt:
    sd = ckpt["model_state_dict"]
else:
    sd = ckpt

# Iterate, but only process items that are actual Tensors
for name, param in sd.items():
    # FIX: Check if the value is a Tensor before accessing .shape
    if isinstance(param, torch.Tensor):
        print(name, param.shape)
        break # Exit after successfully printing the shape of the first Tensor

print(f"Total params loaded: {len(sd)}")
print(f"Keys sample: {list(sd.keys())[:10]}")