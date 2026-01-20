import torch
from tinygpt_encoder import TinyGPT

ckpt = "artifacts/zia_lm_v20/v20_step977_t1765969109.pt"

model = TinyGPT(
    vocab_size=60011,
    d_model=384,
    n_layers=8,
    n_heads=6,
    max_len=4096,
    dropout=0.1
)

state = torch.load(ckpt, map_location="cpu")
state = state["model"] if "model" in state else state

model.load_state_dict(state, strict=True)
print("✅ Checkpoint loads cleanly")
