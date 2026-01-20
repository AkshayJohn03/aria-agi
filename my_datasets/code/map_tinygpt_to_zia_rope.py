import torch

# Input: your existing trained model (IFTv2 or dense)
src = r"D:\aria\aria_ai\aria_ai_assistant\artifacts\zia_ift_v2_runs\best_val\checkpoint.pt"
dst = r"D:\aria\aria_ai\aria_ai_assistant\artifacts\zia_mapped_rope_ready.pt"

ck = torch.load(src, map_location="cpu")
state_dict = ck["model"]
new_state = {}

print(f"[i] Converting checkpoint from {src}")

for k, v in state_dict.items():
    nk = k
    nk = nk.replace("tok.", "transformer.wte.")
    nk = nk.replace("pos.", "transformer.wpe.")
    nk = nk.replace("blocks.", "transformer.h.")
    nk = nk.replace(".ln1.", ".ln_1.")
    nk = nk.replace(".ln2.", ".ln_2.")
    nk = nk.replace(".attn.in_proj_weight", ".attn.c_attn.weight")
    nk = nk.replace(".attn.in_proj_bias", ".attn.c_attn.bias")
    nk = nk.replace(".attn.out_proj.weight", ".attn.c_proj.weight")
    nk = nk.replace(".attn.out_proj.bias", ".attn.c_proj.bias")
    nk = nk.replace(".ff.net.0.", ".mlp.0.")
    nk = nk.replace(".ff.net.2.", ".mlp.2.")
    new_state[nk] = v

torch.save({"model": new_state}, dst)
print(f"[✓] Saved remapped checkpoint → {dst}")
