#!/usr/bin/env python3
# pos_resize.py
import torch, os, glob
import torch.nn.functional as F
import numpy as np

CHECKPOINT = "artifacts/zia_ift_v4_longctx/checkpoints/checkpoint_auto_step5255.pt"
OUT = CHECKPOINT.replace(".pt", ".posresized.pt")
NEW_CTX = 16384  # target new context length

def resize_pos_emb(pos_weight: torch.Tensor, new_len:int):
    # pos_weight shape: (old_len, d_model)
    old_len, d_model = pos_weight.shape
    if new_len == old_len:
        return pos_weight.clone()
    # linear/spline interpolation along position axis
    # make shape (1, d_model, old_len) for F.interpolate (which expects N,C,L)
    tensor = pos_weight.t().unsqueeze(0)  # (1, d_model, old_len)
    new = F.interpolate(tensor, size=new_len, mode='linear', align_corners=True)
    new = new.squeeze(0).t().contiguous()
    return new

def main():
    ck = torch.load(CHECKPOINT, map_location="cpu")
    model_state = ck.get("model", ck)
    if "pos.weight" not in model_state:
        print("No pos.weight found in checkpoint.")
        return
    pos = model_state["pos.weight"]
    print("old pos shape:", pos.shape)
    newpos = resize_pos_emb(pos, NEW_CTX)
    print("new pos shape:", newpos.shape)
    model_state["pos.weight"] = newpos
    # update metadata
    ck["context_len"] = NEW_CTX
    # save new ckpt
    torch.save(ck, OUT)
    print("Saved resized checkpoint to:", OUT)

if __name__ == "__main__":
    main()
