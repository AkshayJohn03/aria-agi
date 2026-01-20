#!/usr/bin/env python3
# inspect_checkpoint.py
import os, glob, torch
from collections import OrderedDict

CHK_DIR = "artifacts/zia_ift_v4_longctx/checkpoints"

def inspect_ckpt(path):
    print("====", path)
    ck = torch.load(path, map_location="cpu")
    if isinstance(ck, dict):
        keys = list(ck.keys())
        meta_keys = [k for k in keys if k not in ("model","optim","scaler","step")]
        print("Top-level keys:", keys)
        for k in meta_keys:
            print(" meta:", k, type(ck[k]), (ck[k] if isinstance(ck[k], (int,float,str)) else "len="+str(len(ck[k]) if hasattr(ck[k], '__len__') else '?')))
        model_state = ck.get("model", ck)
    else:
        model_state = ck
    if isinstance(model_state, dict):
        print("Number of tensors:", len(model_state))
        # Print a compact list of important shapes
        interesting = ["tok.weight","tok.embedding.weight","pos.weight","head.weight","blocks.0.qkv.weight","blocks.0.attn.in_proj_weight","blocks.0.attn.out_proj.weight","blocks.0.ff.net.0.weight"]
        for k in sorted(model_state.keys())[:200]:
            v = model_state[k]
            if hasattr(v, "shape"):
                print(f" {k:60} {tuple(v.shape)}")
        # highlight a few important shapes if present
        for name in interesting:
            if name in model_state:
                print(">> contains", name, "shape", tuple(model_state[name].shape))
    else:
        print("Checkpoint model is not a dict, type:", type(model_state))
    print()

def main():
    ckpts = sorted(glob.glob(os.path.join(CHK_DIR, "checkpoint*.pt")))
    if not ckpts:
        print("No checkpoints found in", CHK_DIR)
        return
    for ck in ckpts:
        try:
            inspect_ckpt(ck)
        except Exception as e:
            print("Failed reading", ck, e)

if __name__ == "__main__":
    main()
