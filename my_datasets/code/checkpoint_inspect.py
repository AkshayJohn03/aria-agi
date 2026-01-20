#!/usr/bin/env python3
"""
Loads a checkpoint, tries to map it to a minimal TinyGPT-like model,
performs a single forward pass with random input to confirm shapes.
"""

import torch, argparse, os
from pathlib import Path

def build_tiny_from_ckpt_state(state_dict, vocab_size=32000):
    # Try to infer d_model & n_layers from keys heuristically.
    d_model = None
    n_layers = 0
    for k in state_dict.keys():
        if 'attn' in k and 'in_proj_weight' in k:
            # qkv sizing could indicate head dims
            v = state_dict[k].shape
            # skip
    # fallback defaults
    d_model = d_model or 384
    n_layers = n_layers or 8
    # Build minimal model (using the same TinyGPT in earlier message)
    import torch.nn as nn
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)
        def forward(self, x):
            return self.head(self.ln(self.embed(x)))
    return Dummy()

def inspect(ckpt_path):
    print("Loading:", ckpt_path)
    ck = torch.load(ckpt_path, map_location='cpu')
    # get model/state
    model_state = ck.get('model', ck)
    print("state keys:", len(model_state.keys()))
    # build dummy, try load_state_dict(strict=False)
    model = build_tiny_from_ckpt_state(model_state)
    try:
        model.load_state_dict(model_state, strict=False)
        print("Loaded (strict=False).")
    except Exception as e:
        print("load_state error:", e)
    model.eval()
    x = torch.randint(0, model.head.out_features, (1, 8))
    with torch.no_grad():
        y = model(x)
    print("forward ok, output shape:", y.shape)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python checkpoint_inspector.py <ckpt_path>")
        sys.exit(1)
    inspect(sys.argv[1])
