#!/usr/bin/env python3
"""
Progressively tries candidate batch sizes and grad-acc to find a combination likely to fit VRAM.
This is a *probe* that runs a tiny forward/backward pass (no real data) to estimate memory.
Run on the machine you intend to use.
"""

import torch, gc, sys
from itertools import product
from math import ceil

def try_combo(batch, seq, d_model=512, n_layers=8, vocab=32000, grad_acc=1, device='cuda'):
    try:
        gc.collect()
        torch.cuda.empty_cache()
        # tiny model
        import torch.nn as nn
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab, d_model)
                self.ln = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, vocab)
            def forward(self,x):
                h = self.emb(x)
                h = self.ln(h)
                return self.head(h)
        m = Tiny().to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        x = torch.randint(0, vocab, (batch, seq), device=device)
        loss = None
        for _ in range(1):  # small iteration
            out = m(x)
            loss = out.view(-1).sum()
            loss = loss / grad_acc
            loss.backward()
            opt.step()
            opt.zero_grad()
        mem = torch.cuda.max_memory_allocated(device)
        print(f"OK batch={batch}, seq={seq}, grad_acc={grad_acc}, mem={mem}")
        del m, out, x, opt, loss
        gc.collect(); torch.cuda.empty_cache()
        return True, mem
    except Exception as e:
        print("OOM or error for", batch, seq, grad_acc, "->", e)
        return False, str(e)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    candidates = [(1,4096),(2,2048),(4,1024)]
    grad_candidates = [1,8,16,32,64,128]
    for b,s in candidates:
        for g in grad_candidates:
            ok, info = try_combo(b, s, grad_acc=g, device=device)
            if not ok:
                break
