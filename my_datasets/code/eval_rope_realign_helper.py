#!/usr/bin/env python3
# eval_rope_auto_probe.py
# Single-file evaluator: defines TinyGPT_RoPE, probes checkpoints for correct head geometry,
# runs a light forward check with RoPE, and writes a summary.

import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_from_disk

# -----------------------
# Config - edit if needed
# -----------------------
CHECKPOINT_DIR = "artifacts/zia_ift_v4_longctx/checkpoints"
TOKENIZER_PATH = "artifacts/zia_tokenizer_60k"   # local tokenizer dir
VAL_SAMPLE_PATH = "artifacts/processed/wiki_tokenized/wiki_chunk_0060.arrow"  # used only to get pad id and a tiny sample
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = False      # small eval -> float32 by default; set True to test with autocast
MAX_TRY_HEADS = [24, 12, 16, 8, 6, 4, 3, 2, 1]   # ordered preference (common choices for d_model=384)
DUMMY_BATCH = 1
DUMMY_SEQ_LEN = 8  # short sequence for quick forward-check
MAX_EVAL_CKPTS = None  # set to int to limit how many ckpts to test (None => all)
SUMMARY_OUT = os.path.join(CHECKPOINT_DIR, "rope_eval_auto_summary.txt")

# -------------------------
# RoPE helpers (lightweight)
# -------------------------
class RotaryEmbedding:
    def __init__(self, head_dim, base=10000, max_seq_len=2048):
        self.head_dim = head_dim
        self.base = base
        self.register_cache(max_seq_len)

    def register_cache(self, max_seq_len):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, head_dim]
        self._cos = torch.cos(emb)
        self._sin = torch.sin(emb)
        self.max_seq_len_cached = max_seq_len

    def get_cos_sin(self, seq_len, device, dtype):
        if seq_len > self.max_seq_len_cached:
            self.register_cache(seq_len * 2)
        return self._cos[:seq_len].to(device=device, dtype=dtype), self._sin[:seq_len].to(device=device, dtype=dtype)

def apply_rope(x, cos, sin):
    # x: (B, T, H, head_dim)
    # cos,sin: (T, head_dim)
    B, T, H, D = x.shape
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # cos/sin shape expand -> (T, 1, head_dim/???) we will unsqueeze appropriately
    cos = cos.unsqueeze(1).unsqueeze(0)  # (1, T, 1, head_dim)
    sin = sin.unsqueeze(1).unsqueeze(0)
    # x1,x2 shape: (B,T,H,D/2)
    # reconstruct rotated vector
    x_rot = torch.cat([x1 * cos - x2 * sin,
                       x1 * sin + x2 * cos], dim=-1)
    return x_rot

# -------------------------
# Model definition (self-contained)
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, mlp_ratio, dropout):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout, max_seq_len):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        # single linear for qkv: 3*d_model x d_model (common implementation)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mlp_ratio, dropout)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h)  # (B, T, 3*D)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)  # (B,T,3,H,hd)
        q = qkv[:, :, 0]  # (B,T,H,hd)
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        cos, sin = self.rope.get_cos_sin(T, device=x.device, dtype=x.dtype)  # cos: (T, head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # compute attention
        q_ = q.permute(0,2,1,3)  # (B,H,T,hd)
        k_ = k.permute(0,2,3,1)  # (B,H,hd,T)
        attn_scores = torch.matmul(q_, k_) / math.sqrt(self.head_dim)
        # causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        v_ = v.permute(0,2,1,3)  # (B,H,T,hd)
        out = torch.matmul(attn, v_)
        out = out.permute(0,2,1,3).contiguous().view(B, T, D)
        out = self.out_proj(out)
        x = x + out
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT_RoPE(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, mlp_ratio, max_len, dropout, pad_token_id=0):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, mlp_ratio, dropout, max_seq_len=max_len) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.max_len = max_len
        self.pad_token_id = pad_token_id

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        if T > self.max_len:
            input_ids = input_ids[:, -self.max_len:]
            T = input_ids.shape[1]
        x = self.tok(input_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                                   labels[:, 1:T].reshape(-1),
                                   ignore_index=self.pad_token_id)
        return logits, loss

# -------------------------
# Helpers
# -------------------------
def try_infer_heads_from_ckpt(ck, d_model):
    # If ck contains metadata, try to use it first.
    # Many checkpoints store "n_heads" or "head_dim" or "d_model" or "context_len"
    if isinstance(ck, dict):
        for key in ("n_heads", "num_heads", "heads"):
            if key in ck:
                val = ck[key]
                try:
                    val = int(val)
                    if d_model % val == 0:
                        return [val]
                except Exception:
                    pass
        # head_dim possibility:
        for key in ("head_dim","headsize","head_size"):
            if key in ck:
                try:
                    hd = int(ck[key])
                    if hd != 0 and (d_model % hd == 0):
                        return [d_model // hd]
                except Exception:
                    pass
    # fallback: return candidate list filtered to divisors of d_model
    candidates = [h for h in MAX_TRY_HEADS if d_model % h == 0]
    # ensure at least something
    if not candidates:
        candidates = [h for h in range(1, min(64, d_model)+1) if d_model % h == 0]
    return candidates

def load_tokenizer_pad_id(tokenizer_path):
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if tok.pad_token is None:
            tok.add_special_tokens({"pad_token":"<pad>"})
        return tok.pad_token_id, len(tok)
    except Exception as e:
        print("[!] Could not load tokenizer locally:", e)
        # fallback pad id
        return 0, 60004

def tiny_val_sample(pad_id):
    # create tiny ids and labels for forward pass test
    ids = torch.full((DUMMY_BATCH, DUMMY_SEQ_LEN), pad_id, dtype=torch.long)
    # fill first tokens with increasing ints to avoid all pad
    for b in range(DUMMY_BATCH):
        for t in range(min(DUMMY_SEQ_LEN, 4)):
            ids[b,t] = (t+1) % 100 + 1
    labels = ids.clone()
    return ids, labels

# -------------------------
# Main evaluator
# -------------------------
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_auto_step*.pt")))
    if MAX_EVAL_CKPTS:
        ckpts = ckpts[:MAX_EVAL_CKPTS]
    print(f"[i] Found {len(ckpts)} checkpoints to probe in {CHECKPOINT_DIR}")

    pad_id, vocab = load_tokenizer_pad_id(TOKENIZER_PATH)
    print(f"[i] Tokenizer pad_id={pad_id} vocab={vocab}")

    results = []

    ids_sample, labels_sample = tiny_val_sample(pad_id)
    ids_sample = ids_sample.to(DEVICE)
    labels_sample = labels_sample.to(DEVICE)

    for ckpt_path in tqdm(ckpts):
        base = os.path.basename(ckpt_path)
        try:
            ck = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            print(f"[✗] Failed to load {base}: {e}")
            continue

        # infer d_model & context_len if checkpoint stores them
        # fallback defaults:
        d_model = ck.get("d_model", None) or ck.get("config", {}).get("d_model", None) if isinstance(ck, dict) else None
        ctx_len = ck.get("context_len", None) if isinstance(ck, dict) else None
        # fallback sensible defaults
        if d_model is None:
            # try inspect weights: qkv weight usually present at blocks.0.qkv.weight or blocks.0.attn.in_proj_weight
            found = False
            for k in ck.keys() if isinstance(ck, dict) else []:
                if "qkv" in k or "attn.in_proj_weight" in k or ("blocks.0" in k and "qkv" in k):
                    w = ck[k]
                    if isinstance(w, torch.Tensor):
                        # shape (3*d_model, d_model)
                        if w.ndim == 2:
                            d_model = w.shape[1]
                            found = True
                            break
            if not found:
                d_model = 384
        if ctx_len is None:
            ctx_len = ck.get("max_len", ck.get("context_len", ck.get("ctx_len", 4096)))

        # determine candidate n_heads to try
        candidates = try_infer_heads_from_ckpt(ck, d_model)
        success = False
        for try_heads in candidates:
            try:
                model = TinyGPT_RoPE(vocab_size=vocab,
                                     d_model=d_model,
                                     n_layers=int(ck.get("n_layers", 8)),
                                     n_heads=int(try_heads),
                                     mlp_ratio=int(ck.get("mlp_ratio", 4)),
                                     max_len=int(ctx_len),
                                     dropout=float(ck.get("dropout", 0.1)),
                                     pad_token_id=pad_id
                                     ).to(DEVICE)
                # load model state (some checkpoints store dict under "model")
                model_state = ck.get("model", ck)
                # try loading; strict=False to allow missing optimizer keys etc
                model.load_state_dict(model_state, strict=False)
                model.eval()
                # little forward pass to validate shapes (use float32 unless FP16 requested)
                with torch.cuda.amp.autocast(enabled=FP16, device_type="cuda" if DEVICE.startswith("cuda") else "cpu"):
                    logits, loss = model(ids_sample, labels_sample)
                # success
                val_loss = float(ck.get("best_val_metric", ck.get("val_loss", float("nan")))) if isinstance(ck, dict) else float("nan")
                results.append({
                    "ckpt": ckpt_path,
                    "d_model": d_model,
                    "n_heads": try_heads,
                    "ctx_len": ctx_len,
                    "val_loss": val_loss,
                })
                print(f"[✓] {base}  — OK (d_model={d_model}, n_heads={try_heads}, ctx={ctx_len})")
                success = True
                # free memory
                del model
                torch.cuda.empty_cache()
                break
            except Exception as e:
                # mismatch likely — continue to next candidate
                # You can uncomment the next line to see exact exception for debugging:
                # print(f"[debug] {base} try_heads={try_heads} failed: {e}")
                try:
                    del model
                except Exception:
                    pass
                torch.cuda.empty_cache()
                continue

        if not success:
            print(f"[✗] {base} — No matching head configuration found (tried: {candidates})")

    # write summary
    with open(SUMMARY_OUT, "w") as f:
        if results:
            f.write("ckpt\tctx_len\td_model\tn_heads\tval_loss\n")
            for r in results:
                f.write(f"{os.path.basename(r['ckpt'])}\t{r['ctx_len']}\t{r['d_model']}\t{r['n_heads']}\t{r['val_loss']}\n")
            print(f"[i] Summary written to {SUMMARY_OUT}")
        else:
            f.write("No valid checkpoints found.\n")
            print("[x] No valid checkpoints found. See printed failures above.")

if __name__ == "__main__":
    main()
