#!/usr/bin/env python3
"""
probe_zia_full_diagnostic.py
Comprehensive diagnostic suite for ZIA checkpoints.

Outputs:
 - JSON report: <out_dir>/diag_report.json
 - CSVs: top_tokens.csv, nn_neighbors.csv, embedding_pca.csv
 - Human-readable summary printed to stdout

Notes:
 - Works offline; requires torch, transformers, numpy, sklearn (optional).
 - If sklearn not available, uses torch PCA fallback (slower / less features).
"""
import os
import json
import csv
import time
import math
import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np

try:
    from transformers import PreTrainedTokenizerFast
except Exception:
    PreTrainedTokenizerFast = None

# Try sklearn for PCA / cosine if available
USE_SKLEARN = True
try:
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    USE_SKLEARN = False

# ---------------- Config / Defaults ----------------
DEFAULT_CKPT = "artifacts/zia_ift_v5/checkpoint_autosave_step6454.pt"
DEFAULT_TOKENIZER = "artifacts/zia_tokenizer_60k_clean"
DEFAULT_OUT = "artifacts/zia_ift_v5/diagnostics"
MAX_TOP_TOKENS = 200
NN_TOPK = 10
EMBED_SAMPLE = 6000   # sample tokens for PCA if vocab large
# ---------------------------------------------------

def load_checkpoint(ckpt_path):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ck = torch.load(str(ckpt_path), map_location="cpu")
    state = ck.get("model", ck)
    return state, ck

def build_model_shell(vocab_size, d_model=384, n_layers=8, n_heads=6, max_len=4096):
    # Minimal model shell matching your TinyGPT
    class TinyGPT(torch.nn.Module):
        def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len):
            super().__init__()
            self.tok = torch.nn.Embedding(vocab_size, d_model)
            self.pos = torch.nn.Embedding(max_len, d_model)
            self.blocks = torch.nn.ModuleList([
                torch.nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                    dropout=0.1, activation="gelu", batch_first=True, norm_first=True
                ) for _ in range(n_layers)
            ])
            self.ln_f = torch.nn.LayerNorm(d_model)
            self.head = torch.nn.Linear(d_model, vocab_size, bias=False)
        def forward(self, x):
            B,T = x.shape
            pos = torch.arange(0, T, device=x.device).unsqueeze(0)
            h = self.tok(x) + self.pos(pos)
            for b in self.blocks: h = b(h)
            return self.head(self.ln_f(h))
    return TinyGPT(vocab_size, d_model, n_layers, n_heads, max_len)

def safe_get_embedding_matrix(state_dict):
    # tries common keys
    for key in ("tok.weight", "tok_weight", "embedding.weight", "emb.weight", "model.tok.weight"):
        if key in state_dict:
            return state_dict[key]
    # fallback: look for .*tok.*weight or .*emb.*weight
    for k in state_dict.keys():
        kl = k.lower()
        if "tok" in kl and "weight" in kl:
            return state_dict[k]
        if ("emb" in kl or "embedding" in kl) and "weight" in kl:
            return state_dict[k]
    raise KeyError("Cannot find embedding weight in checkpoint state dict. Keys: " + ", ".join(list(state_dict.keys())[:30]))

def token_distribution_stats(emb_weight, tokenizer, top_n=MAX_TOP_TOKENS):
    """
    Compute:
     - vocab_size
     - token entropy (from softmax of norms as proxy)
     - top token dominance (freq of top n)
    """
    # Use embedding norms as a lightweight proxy for token importance
    norms = emb_weight.norm(dim=1).cpu().numpy()
    # softmax over norms (temperature smoothing)
    t = 1.0
    p = np.exp(norms / t)
    p = p / p.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    # top tokens by p
    top_idx = np.argsort(-p)[:top_n]
    top_tokens = []
    for i in top_idx:
        tok = tokenizer.decode([int(i)]) if tokenizer else str(int(i))
        top_tokens.append((int(i), tok, float(p[i])))
    return {"vocab_size": emb_weight.shape[0], "entropy": float(entropy), "top_tokens": top_tokens, "prob_dist": p}

def embedding_pca_analysis(emb_weight, out_dir, sample_limit=EMBED_SAMPLE):
    E = emb_weight.detach().cpu().numpy()
    vocab_size, d = E.shape
    if vocab_size > sample_limit:
        # sample uniformly for PCA to save time
        idx = np.linspace(0, vocab_size-1, sample_limit, dtype=int)
        E_s = E[idx]
    else:
        idx = np.arange(vocab_size)
        E_s = E
    explained = None
    components = None
    try:
        if USE_SKLEARN:
            pca = PCA(n_components=min(50, E_s.shape[1]))
            pca.fit(E_s)
            explained = pca.explained_variance_ratio_.tolist()
            components = pca.components_.tolist()
            # Save top-3 PC variance
            var3 = sum(explained[:3])
        else:
            # fallback SVD using numpy
            U, S, Vt = np.linalg.svd(E_s - E_s.mean(axis=0), full_matrices=False)
            explained = (S**2 / (S**2).sum()).tolist()
            components = Vt[:50].tolist()
            var3 = sum(explained[:3])
    except Exception as e:
        return {"error": f"PCA failed: {e}"}
    # Save a small CSV for manual inspection (first 20 tokens projection onto PC1..PC3)
    proj = None
    try:
        if USE_SKLEARN:
            proj = pca.transform(E_s)[:, :3]
        else:
            proj = (E_s - E_s.mean(axis=0)).dot(np.array(components).T)[:, :3]
        proj_csv = Path(out_dir) / "embedding_pca.csv"
        with open(proj_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["token_index", "pc1", "pc2", "pc3"])
            for i, row in enumerate(proj[:200]):
                writer.writerow([int(idx[i]), float(row[0]), float(row[1]), float(row[2])])
    except Exception:
        proj_csv = None
    return {"explained_variance_ratio": explained[:20], "pc1_3_variance": float(sum(explained[:3])), "proj_csv": str(proj_csv) if proj_csv else None}

def nearest_neighbors(emb_weight, tokenizer, probes, topk=NN_TOPK):
    weight = emb_weight.detach().cpu().numpy()
    # normalize
    W = weight / (np.linalg.norm(weight, axis=1, keepdims=True) + 1e-12)
    results = []
    for probe in probes:
        # encode probe token (single token) - we accept strings or token ids
        if isinstance(probe, int):
            pid = probe
        else:
            ids = tokenizer.encode(probe, add_special_tokens=False)
            if not ids:
                pid = None
            else:
                pid = ids[0]
        if pid is None or pid >= W.shape[0]:
            results.append({"probe": probe, "error": "can't encode probe token"})
            continue
        vec = W[pid:pid+1]
        sims = (vec @ W.T).ravel()
        topk_idx = np.argsort(-sims)[:topk]
        topk_tokens = []
        for i in topk_idx:
            tok = tokenizer.decode([int(i)]).strip() if tokenizer else str(int(i))
            topk_tokens.append({"id": int(i), "token": tok, "score": float(sims[i])})
        results.append({"probe": probe, "probe_id": int(pid), "neighbors": topk_tokens})
    return results

def next_token_probe(model, tokenizer, prompts, device, topk=10):
    model = model.to(device)
    model.eval()
    results = []
    for prompt, expected in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            results.append({"prompt": prompt, "error": "encode-empty"})
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)[:, -1, :]
            probs = F.softmax(logits, dim=-1).cpu().numpy().ravel()
        top_idx = np.argsort(-probs)[:topk]
        top_tokens = [tokenizer.decode([int(i)]).strip() for i in top_idx]
        match = any(any(exp.lower() in tok.lower() for tok in top_tokens) for exp in expected)
        results.append({"prompt": prompt, "top_tokens": top_tokens[:topk], "expected": expected, "match": bool(match)})
    return results

def repetition_continuity_tests(model, tokenizer, device):
    """
    check:
     - token repetition when generating greedy vs top-k
     - short sequence next-token stability for 'abcde' like patterns
    """
    model = model.to(device)
    model.eval()
    checks = []
    # test sequences
    sequences = ["abcdefg", "123456789", "The quick brown fox jumps over the lazy dog", "User: Hello\nZia:"]
    for seq in sequences:
        ids = tokenizer.encode(seq, add_special_tokens=False)
        if not ids:
            checks.append({"seq": seq, "error": "encode-empty"})
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)[:, -1, :]
            topk = 10
            vals, idxs = torch.topk(F.softmax(logits, dim=-1), topk)
            top_tokens = [tokenizer.decode([int(i)]).strip() for i in idxs[0].cpu().numpy()]
        checks.append({"seq": seq, "top_tokens": top_tokens})
    return checks

def run_all(ckpt, tokenizer_dir, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    # load tokenizer
    if PreTrainedTokenizerFast is None:
        raise RuntimeError("transformers not installed in environment.")
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir, local_files_only=True)
    # load checkpoint
    state_dict, raw_ck = load_checkpoint(ckpt)
    # find embedding matrix
    emb = safe_get_embedding_matrix(state_dict)
    if isinstance(emb, torch.Tensor) is False:
        emb = torch.tensor(emb)
    emb = emb.float()
    # Stats 1: token dist
    print("[diag] computing token distribution stats...")
    tdist = token_distribution_stats(emb, tok, top_n=MAX_TOP_TOKENS)
    # save top tokens csv
    top_csv = Path(out_dir) / "top_tokens.csv"
    with open(top_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "token", "proxy_prob"])
        for r, (idx, tokstr, p) in enumerate(tdist["top_tokens"], start=1):
            w.writerow([r, idx, tokstr, p])
    # Stats 2: embedding PCA
    print("[diag] running embedding PCA analysis (sample)...")
    pca_res = embedding_pca_analysis(emb, out_dir, sample_limit=EMBED_SAMPLE)
    # Stats 3: nearest neighbors for a curated probe list
    print("[diag] computing semantic nearest neighbors for curated probe tokens...")
    probe_tokens = ["Paris", "France", "dog", "cat", "water", "ice", "computer", "python", "India", "NIFTY", "<|im_start|>", "<user>", "<assistant>"]
    nn_res = nearest_neighbors(emb, tok, probe_tokens, topk=NN_TOPK)
    # save nn csv
    nn_csv = Path(out_dir) / "nn_neighbors.csv"
    with open(nn_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["probe", "probe_id", "neighbor_id", "neighbor_token", "score"])
        for item in nn_res:
            if "neighbors" not in item: continue
            for nb in item["neighbors"]:
                w.writerow([item["probe"], item.get("probe_id", ""), nb["id"], nb["token"], nb["score"]])
    # Stats 4: grammar / next-token probes (expanded)
    print("[diag] running next-token probes...")
    grammar_prompts = [
        ("The cat sat on the", ["mat", "floor", "ground", "bed"]),
        ("One plus one equals", ["two", "2"]),
        ("Paris is the capital of", ["France"]),
        ("Water turns into", ["ice", "steam"]),
        ("The sun rises in the", ["east"]),
        ("My name", ["is"]),
        ("She went to the store to buy", ["food", "milk", "bread"]),
        ("In machine learning, overfitting means the model", ["memorizes", "overfits", "fails"]),
        ("To install python packages use", ["pip", "conda"]),
        ("NIFTY is an index of", ["stocks", "Indian", "India"])
    ]
    # build model shell and load weights for inference
    vocab_size = emb.shape[0]
    # infer d_model from embedding
    d_model = emb.shape[1]
    model_shell = build_model_shell(vocab_size, d_model=d_model)
    # load state into model shell safely
    sd = {}
    # fix common weight keys: map tok.weight / head.weight if present in state_dict
    for k,v in state_dict.items():
        sd[k] = v
    try:
        model_shell.load_state_dict(sd, strict=False)
    except Exception as e:
        # still continue; strict=False already used
        pass
    model_shell = model_shell.to(device)
    next_results = next_token_probe(model_shell, tok, grammar_prompts, device=device, topk=10)
    # Stats 5: repetition / continuity
    print("[diag] running repetition and continuity tests...")
    rep_res = repetition_continuity_tests(model_shell, tok, device=device)
    # Assemble report
    report = {
        "meta": {
            "ckpt": str(ckpt),
            "tokenizer": str(tokenizer_dir),
            "timestamp": time.time(),
            "device": str(device),
            "vocab_size_inferred": int(emb.shape[0]),
            "d_model_inferred": int(emb.shape[1])
        },
        "token_distribution": {
            "entropy": tdist["entropy"],
            "vocab_size": int(tdist["vocab_size"]),
            "top_tokens": [{"id":int(x[0]), "token":x[1], "proxy_prob":float(x[2])} for x in tdist["top_tokens"]]
        },
        "embedding_pca": pca_res,
        "nearest_neighbors": nn_res,
        "next_token_probes": next_results,
        "repetition_tests": rep_res
    }
    # save json
    out_json = Path(out_dir) / "diag_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    # Print human summary & thresholds
    print("\n" + "="*60)
    print("ZIA DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"Checkpoint: {ckpt}")
    print(f"Vocab size (emb): {report['meta']['vocab_size_inferred']}, d_model: {report['meta']['d_model_inferred']}")
    print(f"Token entropy (proxy): {report['token_distribution']['entropy']:.4f}")
    top0 = report['token_distribution']['top_tokens'][0]
    print(f"Top token: id={top0['id']} token={top0['token']!r} proxy_prob={top0['proxy_prob']:.4f}")
    # heuristic checks
    ent = report['token_distribution']['entropy']
    vocab = report['token_distribution']['vocab_size']
    top_prob = report['token_distribution']['top_tokens'][0]['proxy_prob']
    # heuristics
    verdicts = []
    if ent < 4.0:
        verdicts.append("LOW_ENTROPY: token distribution is very peaked (possible collapse).")
    if top_prob > 0.01:
        verdicts.append("TOP_TOKEN_DOMINANCE: top token has >1% proxy probability (monitor).")
    if report['embedding_pca'].get("error"):
        verdicts.append("PCA_ERROR")
    else:
        pc13 = report['embedding_pca'].get("pc1_3_variance", 0.0)
        print(f"PC1-3 variance (sample): {pc13:.4f}")
        if pc13 < 0.05:
            verdicts.append("EMBEDDING_FLAT: low variance explained by top PCs (embeddings look noisy).")
        if pc13 > 0.35:
            verdicts.append("EMBEDDING_STRONG_STRUCTURE: healthy top-PC variance.")
    # next-token probe pass rate
    passes = sum(1 for r in report["next_token_probes"] if r.get("match"))
    total = len(report["next_token_probes"])
    print(f"Next-token simple probe pass: {passes}/{total}")
    if passes == 0:
        verdicts.append("NEXT_TOKEN_FAIL: model fails basic next-token predictions.")
    # print nearest neighbor sanity for a few probes
    print("\nSample nearest neighbors (probe -> top neighbor tokens):")
    for item in report["nearest_neighbors"][:6]:
        probe = item.get("probe")
        nb = item.get("neighbors", [])[:5]
        nstr = ", ".join([f"{n['token']}({n['score']:.2f})" for n in nb])
        print(f"  {probe!r} -> {nstr}")
    # final verdict
    print("\nHeuristic verdicts:")
    if not verdicts:
        print("  OK: no immediate red flags from automated heuristics.")
    else:
        for v in verdicts:
            print("  -", v)
    print("\nSaved report:", out_json)
    print("Saved CSVs:", top_csv, nn_csv, report['embedding_pca'].get("proj_csv"))
    # actionable recommendations
    print("\nRecommendations (automatically generated):")
    if "NEXT_TOKEN_FAIL" in " ".join(verdicts) or passes/total < 0.4:
        print("  1) Stop heavy IFT. Increase text-completion (pretraining) data — add large unlabelled corpora (Wiki, C4, books) and run dense LM pretraining for several epochs.")
        print("  2) If pretraining not possible, do mixed training: 80% LM completion from wiki/c4 + 20% IFT chat data.")
        print("  3) Continue small LR fine-tune from the best checkpoint (e.g., checkpoint_autosave_step6454.pt) only after you confirm embeddings are structured (pc1-3 variance > ~0.08).")
    else:
        print("  Continue IFT but monitor next-token probes and PC variance.")
    # return report path
    return str(out_json)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run_all(args.ckpt, args.tokenizer, args.out_dir, args.device)
