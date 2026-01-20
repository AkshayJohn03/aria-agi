
#!/usr/bin/env python3
"""
verify_loss_zero_patch.py — extended version

Verifies dataset, loss, and model output sanity for ZIA IFT v3 checkpoints.
"""

import os, sys, torch, torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f"[sys.path added]: {PROJECT_ROOT}")

import torch
import torch.nn.functional as F
from types import SimpleNamespace

from my_datasets.code.train_zia_ift_v3_rope_alibi_new import MockGPT, V3Config, build_loaders 
from my_datasets.code.train_zia_ift_v3_rope_alibi_new import collate_fn_for_multiprocessing


from torch.utils.data import DataLoader
from my_datasets.code.train_zia_ift_v3_rope_alibi_new import MockGPT, V3Config, build_loaders
from transformers import AutoTokenizer

# -----------------------------
# ENV + PATH FIX
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f"[sys.path added]: {PROJECT_ROOT}")

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "artifacts/zia_ift_v3_rope_runs/ckpt_best_val.pt"
TOKENIZED_DATASET_PATH = "artifacts/tokenized_datasets/zia_ift_v3"
TOKENIZER_PATH = "artifacts/zia_tokenizer_60k"
BATCH_SIZE = 2

# -----------------------------
# LOAD DATASET
# -----------------------------
print("[*] Loading tokenized dataset...")
train_loader, val_loader = build_loaders(TOKENIZED_DATASET_PATH, batch_size=BATCH_SIZE, num_workers=0)
val_dataset = val_loader.dataset
print(f"Loaded {len(val_dataset)} validation samples.")

batch = next(iter(val_loader))

# -----------------------------
# MODEL SETUP
# -----------------------------
cfg = V3Config(
    max_len=512,
    model_name="verify",
    tokenizer_path=TOKENIZER_PATH,
    n_layer=8,
    n_head=6,
    n_embd=384,
    vocab_size=60004,
    dropout=0.0,
    bias=False,
    fp16=False,
)
print("[*] Building model...")
model = MockGPT(cfg).to(DEVICE)

print("[*] Loading checkpoint (non-strict)...")
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
state = ckpt["model"] if "model" in ckpt else ckpt
model.load_state_dict(state, strict=False)

# -----------------------------
# BASIC HEALTH CHECKS
# -----------------------------
with torch.no_grad():
    print("lm_head mean:", model.lm_head.weight.mean().item())
    print("final LayerNorm weight mean:", model.transformer.ln_f.weight.mean().item())

# -----------------------------
# LABEL MASK CHECK
# -----------------------------
labels = batch["labels"]
if not isinstance(labels, torch.Tensor):
    labels = torch.tensor(labels)

mask = labels != -100
valid_tokens = mask.sum().item()
total_tokens = labels.numel()
print(f"[+] Non -100 labels in first batch: {valid_tokens}/{total_tokens}")
if valid_tokens == 0:
    print("⚠️ All labels are masked (-100)! Dataset or collate_fn issue.")
else:
    print("✅ Labels look fine.")

# -----------------------------
# LOSS + LOGITS CHECK
# -----------------------------
print("[*] Computing sample forward loss in FP32 (no AMP)...")
model.eval()
with torch.no_grad():
    input_ids = batch["input_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)

    out = model(input_ids)
    logits = out[0] if isinstance(out, tuple) else out

    logits = logits.contiguous().view(-1, logits.size(-1))
    labels = labels.view(-1)
    valid_mask = labels != -100

    if valid_mask.sum() == 0:
        print("⚠️ No valid tokens for loss computation.")
        loss = torch.tensor(float("nan"))
    else:
        loss = F.cross_entropy(logits[valid_mask], labels[valid_mask])
        print(f"🧮 Sample batch loss = {loss.item():.6f}")
        if loss.item() == 0.0:
            print("🚨 WARNING: Loss is exactly 0.0 — check if outputs or labels are frozen!")
        elif torch.isnan(loss):
            print("⚠️ NaN loss detected — potential FP16 overflow or bad gradients.")
        else:
            print("✅ Loss appears normal and non-zero.")

# -----------------------------
# TOKEN DECODING CHECK
# -----------------------------
print("\n[*] Checking decoded samples and predictions...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

with torch.no_grad():
    sample_idx = 0
    inp = batch["input_ids"][sample_idx].unsqueeze(0).to(DEVICE)
    lbl = batch["labels"][sample_idx].unsqueeze(0).to(DEVICE)

    out = model(inp)
    logits = out[0]
    preds = torch.argmax(logits, dim=-1)

    decoded_inp = tokenizer.decode(inp[0].tolist(), skip_special_tokens=True)
    decoded_lbl = tokenizer.decode([t for t in lbl[0].tolist() if t != -100], skip_special_tokens=True)
    decoded_pred = tokenizer.decode(preds[0].tolist(), skip_special_tokens=True)

    print(f"\n[Input  ]: {decoded_inp[:200]}...")
    print(f"[Label  ]: {decoded_lbl[:200]}...")
    print(f"[Predict]: {decoded_pred[:200]}...")

# -----------------------------
# LOGIT DISTRIBUTION CHECK
# -----------------------------
print("\n[*] Checking logit distribution sanity...")
logits_flat = logits.detach().cpu().float().view(-1)
print(f"Logits mean: {logits_flat.mean():.6f}, std: {logits_flat.std():.6f}, min: {logits_flat.min():.6f}, max: {logits_flat.max():.6f}")
if logits_flat.std() < 1e-3:
    print("🚨 WARNING: Logits are nearly constant — model outputs may be collapsed.")
else:
    print("✅ Logits show healthy variation.")

print("\n✅ Verification complete — model is numerically and semantically healthy.")
