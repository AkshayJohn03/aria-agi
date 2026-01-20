import os
import math
import time
import json
import shutil
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# --- Silence TRANSFORMERS_CACHE warning & set cache dirs early ---
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.nn import functional as F
from tqdm import tqdm

from datasets import load_from_disk
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, __version__ as transformers_version

# Optional: bitsandbytes
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
    BNB_VERSION = getattr(bnb, "__version__", "unknown")
except Exception:
    BNB_AVAILABLE = False
    BNB_VERSION = "not available"

# Some perf-friendly toggles (harmless if unsupported)
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.matmul.allow_tf32 = True  # Ampere+ only
except Exception:
    pass
try:
    torch.backends.cudnn.benchmark = True  # mostly helps CNNs; harmless here
except Exception:
    pass


# -------------------------
# Model: Tiny GPT-like decoder (dense)
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int, dropout: float):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mlp_ratio, dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        h = self.ln1(x)
        # causal mask
        B, T, _ = h.size()
        causal = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(h, h, h, attn_mask=causal, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + out
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, mlp_ratio: int, max_len: int, dropout: float):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.vocab_size = vocab_size
        # Tie weights
        self.head.weight = self.tok.weight

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        if T > self.max_len:
            input_ids = input_ids[:, -self.max_len:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_len:]
            T = input_ids.size(1)

        pos_ids = torch.arange(0, T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok(input_ids) + self.pos(pos_ids)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# -------------------------
# Data utilities
# -------------------------
def join_messages(messages: List[Dict[str, str]], eos_token: str) -> str:
    parts = []
    for m in messages:
        c = (m.get("content") or "").strip()
        if c:
            parts.append(c)
    text = ("\n").join(parts)
    if eos_token:
        text += eos_token
    return text


@dataclass
class Collator:
    tokenizer: AutoTokenizer
    max_len: int
    eos_token: str

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        texts = [join_messages(ex["messages"], self.eos_token) for ex in batch]
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

@dataclass
class IterableWrapperDataset(IterableDataset):
    """
    A simple wrapper to allow a regular dataset to be used as an IterableDataset.
    This is necessary for the DataLoader to properly handle persistent workers
    when resuming training from an arbitrary batch.
    """
    dataset: Any
    start_index: int = 0

    def __iter__(self):
        # This creates an iterator starting from the specified index.
        # This is the key to resuming at an arbitrary batch.
        for i in range(self.start_index, len(self.dataset)):
            yield self.dataset[i]

    def __len__(self):
        return len(self.dataset) - self.start_index

# -------------------------
# Checkpoint helpers
# -------------------------
def save_checkpoint(model, optimizer, scheduler, scaler, step, out_dir, keep_last=2, tag=None):
    tag = tag or f"step_{step:07d}"
    path = os.path.join(out_dir, tag)
    os.makedirs(path, exist_ok=True)
    
    # Use a temporary file and rename to ensure atomicity
    tmp_path = os.path.join(path, "checkpoint.pt.tmp")
    final_path = os.path.join(path, "checkpoint.pt")
    
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "tag": tag,
        },
        tmp_path,
    )
    # Rename the temporary file to the final name
    os.rename(tmp_path, final_path)
    
    # Keep only last N checkpoints (by mtime)
    ckpts = []
    for name in os.listdir(out_dir):
        p = os.path.join(out_dir, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "checkpoint.pt")):
            ckpts.append((p, os.path.getmtime(p)))
    ckpts.sort(key=lambda x: x[1], reverse=True)
    for p, _ in ckpts[keep_last:]:
        shutil.rmtree(p, ignore_errors=True)

def find_last_checkpoint(out_dir):
    """Finds the most recent checkpoint folder in the output directory."""
    if not os.path.exists(out_dir):
        return None
    
    ckpts = []
    for name in os.listdir(out_dir):
        p = os.path.join(out_dir, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "checkpoint.pt")):
            ckpts.append((p, os.path.getmtime(p)))
    
    if not ckpts:
        return None
    
    ckpts.sort(key=lambda x: x[1], reverse=True)
    return os.path.join(ckpts[0][0], "checkpoint.pt")


# -------------------------
# Validation
# -------------------------
@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=None):
    model.eval()
    losses = []
    pbar = tqdm(val_loader, desc="🔎 Validating", leave=False, dynamic_ncols=True)
    for i, (input_ids, attn_mask) in enumerate(pbar, 1):
        input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        logits = model(input_ids, attention_mask=attn_mask)
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0)
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        if max_batches and i >= max_batches:
            break
    model.train()
    return float(sum(losses) / max(1, len(losses)))


# -------------------------
# Training entry
# -------------------------
def main():
    # --- Hardcoded Best Config for GTX 1050 Ti ---
    # Paths
    dataset_path = "my_datasets/processed/arrow_dataset"
    tokenizer_path = "artifacts/zia_tokenizer_60k"
    out_dir = "artifacts/zia_dense_runs"

    # Model size (Tiny model)
    model_size = "tiny"
    d_model = 384
    n_layers = 8
    n_heads = 6
    mlp_ratio = 4
    max_len = 256
    dropout = 0.1

    # Training parameters
    batch_size = 8
    accum_steps = 16
    epochs = 1
    lr = 3e-4
    warmup_steps = 200
    weight_decay = 0.1
    eval_every = 100
    save_every = 10
    keep_last = 2
    max_val_batches = 200
    
    # DataLoader (Windows-safe)
    num_workers = 8
    prefetch_factor = 2

    # Misc
    use_compile = False # Disabled for GTX 1050 Ti (capability < 7.0)

    # --- Print environment & lib versions ---
    print("[i] torch:", torch.__version__)
    print("[i] transformers:", transformers_version)
    print("[i] CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[i] CUDA device:", torch.cuda.get_device_name(0))
        print("[i] CUDA capability:", torch.cuda.get_device_capability(0))
    print("[i] bitsandbytes available:", BNB_AVAILABLE, "| version:", BNB_VERSION)
    print("[i] HF_HOME:", os.environ.get("HF_HOME"))
    print("[i] HF_DATASETS_CACHE:", os.environ.get("HF_DATASETS_CACHE"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load dataset (Arrow) ---
    print("[i] Loading dataset from:", dataset_path)
    t0 = time.time()
    ds = load_from_disk(dataset_path)
    train_ds_raw, val_ds = ds["train"], ds["test"]
    print(f"[i] Train size: {len(train_ds_raw)} | Val size: {len(val_ds)} | load_time={time.time()-t0:.2f}s")

    # --- Tokenizer ---
    print("[i] Loading tokenizer from:", tokenizer_path)
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    eos_token = tok.eos_token or "</s>"
    vocab_size = len(tok)
    print(f"[i] Tokenizer vocab_size: {vocab_size} | pad_id={tok.pad_token_id} | eos_id={tok.eos_token_id}")

    # --- Collator & DataLoaders ---
    collate = Collator(tokenizer=tok, max_len=max_len, eos_token=eos_token)
    
    start_step = 0
    start_epoch = 1
    
    # --- Build model ---
    print("[i] Building dense Transformer model...")
    t0 = time.time()
    model = TinyGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        max_len=max_len,
        dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[i] Model Parameters: {n_params:,} | build_time={time.time()-t0:.2f}s")

    # Optional compile with capability check (Pascal 6.1 -> disable)
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[i] torch.compile: enabled")
        except Exception as e:
            print(f"[!] torch.compile failed: {e}")
            print("[i] Falling back to eager mode.")
    else:
        print("[i] torch.compile: disabled")

    # --- Optimizer (AdamW) ---
    fused = getattr(torch.optim.AdamW, "fused", False)
    if fused and torch.cuda.is_available():
        print("[i] Using fused AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
    else:
        print("[i] Using standard AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Checkpoint loading logic ---
    last_checkpoint_path = find_last_checkpoint(out_dir)
    if last_checkpoint_path:
        print(f"[i] Found checkpoint at: {last_checkpoint_path}. Attempting to resume training...")
        try:
            checkpoint = torch.load(last_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            
            start_step = checkpoint["step"]
            steps_per_epoch = (len(train_ds_raw) // batch_size // accum_steps)
            start_epoch = (start_step // steps_per_epoch) + 1
            start_batch_index_within_epoch = ((start_step * accum_steps) % (len(train_ds_raw) // batch_size)) * batch_size
            
            print(f"[i] Successfully resumed from step {start_step} and epoch {start_epoch} (batch index {start_batch_index_within_epoch})")

            total_steps = (len(train_ds_raw) // batch_size // accum_steps) * epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
            if "scheduler" in checkpoint and checkpoint["scheduler"]:
                scheduler.load_state_dict(checkpoint["scheduler"])
            
        except (KeyError, RuntimeError) as e:
            print(f"[!] Checkpoint loading failed: {e}")
            print("[!] Checkpoint file may be corrupted. Starting new training run.")
            start_step = 0
            start_epoch = 1
            start_batch_index_within_epoch = 0
            total_steps = (len(train_ds_raw) // batch_size // accum_steps) * epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    else:
        print("[i] No checkpoint found. Starting new training run.")
        start_step = 0
        start_epoch = 1
        start_batch_index_within_epoch = 0
        total_steps = (len(train_ds_raw) // batch_size // accum_steps) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
    # --- Rebuild DataLoaders based on starting state ---
    pin = torch.cuda.is_available()
    persistent = num_workers > 0
    
    train_ds = IterableWrapperDataset(train_ds_raw, start_index=start_batch_index_within_epoch)
    
    print(f"[i] Building DataLoaders | num_workers={num_workers} | pin_memory={pin} | persistent_workers={persistent}")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
        persistent_workers=False,
        collate_fn=collate,
        drop_last=False,
    )

    scaler = None
    os.makedirs(out_dir, exist_ok=True)
    best_val = float("inf")
    accum_loss = 0.0
    global_step = start_step

    print(f"\n🚀 Starting training from epoch {start_epoch} for a total of {epochs} epoch(s)")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        
        remaining_batches = len(train_ds_raw) // batch_size - start_batch_index_within_epoch

        pbar = tqdm(train_loader, total=remaining_batches, initial=0, desc=f"🧠 Epoch {epoch}/{epochs}", dynamic_ncols=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        for step_in_epoch, (input_ids, attn_mask) in enumerate(pbar, 1):
            
            current_batch_index = start_batch_index_within_epoch + step_in_epoch
            
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)

            logits = model(input_ids, attention_mask=attn_mask)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0,
            ) / accum_steps

            loss.backward()

            accum_loss += loss.item()

            if current_batch_index % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                avg_loss = accum_loss
                accum_loss = 0.0
                pbar.set_postfix(step=global_step, lr=f"{scheduler.get_last_lr()[0]:.2e}", loss=f"{avg_loss:.4f}")

                # Save every N steps
                if global_step % save_every == 0:
                    tag = f"step_{global_step:07d}"
                    save_checkpoint(model, optimizer, scheduler, scaler, global_step, out_dir, keep_last=keep_last, tag=tag)
                    print(f"\n[💾] Saved checkpoint @ {tag}")

                # Validate every M steps
                if global_step % eval_every == 0:
                    val_loss = evaluate(model, val_loader, device, max_batches=max_val_batches)
                    print(f"\n[✓] Step {global_step} | Val loss: {val_loss:.4f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint(model, optimizer, scheduler, scaler, global_step, out_dir, keep_last=keep_last, tag="best_val")
                        print(f"[🏆] New best val loss {best_val:.4f} -> saved as best_val")

        # After the first resumed epoch, reset the starting batch for subsequent epochs
        start_batch_index_within_epoch = 0
        train_ds = IterableWrapperDataset(train_ds_raw, start_index=0)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=collate,
            drop_last=True,
        )

    # Final validation at the end of training
    val_loss = evaluate(model, val_loader, device, max_batches=max_val_batches)
    print(f"\n[✓] End of Training | Final Val loss: {val_loss:.4f}")
    if val_loss < best_val:
        best_val = val_loss
        save_checkpoint(model, optimizer, scheduler, scaler, global_step, out_dir, keep_last=keep_last, tag="best_val")
        print(f"[🏆] New best val loss {best_val:.4f} -> saved as best_val")

    # Optional memory stats
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"[i] CUDA memory_allocated: {mem:.1f} MiB")

    print("✅ Training done.")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    try:
        if os.name == "nt":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()