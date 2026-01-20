import os
import sys
import time
import math
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# --- Silence TRANSFORMERS warnings & configure cache dirs ---
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import concatenate_datasets, load_from_disk
from transformers import AutoTokenizer
from tqdm.auto import tqdm


# ==========================================================
# CONFIGURATION
# ==========================================================

@dataclass
class ValidationConfig:
    max_len: int = 512
    batch_size: int = 8
    num_workers: int = 2
    tokenizer_path: str = "artifacts/zia_tokenizer_60k"

@dataclass
class ModelConfig:
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 512
    vocab_size: int = 60004


# ==========================================================
# MODEL DEFINITION (ZIA-compatible placeholder)
# ==========================================================

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer = nn.ModuleDict({
            f"h_{i}": nn.Linear(config.n_embd, config.n_embd) for i in range(config.n_layer)
        })

    def forward(self, input_ids, attention_mask, labels=None):
        # Mock forward for validation sanity check
        if labels is None:
            return torch.rand(input_ids.size(0), input_ids.size(1), self.config.vocab_size, device=input_ids.device)
        logits = torch.rand(input_ids.size(0), input_ids.size(1), self.config.vocab_size, device=input_ids.device)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=self.config.vocab_size - 1
        )
        return loss


# ==========================================================
# DATASET HELPERS
# ==========================================================

def _is_shard_dir(path):
    return os.path.isdir(path) and any(f.startswith("data-") for f in os.listdir(path))

def load_split_from_shards(split_dir: str) -> Optional[Dataset]:
    if not os.path.exists(split_dir):
        return None
    if os.path.exists(os.path.join(split_dir, "dataset_info.json")) or any(f.startswith("data-") for f in os.listdir(split_dir)):
        return load_from_disk(split_dir)
    shards = sorted([os.path.join(split_dir, d) for d in os.listdir(split_dir)])
    shard_dirs = [s for s in shards if _is_shard_dir(s)]
    if not shard_dirs:
        print(f"[!] No dataset or shards found in: {split_dir}")
        return None
    ds_list = []
    for sd in shard_dirs:
        try:
            ds = load_from_disk(sd)
            ds_list.append(ds)
        except Exception as e:
            print(f"[!] Failed to load shard {sd}: {e}")
    if not ds_list:
        return None
    return ds_list[0] if len(ds_list) == 1 else concatenate_datasets(ds_list, axis=0)


TEXT_COLUMN_KEY = "text"

def tokenize_fn(batch, tokenizer):
    if TEXT_COLUMN_KEY not in batch:
        raise KeyError(f"Missing expected text column '{TEXT_COLUMN_KEY}' — available: {list(batch.keys())}")
    return tokenizer(batch[TEXT_COLUMN_KEY], truncation=True, max_length=512, padding="max_length")

def load_and_cache_dataset(name, path, tokenizer):
    print(f"[i] Loading dataset from {path}")
    ds = load_split_from_shards(path)
    if ds is None:
        raise FileNotFoundError(f"Could not load dataset: {path}")
    print(f"[i] Loaded {len(ds)} samples | Columns: {ds.column_names}")
    print("[🔤] Tokenizing dataset (cached if already done)...")
    ds = ds.map(tokenize_fn, batched=True, num_proc=2, fn_kwargs={"tokenizer": tokenizer})
    return ds


# ==========================================================
# COLLATE FUNCTION (fixed for CPU pinning)
# ==========================================================

class CollateWrapper:
    def __init__(self, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]

        # ✅ Create CPU tensors only (DataLoader handles pinning)
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        labels_tensor = input_ids_tensor.clone()

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels_tensor,
        }


# ==========================================================
# VALIDATION LOOP
# ==========================================================

def run_validation(model, loader, device):
    model.eval()
    total_loss, total_batches = 0.0, 0
    with torch.no_grad():
        print(f"    -> Running validation on {len(loader)} batches...")
        for batch in tqdm(loader, desc="Validation"):
            try:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                loss = model(**batch)
                total_loss += loss.item()
                total_batches += 1
            except Exception as e:
                print(f"[!] Skipped batch (error: {e})")
                continue
    if total_batches == 0:
        return {"loss": float("nan"), "perplexity": float("nan")}
    avg_loss = total_loss / total_batches
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return {"loss": avg_loss, "perplexity": ppl}


# ==========================================================
# MAIN
# ==========================================================

def main():
    if sys.platform.startswith("win"):
        torch.multiprocessing.set_start_method("spawn", force=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    ROOT = os.path.abspath(os.getcwd())
    ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")

    cfg = ValidationConfig()
    model_cfg = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpts = [
        os.path.join(ARTIFACTS_DIR, "zia_dense_base", "recovered_best.pt"),
        os.path.join(ARTIFACTS_DIR, "zia_dense_runs", "best_val", "checkpoint.pt"),
        os.path.join(ARTIFACTS_DIR, "zia_ift_v2_runs", "best_val", "checkpoint.pt"),
        os.path.join(ARTIFACTS_DIR, "zia_ift_v3_fix_stage_2", "best_val", "checkpoint.pt"),
        os.path.join(ARTIFACTS_DIR, "zia_ift_v3_runs", "best_val", "checkpoint.pt"),
    ]

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [🚀] Starting validation on {device}")

    # Tokenizer
    tok_path = os.path.join(ROOT, cfg.tokenizer_path)
    try:
        tok = AutoTokenizer.from_pretrained(tok_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print(f"[✓] Tokenizer loaded (vocab={len(tok)})")
    except Exception as e:
        print(f"[❌] Tokenizer load failed: {e}")
        return

    # Dataset
    try:
        ds_path = os.path.join(ROOT, "my_datasets", "processed", "arrow_cleaned_v1")
        ds = load_and_cache_dataset("arrow_cleaned_v1", ds_path, tok)
    except Exception as e:
        print(f"[❌] Dataset load/tokenize failed: {e}")
        return

    collate = CollateWrapper(tok, cfg.max_len)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate,
        drop_last=False,
    )

    print(f"[i] Validation DataLoader ready ({len(loader)} batches).")
    print("-" * 60)

    model = Model(model_cfg).to(device)
    results = {}

    for ckpt in ckpts:
        print(f"\n[🔄] Processing checkpoint: {os.path.basename(ckpt)}")
        if not os.path.exists(ckpt):
            print(f"[!] Missing file: {ckpt}")
            results[ckpt] = {"loss": float("nan"), "perplexity": float("nan"), "status": "Not Found"}
            continue

        try:
            ckpt_obj = torch.load(ckpt, map_location=device)
            state_dict = ckpt_obj.get("model", ckpt_obj)
            model.load_state_dict(state_dict, strict=False)
            metrics = run_validation(model, loader, device)
            results[ckpt] = {**metrics, "status": "OK"}
            print(f"    [✓] Loss: {metrics['loss']:.4f} | PPL: {metrics['perplexity']:.2f}")
        except Exception as e:
            print(f"[❌] Failed: {e}")
            results[ckpt] = {"loss": float("nan"), "perplexity": float("nan"), "status": f"Error: {e}"}

    print("-" * 60)
    print("[✅] Final Validation Summary:")
    for path, r in results.items():
        rel = os.path.relpath(path, ROOT)
        print(f"-> {rel}: Loss={r['loss']:.4f}, PPL={r['perplexity']:.2f}, Status={r['status']}")
    print("-" * 60)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
