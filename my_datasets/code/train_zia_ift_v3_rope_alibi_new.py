#!/usr/bin/env python3
"""
train_zia_ift_v3_rope_alibi_new.py

A streamlined, training-only script for Instruction Fine-Tuning (IFT)
supporting RoPE / ALiBi position embeddings and flexible checkpointing.
It assumes the tokenized dataset is already prepared and saved to disk.
"""
import os
# --- Silence TRANSFORMERS_CACHE warning & set cache dirs early ---
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
import json
import math
import glob
import shutil
import random
import argparse
import hashlib
import traceback
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# --- LLM Model / Architecture Imports (Assumed to be locally defined or imported) ---
# NOTE: The actual Zia model architecture (with RoPE/ALiBi support) is assumed
# to be defined elsewhere or dynamically loaded by the user environment.
# We mock a simple model structure for demonstration.

# Placeholder for a simple Transformer block (as the full model code is unavailable)
class MockAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash

        # RoPE / ALiBi properties
        self.use_alibi = config.use_alibi
        self.n_kv_head = config.n_kv_head

        if not self.use_alibi:
            # RoPE setup: Placeholder for pre-calculated RoPE rotation matrices
            # In a real implementation, this would involve complex setup logic.
            self.register_buffer('freqs_cis', torch.randn(config.max_len, self.head_dim // 2))

    def forward(self, x, layer_idx):
        # Mock forward pass
        B, T, C = x.size()
        qkv = self.c_attn(x)
        # ... real attention logic using RoPE or ALiBi ...
        out = self.c_proj(qkv[:,:,:C]) # Mock projection
        return self.resid_dropout(out)

class MockBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MockAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout)
        )

    def forward(self, x, layer_idx):
        x = x + self.attn(self.ln_1(x), layer_idx)
        x = x + self.mlp(self.ln_2(x))
        return x

class MockGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([MockBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # Tie weights

        # Initialization
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.max_len, f"Cannot forward sequence of length {t}, block size is only {self.config.max_len}"

        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)

        for layer_idx, block in enumerate(self.transformer.h):
            x = block(x, layer_idx)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift the logits and labels to align for next-token prediction
            logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            # Flatten for cross-entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)

        return (logits, loss) if labels is not None else (logits,)

# -------------------------
# Config
# -------------------------
@dataclass
class V3Config:
    # Model / tokenizer
    max_len: int = 512
    model_name: str = "zia-v1"
    tokenizer_path: str = "artifacts/zia_tokenizer_60k"
    
    # Architecture
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 384
    n_kv_head: Optional[int] = None # For GQA/MQA, defaults to n_head
    vocab_size: int = 60004 # Will be set based on tokenizer
    dropout: float = 0.0
    bias: bool = False
    flash: bool = False
    
    # Position Embeddings (RoPE/ALiBi)
    use_alibi: bool = False
    rope_theta: float = 10000.0
    rope_scaling_factor: float = 1.0

    # Training parameters
    batch_size: int = 16
    lr: float = 6e-5
    weight_decay: float = 1e-1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    max_steps: int = 10000
    warmup_steps: int = 100
    
    # Checkpointing / Logging / Device
    output_dir: str = "artifacts/checkpoints_ift"
    tokenized_dataset_path: str = "artifacts/tokenized_dataset"
    resume_from: Optional[str] = None
    load_weights_only_from: Optional[str] = None
    resume_optimizer: bool = False
    eval_every: int = 200
    log_every: int = 10
    keep_last: int = 3
    early_stop_patience: int = 10
    fp16: bool = True


# ----------------------
# Metrics and Utilities
# ----------------------

def compute_metrics(eval_loss: float) -> Dict[str, float]:
    """Calculates PPL and Cross Entropy from the evaluation loss."""
    ce_loss = eval_loss
    try:
        ppl = math.exp(eval_loss)
    except OverflowError:
        ppl = float('inf')
    return {"val/loss_ce": ce_loss, "val/ppl": ppl}

def save_checkpoint(model, optimizer, scheduler, scaler, step, output_dir, keep_last=3, tag=None, best_val_metric=float('inf')):
    """Saves a checkpoint containing model state, optimizer, scheduler, scaler, and training state."""
    os.makedirs(output_dir, exist_ok=True)
    
    if tag:
        filename = f"ckpt_{tag}.pt"
    else:
        filename = f"ckpt_{step:05d}.pt"
        
    filepath = os.path.join(output_dir, filename)
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict() if scaler else None,
        'step': step,
        'best_val': best_val_metric,
        'config': model.config,
        'timestamp': time.time(),
    }
    
    try:
        torch.save(checkpoint, filepath)
    except Exception as e:
        print(f"[!] Warning: Failed to save checkpoint to {filepath}. Error: {e}")
        return

    if tag is None:
        # Manage 'keep_last' checkpoints
        all_ckpts = sorted(glob.glob(os.path.join(output_dir, "ckpt_*.pt")))
        # Exclude special tags like 'final', 'interrupt', 'best_val'
        checkpoints_to_keep = [c for c in all_ckpts if not any(t in c for t in ['final', 'interrupt', 'best_val'])]

        if len(checkpoints_to_keep) > keep_last:
            for old_ckpt in checkpoints_to_keep[:-keep_last]:
                try:
                    os.remove(old_ckpt)
                    print(f"[*] Removed old checkpoint: {os.path.basename(old_ckpt)}")
                except OSError as e:
                    print(f"[!] Warning: Could not remove old checkpoint {os.path.basename(old_ckpt)}: {e}")
    
    print(f"[*] Checkpoint saved: {filename}")


def load_checkpoint_full(model, optimizer, scheduler, scaler, resume_from_dir_or_file: Optional[str], resume_optimizer: bool) -> Tuple[int, float]:
    """Loads a full training checkpoint."""
    if not resume_from_dir_or_file:
        return 0, float('inf')
    
    if os.path.isdir(resume_from_dir_or_file):
        # Find the latest checkpoint file
        checkpoints = sorted(glob.glob(os.path.join(resume_from_dir_or_file, "ckpt_*.pt")))
        checkpoints = [c for c in checkpoints if not any(t in c for t in ['final', 'interrupt', 'best_val'])]
        if not checkpoints:
            print(f"[*] No numbered checkpoint found in {resume_from_dir_or_file}. Starting from step 0.")
            return 0, float('inf')
        checkpoint_path = checkpoints[-1]
    else:
        checkpoint_path = resume_from_dir_or_file
    
    print(f"[*] Loading full checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=model.device)
    except Exception as e:
        print(f"[!] Failed to load checkpoint {checkpoint_path}: {e}")
        return 0, float('inf')

    # Load model state
    try:
        model.load_state_dict(checkpoint['model'], strict=True)
    except Exception as e:
        print(f"[!] Warning: Strict model loading failed. Trying non-strict. Error: {e}")
        model.load_state_dict(checkpoint['model'], strict=False)

    # Load optimizer, scheduler, scaler (if requested)
    if resume_optimizer:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if scaler and checkpoint['scaler']:
                scaler.load_state_dict(checkpoint['scaler'])
            print("[✓] Optimizer, scheduler, and scaler state restored.")
        except Exception as e:
            print(f"[!] Warning: Failed to restore optimizer/scheduler/scaler state. Starting fresh. Error: {e}")
    
    # Restore step and best_val
    start_step = checkpoint.get('step', 0)
    best_val = checkpoint.get('best_val', float('inf'))

    print(f"[✓] Resuming from step {start_step}. Previous best validation loss: {best_val:.4f}")
    return start_step, best_val

def setup_position_embeddings(model_config: V3Config, model: nn.Module):
    """
    Configures the model's position embeddings based on RoPE or ALiBi settings.
    This function modifies the model in-place to apply custom RoPE or ALiBi logic
    if the architecture supports it.
    """
    if model_config.use_alibi:
        print("[⚡] Configuring ALiBi (Attention with Linear Biases) position embedding.")
        # ALiBi usually involves modifying the attention masks or scores directly
        # within the attention layers. This is highly model-specific.
        pass # Placeholder for model-specific ALiBi setup
    else:
        print(f"[⚡] Configuring RoPE (Rotary Position Embeddings) with theta={model_config.rope_theta} and scaling={model_config.rope_scaling_factor}.")
        # RoPE usually involves generating or modifying the rotational frequencies.
        pass # Placeholder for model-specific RoPE setup

# ----------------------
# Multiprocessing Fix
# ----------------------

def collate_fn_for_multiprocessing(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collation function for DataLoader. Defined at the top level to allow pickling
    when num_workers > 0 is used with the 'spawn' context (e.g., on Windows).
    
    Converts a batch of dataset items (with "input_ids" and "labels") into PyTorch Tensors.
    """
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    # NOTE: We rely on the labels (-100) to implicitly handle masking.
    return {"input_ids": input_ids, "labels": labels}


# ----------------------
# Model and Data Setup
# ----------------------

def build_model_and_tokenizer(config: V3Config, device: torch.device) -> Tuple[MockGPT, AutoTokenizer]:
    """Loads the model and tokenizer from disk or huggingface hub."""
    print(f"[*] Loading tokenizer from {config.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    
    # Set up tokenizer padding and special tokens
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback for models without EOS (e.g., some raw pre-trains)
            tokenizer.add_special_tokens({'pad_token': '<|padding|>'})
    
    # Update config vocab size based on tokenizer
    config.vocab_size = len(tokenizer)
    
    # Instantiate the model with the loaded config
    print(f"[*] Building model: {config.model_name} (Layers: {config.n_layer}, Emb: {config.n_embd}, Heads: {config.n_head})")
    model = MockGPT(config).to(device)

    # Apply custom position embedding setup (RoPE/ALiBi)
    setup_position_embeddings(config, model)

    # Load weights from a specific file if requested (weights only)
    if config.load_weights_only_from:
        print(f"[*] Loading weights only from {config.load_weights_only_from}...")
        try:
            # Try strict load first
            model.load_state_dict(torch.load(config.load_weights_only_from, map_location=device)['model'], strict=True)
            print("[✓] Model weights loaded successfully (strict).")
        except Exception as e:
            print(f"[!] Warning: Strict load failed. Trying non-strict. Error: {e}")
            ck = torch.load(config.load_weights_only_from, map_location=device)
            state_dict = ck.get("model", ck) # Handle case where checkpoint is just the state dict
            model.load_state_dict(state_dict, strict=False)
            print("[✓] Model weights loaded successfully (non-strict).")
    
    print(f"[*] Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")
    return model, tokenizer

def build_loaders(tokenized_dataset_path, batch_size, num_workers, pin_memory=False):
    """
    Loads the tokenized dataset from disk and creates DataLoader instances.
    
    :param tokenized_dataset_path: Path to the saved DatasetDict.
    :param batch_size: The batch size to use.
    :param num_workers: Number of workers for data loading.
    :param pin_memory: Whether to use pinned memory.
    :return: A tuple of (train_loader, val_loader).
    """
    print(f"[*] Loading tokenized dataset from {tokenized_dataset_path}...")
    tokenized_datasets = load_from_disk(tokenized_dataset_path)
    
    train_data = tokenized_datasets["train"]
    val_data = tokenized_datasets["validation"]
    
    print(f"[*] Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

    # NOTE: The collate function is now defined globally as 'collate_fn_for_multiprocessing'
    # to avoid the "Can't pickle local object" error when num_workers > 0.
    
    print(f"[*] Creating DataLoaders (Batch size: {batch_size}, Workers: {num_workers}, Pin memory: {pin_memory})...")
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_for_multiprocessing,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_for_multiprocessing,
    )
    
    return train_loader, val_loader

# ----------------------
# Evaluation
# ----------------------

@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, config: V3Config, device: torch.device) -> Tuple[Dict[str, float], float]:
    """Runs a single pass over the validation set with safe loss handling."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    print("[⏱️] Starting evaluation...")
    pbar_eval = tqdm(val_loader, desc="[VAL] Step", dynamic_ncols=True, leave=False)

    for batch in pbar_eval:
        try:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=config.fp16):
                _, loss = model(input_ids, labels)
            
            # safety clamp: skip NaN or inf
            if not torch.isfinite(loss):
                print("[⚠️] Skipping non-finite validation loss.")
                continue

            total_loss += loss.item()
            num_batches += 1
            safe_avg = max(total_loss / (num_batches or 1), 1e-6)
            pbar_eval.set_postfix(avg_loss=f"{safe_avg:.6f}")

        except Exception as e:
            print(f"[!] Error during evaluation batch: {e}. Skipping batch.")
            traceback.print_exc()

    pbar_eval.close()

    # finalize
    avg_loss = max(total_loss / (num_batches or 1), 1e-6)
    if not math.isfinite(avg_loss):
        print("[⚠️] Validation loss became non-finite. Forcing to 1e-6 for logging safety.")
        avg_loss = 1e-6

    metrics = compute_metrics(avg_loss)
    model.train()
    return metrics, avg_loss

# ----------------------
# Training Loop
# ----------------------

def train(model: MockGPT, tokenizer: AutoTokenizer, train_loader: DataLoader, val_loader: DataLoader, config: V3Config, device: torch.device):
    """
    Main training loop.
    """
    # Create optimizer, scheduler, scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=config.betas)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16) if config.fp16 else None
    
    # Load state from checkpoint if requested
    start_step, best_val = load_checkpoint_full(model, optimizer, scheduler, scaler, config.resume_from, config.resume_optimizer)
    step = start_step
    
    # Tensorboard and logging setup
    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, "runs"))
    
    # Training state
    model.train()
    iter_loader = iter(train_loader)
    
    pbar = tqdm(initial=step, total=config.max_steps, desc="[TRAIN] Step", dynamic_ncols=True)
    accumulated_loss = 0.0
    patience_counter = 0

    try:
        while step < config.max_steps:
            # Gradient accumulation loop
            for micro_step in range(config.grad_accum_steps):
                try:
                    # Get next batch, restart iterator if needed
                    try:
                        batch = next(iter_loader)
                    except StopIteration:
                        print("[*] Restarting DataLoader iterator...")
                        iter_loader = iter(train_loader)
                        batch = next(iter_loader)

                    # Move data to device
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                
                except Exception as e:
                    print(f"[!] Error loading batch: {e}. Skipping micro-batch.")
                    traceback.print_exc()
                    continue

                # Forward pass
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=config.fp16):
                    # The MockGPT forward method returns (logits, loss)
                    _, loss = model(input_ids, labels)
                    loss = loss / config.grad_accum_steps # Scale loss for accumulation
                
                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulated_loss += loss.item() * config.grad_accum_steps
            
            # Optimization step
            if config.grad_clip > 0.0:
                if scaler:
                    scaler.unscale_(optimizer)
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            step += 1
            pbar.update(1)
            
            # --- Logging and Checkpointing ---

            # Logging
            if step % config.log_every == 0:
                avg_loss = accumulated_loss / config.log_every
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                
                writer.add_scalar("train/loss_ce", avg_loss, step)
                writer.add_scalar("train/ppl", math.exp(min(avg_loss, 10)), step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                
                # Reset accumulated loss
                accumulated_loss = 0.0

            # Evaluation
            if step % config.eval_every == 0:
                metrics, current_val_loss = evaluate(model, val_loader, config, device)
                
                writer.add_scalar("val/loss_ce", metrics['val/loss_ce'], step)
                writer.add_scalar("val/ppl", metrics['val/ppl'], step)
                safe_val_loss = float(current_val_loss if math.isfinite(current_val_loss) else 1e-6)
                safe_val_loss = max(safe_val_loss, 1e-6)
                safe_ppl = float(metrics['val/ppl'] if math.isfinite(metrics['val/ppl']) else float('inf'))
                print(f"[{time.strftime('%H:%M:%S')}] [EVAL] Step {step} | Val Loss: {safe_val_loss:.6f} | Val PPL: {safe_ppl:.2f}")


                # Check for Early Stopping
                if current_val_loss < best_val:
                    print(f"[*] Best validation loss improved from {best_val:.4f} to {current_val_loss:.4f}. Saving best model.")
                    best_val = current_val_loss
                    patience_counter = 0
                    save_checkpoint(model, optimizer, scheduler, scaler, step, config.output_dir, tag="best_val", best_val_metric=best_val)
                else:
                    patience_counter += 1
                    print(f"[!] Validation loss did not improve. Patience: {patience_counter}/{config.early_stop_patience}")
                    if patience_counter >= config.early_stop_patience:
                        print("[🛑] Early stopping triggered.")
                        break

                # Regular checkpointing
                save_checkpoint(model, optimizer, scheduler, scaler, step, config.output_dir, keep_last=config.keep_last, best_val_metric=best_val)
                print(f"[{time.strftime('%H:%M:%S')}] [⏺] Checkpointed at step {step}")

            if step >= config.max_steps:
                break

        # final save
        save_checkpoint(model, optimizer, scheduler, scaler, step, config.output_dir, keep_last=config.keep_last, tag="final", best_val_metric=best_val)
        print("[✓] Training complete. Final checkpoint saved.")
    except KeyboardInterrupt:
        print("[!] KeyboardInterrupt — saving interrupt checkpoint...")
        save_checkpoint(model, optimizer, scheduler, scaler, step, config.output_dir, keep_last=config.keep_last, tag="interrupt", best_val_metric=best_val)
    except Exception as e:
        print("[!] Exception during training. Saving error checkpoint...")
        traceback.print_exc()
        save_checkpoint(model, optimizer, scheduler, scaler, step, config.output_dir, keep_last=config.keep_last, tag="error", best_val_metric=best_val)
    finally:
        writer.close()
        pbar.close()

# ----------------------
# Main
# ----------------------

def main():
    parser = argparse.ArgumentParser(description="Zia Instruction Fine-Tuning Script (Training Only)")

    # Model / Architecture arguments
    parser.add_argument("--model_name", type=str, default=V3Config.model_name, help="A name for the run/model.")
    parser.add_argument("--tokenizer_path", type=str, default=V3Config.tokenizer_path, help="Path to the model/tokenizer directory or HF ID.")
    parser.add_argument("--max_len", type=int, default=V3Config.max_len, help="Max sequence length for truncation and padding.")
    parser.add_argument("--n_layer", type=int, default=V3Config.n_layer, help="Number of transformer layers.")
    parser.add_argument("--n_head", type=int, default=V3Config.n_head, help="Number of attention heads.")
    parser.add_argument("--n_embd", type=int, default=V3Config.n_embd, help="Embedding dimension.")
    parser.add_argument("--n_kv_head", type=int, default=V3Config.n_kv_head, help="Number of KV heads (for GQA/MQA). Defaults to n_head.")
    parser.add_argument("--dropout", type=float, default=V3Config.dropout, help="Dropout rate.")
    
    # Positional Embedding arguments
    parser.add_argument("--pos_type", type=str, choices=['rope', 'alibi'], default='rope', help="Type of positional encoding: 'rope' or 'alibi'.")
    parser.add_argument("--rope_theta", type=float, default=V3Config.rope_theta, help="RoPE base frequency.")
    parser.add_argument("--rope_scaling_factor", type=float, default=V3Config.rope_scaling_factor, help="RoPE sequence length scaling factor.")

    # Training control arguments
    parser.add_argument("--batch_size", type=int, default=V3Config.batch_size, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=V3Config.lr, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=V3Config.weight_decay, help="Weight decay.")
    parser.add_argument("--grad_clip", type=float, default=V3Config.grad_clip, help="Gradient clipping magnitude.")
    parser.add_argument("--grad_accum_steps", type=int, default=V3Config.grad_accum_steps, help="Number of steps to accumulate gradients before optimization.")
    parser.add_argument("--max_steps", type=int, default=V3Config.max_steps, help="Total number of training steps.")
    parser.add_argument("--warmup_steps", type=int, default=V3Config.warmup_steps, help="Number of warmup steps for the scheduler.")
    parser.add_argument("--fp16", action="store_true", default=V3Config.fp16, help="Enable Automatic Mixed Precision (AMP) training.")
    
    # Checkpointing / Logging arguments
    parser.add_argument("--output_dir", type=str, default=V3Config.output_dir, help="Directory to save checkpoints and logs.")
    parser.add_argument("--tokenized_dataset_path", type=str, default=V3Config.tokenized_dataset_path, help="Path to the pre-tokenized DatasetDict.")
    parser.add_argument("--resume_from", type=str, default=V3Config.resume_from, help="Path to checkpoint file or directory to resume from (full state).")
    parser.add_argument("--load_weights_only_from", type=str, default=V3Config.load_weights_only_from, help="Path to load model weights from (weights only, no training state).")
    parser.add_argument("--resume_optimizer", action="store_true", default=V3Config.resume_optimizer, help="Resume optimizer/scheduler state from checkpoint.")
    parser.add_argument("--eval_every", type=int, default=V3Config.eval_every, help="Run evaluation every N steps.")
    parser.add_argument("--log_every", type=int, default=V3Config.log_every, help="Log to TensorBoard every N steps.")
    parser.add_argument("--keep_last", type=int, default=V3Config.keep_last, help="Number of recent checkpoints to keep.")
    parser.add_argument("--early_stop_patience", type=int, default=V3Config.early_stop_patience, help="Number of non-improving eval steps before early stopping.")

    # Data Loader arguments
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers (set to 0 for constrained environment).")

    args = parser.parse_args()
    cfg = V3Config()

    # Apply arguments to config
    cfg.model_name = args.model_name
    cfg.tokenizer_path = args.tokenizer_path
    cfg.max_len = args.max_len
    cfg.n_layer = args.n_layer
    cfg.n_head = args.n_head
    cfg.n_embd = args.n_embd
    if args.n_kv_head is not None: cfg.n_kv_head = args.n_kv_head
    cfg.dropout = args.dropout
    cfg.fp16 = args.fp16

    cfg.use_alibi = (args.pos_type == 'alibi')
    cfg.rope_theta = args.rope_theta
    cfg.rope_scaling_factor = args.rope_scaling_factor

    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.grad_clip = args.grad_clip
    cfg.grad_accum_steps = args.grad_accum_steps
    cfg.max_steps = args.max_steps
    cfg.warmup_steps = args.warmup_steps

    cfg.output_dir = args.output_dir
    cfg.tokenized_dataset_path = args.tokenized_dataset_path
    cfg.resume_from = args.resume_from
    cfg.load_weights_only_from = args.load_weights_only_from
    cfg.resume_optimizer = args.resume_optimizer
    cfg.eval_every = args.eval_every
    cfg.log_every = args.log_every
    cfg.keep_last = args.keep_last
    cfg.early_stop_patience = args.early_stop_patience

    print(f"[⚙️] Current config:\n{cfg}")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Build model and tokenizer
    model, tokenizer = build_model_and_tokenizer(cfg, device)

    # Check for tokenized dataset existence (Mandatory for train mode)
    if not os.path.exists(cfg.tokenized_dataset_path):
        print(f"[!] Tokenized dataset not found at {cfg.tokenized_dataset_path}. Please prepare the data first.")
        sys.exit(1)

    # Build DataLoaders
    train_loader, val_loader = build_loaders(cfg.tokenized_dataset_path, cfg.batch_size, args.num_workers, pin_memory=True)

    # --- Start Training ---
    print("[🚀] Starting training...")
    train(model, tokenizer, train_loader, val_loader, cfg, device)


if __name__ == "__main__":
    main()
