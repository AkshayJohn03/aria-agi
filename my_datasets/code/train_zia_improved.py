#!/usr/bin/env python3
"""
train_zia_improved.py
Improved training script for ZIA with better architecture and training practices.
"""

import os
import time
import math
import json
import traceback
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# --- Silence TRANSFORMERS_CACHE warning & set cache dirs early ---
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

@dataclass
class ImprovedConfig:
    # Model architecture - slightly larger for better performance
    vocab_size: int = 60004
    d_model: int = 512  # Increased from 384
    n_layers: int = 10  # Increased from 8
    n_heads: int = 8    # Increased from 6
    mlp_ratio: int = 4
    max_len: int = 512  # Increased context length
    dropout: float = 0.1
    
    # Training parameters
    dataset_path: str = "my_datasets/processed/arrow_cleaned_v1"
    tokenizer_path: str = "artifacts/zia_tokenizer_60k"
    output_dir: str = "artifacts/zia_improved"
    
    batch_size: int = 4  # Reduced due to larger model
    grad_accum: int = 16  # Increased to maintain effective batch size
    lr: float = 2e-4     # Slightly lower for stability
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100000
    eval_every: int = 1000
    save_every: int = 2000
    keep_last: int = 3
    
    num_workers: int = 4
    fp16: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Resume options
    resume_from: Optional[str] = None
    load_weights_only_from: Optional[str] = None

class RotaryPositionalEmbedding(nn.Module):
    """RoPE implementation for better positional encoding"""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def rotate_half(self, x):
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, seq_len=None):
        seq_len = seq_len or q.size(-2)
        cos = self.cos_cached[:, :, :seq_len, :].to(q.device)
        sin = self.sin_cached[:, :, :seq_len, :].to(q.device)
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot

class ImprovedDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        # Multi-head attention with RoPE
        self.attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, 
            dropout=config.dropout, batch_first=True
        )
        self.rope = RotaryPositionalEmbedding(config.d_model // config.n_heads, config.max_len)
        
        # Feed-forward network
        hidden_dim = config.d_model * config.mlp_ratio
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(self, x, key_padding_mask=None):
        # Self-attention with RoPE
        B, T, C = x.shape
        h = self.ln1(x)
        
        # Apply RoPE to Q and K
        q = h.view(B, T, self.attn.num_heads, C // self.attn.num_heads).transpose(1, 2)
        k = h.view(B, T, self.attn.num_heads, C // self.attn.num_heads).transpose(1, 2)
        v = h.view(B, T, self.attn.num_heads, C // self.attn.num_heads).transpose(1, 2)
        
        q_rot, k_rot = self.rope.apply_rotary_pos_emb(q, k, T)
        q_rot = q_rot.transpose(1, 2).contiguous().view(B, T, C)
        k_rot = k_rot.transpose(1, 2).contiguous().view(B, T, C)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        
        attn_out, _ = self.attn(
            q_rot, k_rot, v.transpose(1, 2).contiguous().view(B, T, C),
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x

class ImprovedTinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            ImprovedDecoderBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.max_len = config.max_len
        self.pad_token_id = 0
        
        # Weight tying
        self.head.weight = self.tok.weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape
        if T > self.max_len:
            input_ids = input_ids[:, -self.max_len:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_len:]
            T = input_ids.size(1)

        x = self.tok(input_ids)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        for block in self.blocks:
            x = block(x, key_padding_mask)
        
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id
            )
        return logits, loss

def collate_fn(batch):
    """Improved collation with proper instruction formatting"""
    texts = []
    for ex in batch:
        if isinstance(ex, dict):
            if "text" in ex:
                text = ex["text"]
            elif "messages" in ex:
                # Format as instruction-following
                messages = ex["messages"]
                text = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        text += f"Human: {content}\n"
                    elif role == "assistant":
                        text += f"Assistant: {content}\n"
                text += "Assistant:"
            else:
                text = str(ex)
        else:
            text = str(ex)
        texts.append(text)
    
    # Tokenize with proper formatting
    tokenizer = AutoTokenizer.from_pretrained("artifacts/zia_tokenizer_60k")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    
    enc = tokenizer(
        texts, 
        truncation=True, 
        padding=True, 
        max_length=512, 
        return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"]

@torch.no_grad()
def evaluate(model, data_loader, device, max_batches=100):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    for i, (ids, mask) in enumerate(data_loader):
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        _, loss = model(ids, attention_mask=mask, labels=ids)
        total_loss += loss.item()
        n_batches += 1
        
        if i + 1 >= max_batches:
            break
    
    model.train()
    return total_loss / max(1, n_batches)

def train_loop(config: ImprovedConfig):
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)
    print(f"[i] Device: {device} | CUDA available: {torch.cuda.is_available()}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    print(f"[i] Tokenizer: vocab={len(tokenizer)} | pad_id={tokenizer.pad_token_id}")

    # Load dataset
    print(f"[i] Loading dataset from: {config.dataset_path}")
    ds_dict = load_from_disk(config.dataset_path)
    train_ds = ds_dict["train"]
    val_ds = ds_dict.get("validation", ds_dict.get("test"))
    
    if val_ds is None:
        # Create validation split
        split = train_ds.train_test_split(test_size=0.01, seed=config.seed)
        train_ds = split["train"]
        val_ds = split["test"]
    
    print(f"[i] Dataset sizes: train={len(train_ds)} | val={len(val_ds)}")

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=min(2, config.num_workers), pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn, drop_last=False
    )

    # Model
    model = ImprovedTinyGPT(config).to(device)
    model.pad_token_id = tokenizer.pad_token_id
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[i] Model parameters: {total_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.warmup_steps, 
        num_training_steps=config.max_steps
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.fp16 and device.type == "cuda" else None

    # TensorBoard
    os.makedirs(config.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(config.output_dir, "runs"))

    # Training loop
    step = 0
    best_val_loss = float("inf")
    micro_steps = 0
    running_loss = 0.0

    pbar = tqdm(total=config.max_steps, desc="Training", unit="step")
    
    try:
        model.train()
        while step < config.max_steps:
            for ids, mask in train_loader:
                ids = ids.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast(enabled=config.fp16 and device.type == "cuda"):
                    _, loss = model(ids, attention_mask=mask, labels=ids)
                    loss = loss / config.grad_accum

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                micro_steps += 1
                running_loss += loss.item() * config.grad_accum

                if micro_steps % config.grad_accum == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    step += 1
                    pbar.update(1)
                    
                    avg_loss = running_loss / config.grad_accum
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "step": step})
                    
                    writer.add_scalar("train/loss", avg_loss, step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                    running_loss = 0.0

                    # Evaluation
                    if step % config.eval_every == 0:
                        val_loss = evaluate(model, val_loader, device)
                        print(f"\n[Eval] step={step} val_loss={val_loss:.4f}")
                        writer.add_scalar("eval/loss", val_loss, step)
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            # Save best model
                            os.makedirs(os.path.join(config.output_dir, "best_val"), exist_ok=True)
                            torch.save({
                                "model": model.state_dict(),
                                "step": step,
                                "val_loss": val_loss,
                                "config": config
                            }, os.path.join(config.output_dir, "best_val", "checkpoint.pt"))
                            print(f"[🏆] New best model saved (val_loss={val_loss:.4f})")

                    # Checkpointing
                    if step % config.save_every == 0:
                        os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
                        torch.save({
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": step,
                            "config": config
                        }, os.path.join(config.output_dir, "checkpoints", f"step_{step:06d}.pt"))

                    if step >= config.max_steps:
                        break
            
            if step >= config.max_steps:
                break

        print("[✓] Training complete.")

    except KeyboardInterrupt:
        print("[!] Training interrupted. Saving checkpoint...")
        os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": config
        }, os.path.join(config.output_dir, "checkpoints", f"interrupt_step_{step:06d}.pt"))
    except Exception as e:
        traceback.print_exc()
        raise
    finally:
        writer.close()
        pbar.close()

if __name__ == "__main__":
    config = ImprovedConfig()
    train_loop(config)
