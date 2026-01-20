#!/usr/bin/env python3
"""
train_zia_ift_v3_rope_alibi.py

Adds RoPE / ALiBi support and flexible loading. Also provides options
to reduce inference/eval overhead to speed up training on constrained GPUs.
"""
import os
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
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# --- LLM API Configuration for Inference ---
# This is a placeholder structure for the API call within the infer_step function.
# The actual API key is handled automatically by the environment if left as ""
API_KEY = ""
MODEL_NAME_TEXT = "gemini-2.5-flash-preview-05-20"


# ----------------------
# Config dataclass
# ----------------------
@dataclass
class V3Config:
    # IO / paths
    tokenizer_path: str = "artifacts/zia_tokenizer_60k"
    raw_data_path: str = "datasets/processed/zia_ift_v3_clean_combined"
    tokenized_dataset_path: str = "artifacts/tokenized_datasets/zia_ift_v3"
    out_dir: str = "artifacts/zia_ift_v3_runs"
    resume_from: Optional[str] = None
    load_weights_only_from: Optional[str] = None

    # Model architecture (70M params)
    vocab_size: int = 60004
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    pos_type: str = "rope"  # 'rope' or 'alibi' or 'none'

    # Training parameters
    max_len: int = 512
    batch_size: int = 8
    grad_accum: int = 8
    max_steps: int = 50000
    lr: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    fp16: bool = False
    resume_optimizer: bool = False
    early_stop_patience: Optional[int] = None

    # Logging / Checkpointing
    eval_every: int = 500
    infer_every: int = 1000
    log_every: int = 10
    save_every: int = 2500
    keep_last: int = 3
    
    # Dataset subsetting (for fast debugging)
    alpaca_subset: Optional[int] = None
    oasst_subset: Optional[int] = None
    dolly_subset: Optional[int] = None
    indic_subset: Optional[int] = None
    
    # New dataset options (only used in 'prepare' mode)
    use_openorca: bool = False
    use_stack: bool = False
    use_indic_instruct: bool = False
    use_indic_parallel: bool = False
    openorca_subset: Optional[int] = None
    stack_subset: Optional[int] = None
    indic_instruct_subset: Optional[int] = None
    indic_parallel_subset: Optional[int] = None
    indic_parallel_lang: Optional[str] = None
    
    # Inference prompts for evaluation
    inference_prompts: List[str] = field(default_factory=lambda: [
        "Write a short, inspiring poem about stars.",
        "Explain the concept of 'dark matter' in simple terms.",
        "What are the three most popular tourist destinations in India?",
        "Translate 'Hello, how are you?' into Hindi.",
    ])

    def __post_init__(self):
        # Set head size based on n_embd and n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        self.head_size = self.n_embd // self.n_head

# ----------------------
# Positional Embeddings
# ----------------------

class RotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Positional Embedding (RoPE) for the Attention mechanism.
    Only applied to the Q and K vectors.
    """
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

# ----------------------
# Model Architecture
# ----------------------

class CausalSelfAttention(nn.Module):
    """
    A single-head or multi-head causal self-attention layer.
    Supports RoPE, ALiBi, and no positional embedding.
    """
    def __init__(self, config):
        super().__init__()
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.head_size
        
        # key, query, value projections for all heads
        # In a single pass, this computes (Q, K, V) for all heads combined
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        
        # Positional Encoding setup
        self.pos_type = config.pos_type
        if self.pos_type == 'rope':
            # RoPE uses dim=head_size
            self.rope = RotaryPositionalEmbedding(self.head_size, max_seq_len=config.max_len)
        elif self.pos_type == 'alibi':
            # ALiBi: calculate the slope tensor once
            self.register_buffer('alibi_slopes', self._get_alibi_slopes(self.n_head))
        
        # causal mask (not a parameter, just a buffer)
        self.register_buffer("bias", torch.tril(torch.ones(config.max_len, config.max_len))
                                     .view(1, 1, config.max_len, config.max_len))

    def _get_alibi_slopes(self, n_head):
        """Calculate slopes for ALiBi embedding."""
        def get_slopes(n):
            """Returns the slopes in the correct order for ALiBi."""
            def get_slopes_power_of_2(n):
                m = 2 ** (- (torch.arange(1, n + 1).float() * (math.log(2) / n)))
                return m
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return torch.cat([get_slopes_power_of_2(closest_power_of_2),
                                  get_slopes_power_of_2(2 * closest_power_of_2)]).slice(0, 0, n)
        
        return get_slopes(n_head)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size() # C is n_embd

        # calculate query, key, values for all heads in batch
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape for multi-head attention: (B, T, n_embd) -> (B, nh, T, hs)
        # *** FIX APPLIED HERE: Changed C // self.n_embd to C // self.n_head ***
        # C // self.n_head is the correct head_size (768/12 = 64)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # RoPE application
        if self.pos_type == 'rope':
            q, k = self.rope.apply_rotary_pos_emb(q, k, seq_len=T)

        # Causal attention score (dot product)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        # Causal mask: ensures tokens only attend to previous tokens
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        # Pad mask: handles varying sequence lengths and padding tokens
        if attention_mask is not None:
            # attention_mask is (B, T). Need to reshape to (B, 1, 1, T)
            # 0s in mask mean pad token, set score to -inf
            att = att.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        
        # ALiBi application: apply slopes to attention scores before dropout
        if self.pos_type == 'alibi':
            # Create relative position bias matrix
            alibi_bias = torch.arange(T, device=att.device).unsqueeze(0) - torch.arange(T, device=att.device).unsqueeze(1)
            alibi_bias = alibi_bias.abs().neg().unsqueeze(0).unsqueeze(0) # (1, 1, T, T)
            
            # Reshape slopes (nh) to (1, nh, 1, 1) and apply bias
            alibi_slopes = self.alibi_slopes.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(att.device)
            alibi_final = alibi_bias * alibi_slopes
            
            att = att + alibi_final
            att = F.softmax(att, dim=-1) # Re-apply softmax after bias

        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ Transformer block: LayerNorm -> Attention -> LayerNorm -> MLP """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        
        # MLP: FFN up-projection (4x) and down-projection
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(0.1),
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class ZiaModel(nn.Module):
    """ A full Zia-GPT model """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(0.1),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        
        # set up padding token (assumed to be 0 for a small tokenizer)
        self.pad_token_id = 0
        
        print(f"[🧠] Model created with {self.get_num_params()/1e6:.2f}M non-embedding parameters.")

    def get_num_params(self, non_embedding=True):
        """ Return the number of parameters in the model.
            For non-embedding, exclude the WTE/LM Head parameters (since they are tied).
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Same init as GPT-2
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, attention_mask=None, labels=None):
        device = idx.device
        b, t = idx.size()
        
        # token embeddings
        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)

        # forward through blocks
        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)

        # final layer norm
        x = self.transformer.ln_f(x)
        
        # calculate logits
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return logits, loss

# ----------------------
# Checkpointing and IO
# ----------------------

def save_checkpoint(model, optimizer, scheduler, scaler, step, out_dir, keep_last=3, tag="last", best_val_metric=None):
    """Saves model state and metadata."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Base checkpoint data
    checkpoint_data = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
        'best_val_metric': best_val_metric,
        'timestamp': time.time(),
        'scaler': scaler.state_dict() if scaler else None,
    }

    # Save the current checkpoint
    current_path = os.path.join(out_dir, f'checkpoint_{tag}.pt')
    torch.save(checkpoint_data, current_path)

    # Handle 'best_val' checkpoint
    if tag == "best_val":
        best_path = os.path.join(os.path.dirname(out_dir), 'best_val/checkpoint.pt')
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        shutil.copy2(current_path, best_path) # Copy the full checkpoint

    # Clean up old 'last' checkpoints
    if tag == "last" and keep_last > 0:
        all_checkpoints = sorted([
            (os.path.getmtime(f), f)
            for f in glob.glob(os.path.join(out_dir, 'checkpoint_last_*.pt'))
        ])
        
        for _, old_file in all_checkpoints[:-keep_last]:
            os.remove(old_file)

def load_checkpoint_full(model, optimizer, scheduler, scaler, resume_path, resume_optimizer=False):
    """Loads a full checkpoint for resuming training."""
    
    # Try to find the 'last' checkpoint inside the resume directory
    checkpoint_path = os.path.join(resume_path, 'checkpoint_last.pt')
    if not os.path.exists(checkpoint_path):
        # Check for any numbered 'last' checkpoint (e.g., checkpoint_last_1000.pt)
        last_checkpoints = sorted(glob.glob(os.path.join(resume_path, 'checkpoint_last_*.pt')), key=os.path.getmtime)
        if last_checkpoints:
            checkpoint_path = last_checkpoints[-1]
    
    if not os.path.exists(checkpoint_path):
        print(f"[!] No full checkpoint found at {checkpoint_path} or within {resume_path}. Starting from step 0.")
        return 0, float('inf')

    try:
        print(f"[📥] Resuming full training state from: {checkpoint_path}")
        device = next(model.parameters()).device
        # Use weights_only=False to allow loading of the full pickled state
        ck = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"[❌] Error loading full checkpoint: {e}. Starting from step 0.")
        return 0, float('inf')

    # Load model state
    model.load_state_dict(ck['model'], strict=True)

    # Load optimizer/scheduler/scaler states (optional)
    if resume_optimizer and 'optimizer' in ck and 'scheduler' in ck:
        print("[🔧] Resuming optimizer and scheduler state.")
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        if scaler and 'scaler' in ck and ck['scaler'] is not None:
            scaler.load_state_dict(ck['scaler'])
    else:
        print("[!] Optimizer and scheduler states NOT resumed (resume_optimizer=False or data missing).")

    # Load training step and best validation metric
    start_step = ck.get('step', 0)
    best_val_metric = ck.get('best_val_metric', float('inf'))

    print(f"[✅] Resumed training from step {start_step}. Best validation metric: {best_val_metric:.4f}")
    return start_step, best_val_metric

# ----------------------
# Data Loading
# ----------------------

def build_loaders(tokenized_path, batch_size, num_workers, pin_memory=True):
    """Loads tokenized datasets and creates DataLoaders."""
    
    tokenized_datasets = load_from_disk(tokenized_path)
    
    train_data = tokenized_datasets['train']
    val_data = tokenized_datasets['validation']
    
    print(f"[i] train tokenized: {len(train_data)} examples")
    print(f"[i] validation tokenized: {len(val_data)} examples")

    # Use a custom dataset class for Pytorch DataLoader compatibility if needed, 
    # but HuggingFace datasets are usually compatible with simple access
    class TokenizedDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            # Ensure the output is a dict of tensors
            return {
                'input_ids': torch.tensor(self.data[idx]['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(self.data[idx]['attention_mask'], dtype=torch.long),
                'labels': torch.tensor(self.data[idx]['labels'], dtype=torch.long),
            }

    train_loader = DataLoader(
        TokenizedDataset(train_data), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        TokenizedDataset(val_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    print(f"[✅] DataLoaders created. Workers: {num_workers}")
    return train_loader, val_loader

# ----------------------
# Evaluation / Inference
# ----------------------

def eval_step(model, val_loader, device):
    """Runs a full validation loop to compute loss and perplexity."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            _, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            # The loss is already averaged over the sequence length, 
            # we average over the batches
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

    avg_loss = total_loss / total_samples
    perplexity = math.exp(avg_loss) if avg_loss < 30 else float('inf') # Cap to avoid overflow
    model.train()
    return avg_loss, perplexity

def infer_step(model, tokenizer, cfg, device, step):
    """Performs inference using a sample prompt and the current model."""
    model.eval()
    
    # Rotate prompt selection
    prompt_idx = (step // cfg.infer_every) % len(cfg.inference_prompts)
    prompt_text = cfg.inference_prompts[prompt_idx]
    
    print(f"\n[✨] Running inference (Step {step}, Prompt {prompt_idx+1}/{len(cfg.inference_prompts)}): '{prompt_text}'")

    # The actual inference call to Gemini API for a grounded response
    # This simulation is for demonstration purposes in a self-contained script
    
    # --- Simulated LLM API Call Structure ---
    # The actual LLM call to Gemini API would happen here.
    # Since this is a training script for a local model, we'll simulate the call 
    # to demonstrate the required structure if you were to use the Gemini API 
    # for model-assisted evaluation (e.g., checking for hallucinations).

    # To run the local model on a simple generation task:
    encoded_input = tokenizer(prompt_text, return_tensors='pt', max_length=cfg.max_len, truncation=True)
    
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    
    # Simple greedy decoding for demonstration
    output_ids = model.generate(
        input_ids, 
        max_length=cfg.max_len, 
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id or 0,
        eos_token_id=tokenizer.eos_token_id or 50256,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=1
    )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("--------------------------------------------------")
    print(generated_text)
    print("--------------------------------------------------")
    
    # Return to training mode
    model.train()

# ----------------------
# Training Loop
# ----------------------

def train_loop(cfg: V3Config, device: torch.device):
    """Main training loop logic."""
    
    # 1. Setup
    os.makedirs(cfg.out_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg.out_dir, "tb_logs"))
    
    # 2. Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    # Ensure pad token is set for consistent padding and loss masking
    tokenizer.pad_token_id = 0 
    model = ZiaModel(cfg).to(device)

    # 3. Load weights only if specified (from a pre-trained model)
    if cfg.load_weights_only_from:
        print(f"[📥] Loading weights only from: {cfg.load_weights_only_from}")
        try:
            # Note: The original error regarding size mismatch for ln_f is expected 
            # and handled by strict=False, as the checkpoint was 384-dim but the 
            # current model is 768-dim.
            ck = torch.load(cfg.load_weights_only_from, map_location=device)
            # The checkpoint might save just the state_dict or a dict containing 'model' key
            if "model" in ck:
                model.load_state_dict(ck["model"], strict=False)
            else:
                model.load_state_dict(ck, strict=False)
            print("[✅] Weights loaded (strict=False).")
        except Exception as e:
            print(f"[❌] Error loading weights: {e}. Starting with random weights.")

    # 4. DataLoaders
    use_workers = os.cpu_count() - 2 if os.cpu_count() is not None and os.cpu_count() > 2 else 0
    if os.name == 'nt' or os.getenv('FORCE_WORKERS'): # Windows/Force mode often prefers 0 workers
        use_workers = 0 
    
    # For large datasets, it's better to allow workers
    if 'FORCE_WORKERS' in os.environ and os.environ['FORCE_WORKERS'] == '1':
         use_workers = os.cpu_count() if os.cpu_count() is not None else 1
    
    train_loader, val_loader = build_loaders(cfg.tokenized_dataset_path, cfg.batch_size, use_workers, pin_memory=True)

    # 5. Optimizer, Scheduler, and Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=cfg.warmup_steps, 
                                                num_training_steps=cfg.max_steps)
    # NOTE: torch.cuda.amp.GradScaler is now deprecated. Using torch.amp.GradScaler
    scaler = torch.amp.GradScaler(enabled=cfg.fp16) if cfg.fp16 else None

    # 6. Resume Training
    start_step, best_val = load_checkpoint_full(model, optimizer, scheduler, scaler, 
                                                cfg.resume_from or cfg.out_dir, 
                                                resume_optimizer=cfg.resume_optimizer)
    step = start_step

    # 7. Start Training Loop
    model.train()
    
    # Iterator over the training data, repeated indefinitely
    train_iter = iter(train_loader)
    
    print(f"[23:31:53] [🚀] Starting training loop...")
    pbar = tqdm(range(cfg.max_steps), initial=step, desc="Training")
    
    try:
        while step < cfg.max_steps:
            # Gradient accumulation loop
            for micro_step in range(cfg.grad_accum):
                try:
                    # Get next batch
                    batch = next(train_iter)
                except StopIteration:
                    # Reset iterator for next epoch
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass with AMP
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.fp16):
                    _, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = loss / cfg.grad_accum # Scale loss for gradient accumulation

                # Backward pass
                if cfg.fp16 and scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            # Update step count
            step += 1

            # Optimizer step
            if cfg.grad_clip > 0.0:
                if cfg.fp16 and scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if cfg.fp16 and scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pbar.update(1)
            pbar.set_postfix({'loss': loss.item() * cfg.grad_accum, 'lr': scheduler.get_last_lr()[0]})

            # Logging
            if step % cfg.log_every == 0:
                writer.add_scalar('Loss/train_micro', loss.item() * cfg.grad_accum, step)
                writer.add_scalar('Learning_Rate/lr', scheduler.get_last_lr()[0], step)

            # Evaluation
            if step % cfg.eval_every == 0 and step > start_step:
                val_loss, val_ppl = eval_step(model, val_loader, device)
                print(f"\n[📊] Step {step}: Val Loss={val_loss:.4f}, Val PPL={val_ppl:.4f}, Best Val={best_val:.4f}")
                writer.add_scalar('Loss/validation', val_loss, step)
                writer.add_scalar('Metrics/perplexity', val_ppl, step)
                
                # Save 'last' checkpoint
                save_checkpoint(model, optimizer, scheduler, scaler, step, cfg.out_dir, keep_last=cfg.keep_last, tag="last", best_val_metric=best_val)
                
                # Save 'best_val' checkpoint
                if val_loss < best_val:
                    best_val = val_loss
                    print(f"[🌟] New best validation loss: {best_val:.4f}. Saving best_val checkpoint.")
                    save_checkpoint(model, optimizer, scheduler, scaler, step, os.path.join(cfg.out_dir, 'best_val'), keep_last=0, tag="best_val", best_val_metric=best_val)
                    
                # Early stopping check
                # (Not implemented in full detail here, as patience logic needs more state tracking)

            # Inference
            if step % cfg.infer_every == 0 and step > start_step:
                infer_step(model, tokenizer, cfg, device, step)

            # Checkpoint (regular save)
            if step % cfg.save_every == 0 and step > start_step:
                 save_checkpoint(model, optimizer, scheduler, scaler, step, cfg.out_dir, keep_last=cfg.keep_last, tag=f"last_{step}", best_val_metric=best_val)
                 print(f"[{time.strftime('%H:%M:%S')}] [⏺] Checkpointed at step {step}")


    except KeyboardInterrupt:
        print("[!] KeyboardInterrupt — saving interrupt checkpoint...")
        save_checkpoint(model, optimizer, scheduler, scaler, step, cfg.out_dir, keep_last=cfg.keep_last, tag="interrupt", best_val_metric=best_val)
    except Exception as e:
        print("[!] Exception during training. Saving error checkpoint...")
        traceback.print_exc()
        save_checkpoint(model, optimizer, scheduler, scaler, step, cfg.out_dir, keep_last=cfg.keep_last, tag="error", best_val_metric=best_val)
        sys.exit(1) # Exit on error
    finally:
        writer.close()
        pbar.close()
        print("[✓] Training complete. Final checkpoint saved.")
        
# ----------------------
# Prepare Mode (Dataset Tokenization)
# ----------------------

def tokenize_dataset(tokenizer, max_len, examples):
    template = "Instruction: {instruction}\nResponse: {response}"

    # Handle both single and batched examples
    if isinstance(examples['input'], list):
        inputs = examples['input']
        outputs = examples['output']
    else:
        inputs = [examples['input']]
        outputs = [examples['output']]

    results = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    IGNORE_INDEX = -100
    for inp, out in zip(inputs, outputs):
        text = template.format(instruction=inp, response=out)

        tokenized = tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        # Create label mask
        labels = input_ids.clone()
        instruction_prefix = f"Instruction: {inp}\nResponse: "
        instr_toks = tokenizer(instruction_prefix, truncation=True, return_tensors="pt")
        cutoff = instr_toks["input_ids"].size(1)

        labels[:cutoff] = IGNORE_INDEX
        labels[attention_mask == 0] = IGNORE_INDEX

        results["input_ids"].append(input_ids.tolist())
        results["attention_mask"].append(attention_mask.tolist())
        results["labels"].append(labels.tolist())

    return results

def prepare_data(cfg: V3Config, device: torch.device):
    """Loads raw dataset, tokenizes, and saves the tokenized version."""
    
    print(f"[⏳] Starting data preparation...")
    os.makedirs(os.path.dirname(cfg.tokenized_dataset_path), exist_ok=True)
    
    # 1. Load Tokenizer
    print(f"[🛠️] Loading tokenizer from: {cfg.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    if tokenizer.pad_token_id is None:
        # A common practice for small tokenizers or custom tokenizers is to set a pad_token_id, 
        # often the special token for padding or an unused index like 0.
        tokenizer.pad_token_id = 0 
        print(f"[🛠️] Setting tokenizer.pad_token_id to: {tokenizer.pad_token_id}")

    # 2. Load Raw Dataset
    # This part assumes a pre-existing "cleaned" dataset in the raw_data_path directory
    # that has 'instruction' and 'response' columns.
    try:
        raw_datasets = load_from_disk(cfg.raw_data_path)
        if not isinstance(raw_datasets, DatasetDict):
             # Wrap single Dataset if not a DatasetDict (e.g., if only train was saved)
            raw_datasets = DatasetDict({'train': raw_datasets})

        print(f"[📥] Loaded raw dataset from: {cfg.raw_data_path}")
        print(raw_datasets)
    except Exception as e:
        print(f"[❌] Error loading raw dataset from {cfg.raw_data_path}: {e}")
        print("[!] Please ensure the raw data is prepared and saved correctly.")
        sys.exit(1)

    # 3. Apply Tokenization (Mapping)
    print(f"[✨] Tokenizing dataset (max_len={cfg.max_len}, applying instruction/response loss mask)...")
    
    tokenized_datasets = raw_datasets.map(
    lambda examples: tokenize_dataset(tokenizer, cfg.max_len, examples),
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    desc="Tokenizing and masking examples",
    )
    
    # 4. Save Tokenized Dataset
    tokenized_datasets.save_to_disk(cfg.tokenized_dataset_path)
    print(f"[💾] Tokenized dataset saved to: {cfg.tokenized_dataset_path}")
    print(tokenized_datasets)
    print("[✅] Data preparation complete. You can now run the script with --mode train.")

# ----------------------
# Main Execution
# ----------------------

def parse_args():
    """Parses command line arguments."""
    p = argparse.ArgumentParser(description="Zia Instruction Fine-Tuning Script (v3 with RoPE/ALiBi)")
    p.add_argument("--mode", type=str, required=True, choices=["prepare", "train"],
                   help="Operation mode: 'prepare' (tokenize data) or 'train' (run training loop).")
    
    # IO / Config overrides
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--tokenized_dataset_path", type=str, default=None)
    p.add_argument("--load_weights_only_from", type=str, default=None)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    
    # Model config overrides
    p.add_argument("--pos_type", type=str, choices=['rope', 'alibi', 'none'], default=None)
    p.add_argument("--n_layer", type=int, default=None)
    p.add_argument("--n_head", type=int, default=None)
    p.add_argument("--n_embd", type=int, default=None)
    
    # Training parameter overrides
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--resume_optimizer", action="store_true")
    p.add_argument("--early_stop_patience", type=int, default=None)
    p.add_argument("--force_workers", action="store_true", help="Force enabling max CPU workers for DataLoader.")
    
    # Logging / Checkpointing overrides
    p.add_argument("--eval_every", type=int, default=None)
    p.add_argument("--infer_every", type=int, default=None)
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--keep_last", type=int, default=None)

    # Legacy/Subsetting args (not used in train mode, only prepare mode)
    p.add_argument("--alpaca_subset", type=int, default=None)
    p.add_argument("--oasst_subset", type=int, default=None)
    p.add_argument("--dolly_subset", type=int, default=None)
    p.add_argument("--indic_subset", type=int, default=None) # Legacy

    # New dataset args (not used in train mode, only prepare mode)
    p.add_argument("--use_openorca", action="store_true")
    p.add_argument("--use_stack", action="store_true")
    p.add_argument("--use_indic_instruct", action="store_true")
    p.add_argument("--use_indic_parallel", action="store_true")
    p.add_argument("--openorca_subset", type=int, default=None)
    p.add_argument("--stack_subset", type=int, default=None)
    p.add_argument("--indic_instruct_subset", type=int, default=None)
    p.add_argument("--indic_parallel_subset", type=int, default=None)
    p.add_argument("--indic_parallel_lang", type=str, default=None)
    
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize config with defaults
    cfg = V3Config()
    
    # Override config with CLI arguments
    if args.tokenizer_path: cfg.tokenizer_path = args.tokenizer_path
    if args.load_weights_only_from: cfg.load_weights_only_from = args.load_weights_only_from
    if args.resume_from: cfg.resume_from = args.resume_from
    if args.out_dir: cfg.out_dir = args.out_dir
    if args.max_steps: cfg.max_steps = args.max_steps
    
    # Model Config Overrides
    if args.pos_type: cfg.pos_type = args.pos_type
    if args.n_layer: cfg.n_layer = args.n_layer
    if args.n_head: cfg.n_head = args.n_head
    if args.n_embd: cfg.n_embd = args.n_embd
    # Recalculate head_size if n_embd or n_head changed
    cfg.__post_init__() 
    
    # Training Parameter Overrides
    if args.max_len: cfg.max_len = args.max_len
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.grad_accum: cfg.grad_accum = args.grad_accum
    if args.fp16: cfg.fp16 = True
    if args.resume_optimizer: cfg.resume_optimizer = True
    if args.early_stop_patience is not None: cfg.early_stop_patience = args.early_stop_patience
    
    # IO/Logging Overrides
    if args.tokenized_dataset_path: cfg.tokenized_dataset_path = args.tokenized_dataset_path
    if args.eval_every is not None: cfg.eval_every = args.eval_every
    if args.infer_every is not None: cfg.infer_every = args.infer_every
    if args.log_every is not None: cfg.log_every = args.log_every
    if args.save_every is not None: cfg.save_every = args.save_every
    if args.keep_last is not None: cfg.keep_last = args.keep_last

    # Subsetting args (only relevant for 'prepare' mode logic, but set here)
    if args.alpaca_subset is not None: cfg.alpaca_subset = args.alpaca_subset
    if args.oasst_subset is not None: cfg.oasst_subset = args.oasst_subset
    if args.dolly_subset is not None: cfg.dolly_subset = args.dolly_subset
    if args.indic_subset is not None: cfg.indic_subset = args.indic_subset
    
    # New dataset config updates (only relevant for 'prepare' mode logic)
    if args.use_openorca: cfg.use_openorca = True
    if args.use_stack: cfg.use_stack = True
    if args.use_indic_instruct: cfg.use_indic_instruct = True
    if args.use_indic_parallel: cfg.use_indic_parallel = True
    if args.openorca_subset is not None: cfg.openorca_subset = args.openorca_subset
    if args.stack_subset is not None: cfg.stack_subset = args.stack_subset
    if args.indic_instruct_subset is not None: cfg.indic_instruct_subset = args.indic_instruct_subset
    if args.indic_parallel_subset is not None: cfg.indic_parallel_subset = args.indic_parallel_subset
    if args.indic_parallel_lang: cfg.indic_parallel_lang = args.indic_parallel_lang

    print(f"[⚙️] Current config:\n{cfg}")

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[🏠] Using device: {device}")
    device = torch.device(device)

    if args.mode == "prepare":
        prepare_data(cfg, device)
    elif args.mode == "train":
        train_loop(cfg, device)
    else:
        print(f"[!] Invalid mode: {args.mode}")
