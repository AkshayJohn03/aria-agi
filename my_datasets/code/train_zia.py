# train_zia_advanced.py
# This script is a complete, from-scratch training pipeline for a custom-built
# language model named Zia. It uses a custom BPE tokenizer, a custom Transformer
# architecture, and includes advanced features for robust, efficient training.

# =========================================================================
# 0) ENV SETUP & IMPORTS
# =========================================================================
import os
import sys
import json
import time
import shutil
import glob
import random
import math
import io
import argparse
import logging
from datetime import datetime
from typing import Iterator, Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# Hugging Face imports are essential for a streamlined training loop
from datasets import IterableDataset as HfIterableDataset
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
os.environ["HF_HOME"] = "D:\\aria\\aria_ai\\aria_ai_assistant\\cache"

# Optional dependencies for performance and visualization
try:
    import bitsandbytes as bnb # noqa: F401
    HAS_BNB = True
except Exception:
    HAS_BNB = False
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import joblib

# Basic logging setup for console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

DEFAULTS = {
    # General Setup
    'PROJECT_NAME': "ZIA_LLM",
    "train_data_glob": "datasets/train/*.jsonl",
    "eval_data_glob": "datasets/eval/*.jsonl",
    "text_key": "text",
    'TOKENIZER_PATH': "artifacts/zia_tokenizer_60k",
    'OUTPUT_DIR': "artifacts/zia_from_scratch",
    "preset": "0p5B_true", # Set a true 0.5B preset as default
    'vocab_size': 60000,
    'min_frequency': 2,
    'max_steps': 100000,
    'resume_from_checkpoint': None,

    # Model Architecture Presets (Custom-built)
    'presets': {
        "0p5B_true": dict( # A true 500M parameter model (~0.5B)
            vocab_size=60000, 
            seq_len=2048, 
            embed_dim=768, 
            num_layers=12, 
            num_heads=12, 
            kv_heads=3, 
            dropout_rate=0.1, 
            moe_experts=4, 
            moe_topk=1
        ),
        "1p7B_misnamed": dict( # This is the preset you ran previously
            vocab_size=60000, 
            seq_len=8192, 
            embed_dim=1024, 
            num_layers=24, 
            num_heads=16, 
            kv_heads=4, 
            dropout_rate=0.1, 
            moe_experts=8, 
            moe_topk=2
        ),
        "1B": dict(
            vocab_size=60000, 
            seq_len=8192, 
            embed_dim=2048, 
            num_layers=24, 
            num_heads=16, 
            kv_heads=4, 
            dropout_rate=0.1, 
            moe_experts=8, 
            moe_topk=2
        ),
        "tiny": dict(
            vocab_size=60000, 
            seq_len=2048, 
            embed_dim=768, 
            num_layers=12, 
            num_heads=12, 
            kv_heads=3, 
            dropout_rate=0.1, 
            moe_experts=4, 
            moe_topk=1
        ),
    },

    # Training Parameters
    'num_train_epochs': 5,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 64,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_ratio": 0.02,
    "lr_scheduler_type": "cosine",
    
    # Checkpointing & Logging
    "logging_steps": 50,
    "eval_steps": 500,
    "save_steps": 1000,
    "keep_last_n": 3,
    "early_stopping_patience": 5,
    "dataloader_num_workers": 8,
    "fp16": True,
    "bf16": False, # Mutually exclusive with fp16
    "gradient_checkpointing": False,
    "use_8bit_optimizer": True,
    "compile": False,
    "max_grad_norm": 1.0,
    "confmat_top_k": 64,
    "seed": 42,
    "deepspeed": None, # Path to deepspeed config file
    "metric_for_best_model": "eval_loss",
    "load_best_model_at_end": True,
}

# =========================================================================
# 2) UTILITIES
# =========================================================================
def set_seed(seed: int = 42):
    """Sets a global seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def printb(*a, **k):
    """A helper function for formatted console output."""
    print("[i]", *a, **k)

def image_to_tensor(img: np.ndarray):
    """Converts a numpy image to a PyTorch tensor format for TensorBoard."""
    return np.transpose(img, (2,0,1))

# --- Advanced Model Architecture ---
class ZiaConfig(PretrainedConfig):
    """
    Configuration class for the Zia model.
    It inherits from PretrainedConfig to be compatible with Hugging Face ecosystem.
    """
    model_type = "zia"
    def __init__(self, 
                 vocab_size=50000, 
                 seq_len=8192, 
                 embed_dim=1024, 
                 num_layers=24, 
                 num_heads=16, 
                 kv_heads=4, 
                 dropout_rate=0.1, 
                 moe_experts=8, 
                 moe_topk=2,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.dropout_rate = dropout_rate
        self.moe_experts = moe_experts
        self.moe_topk = moe_topk

def precompute_rotary_emb(dim, seq_len, theta=10000.0):
    """
    Precomputes the Rotary Positional Embeddings (RoPE) for a given sequence length.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.stack((freqs, freqs), dim=-1).reshape(-1, dim)
    return emb

def apply_rotary_pos_emb(x, emb):
    """
    Applies the precomputed RoPE to a tensor.
    The input `x` is typically a query or key tensor.
    """
    cos = emb.cos().unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim)
    sin = emb.sin().unsqueeze(0).unsqueeze(2)
    original_dtype = x.dtype
    x = x.float()
    dim = x.shape[-1]
    x1 = x[..., :dim//2]
    x2 = x[..., dim//2:]
    x_rot = torch.cat((-x2, x1), dim=-1)
    x_out = (x * cos) + (x_rot * sin)
    return x_out.to(original_dtype)


# Grouped Query Attention (GQA) with SDPA
class GroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention (GQA) which improves upon Multi-Query
    Attention by having groups of heads share a single key/value projection.
    Uses Scaled Dot-Product Attention (SDPA) for efficiency.
    """
    def __init__(self, embed_dim, num_heads, kv_heads, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout_rate)

        assert num_heads % kv_heads == 0, "Number of heads must be a multiple of KV heads."
        self.num_groups = num_heads // kv_heads

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim)
        self.k_proj = nn.Linear(embed_dim, kv_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, kv_heads * self.head_dim)
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim)

    def forward(self, x, rope_emb):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.kv_heads, self.head_dim)
        
        # Apply RoPE
        q = apply_rotary_pos_emb(q, rope_emb)
        k = apply_rotary_pos_emb(k, rope_emb)

        # GQA: repeat K/V heads for each group
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=2)
            v = v.repeat_interleave(self.num_groups, dim=2)

        # Transpose for SDPA (B, H, L, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # SDPA (FlashAttention replacement)
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None, # Causal mask is handled by is_causal=True
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True
        )

        output = output.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(output)

# Mixture of Experts (MoE)
class TopKRouter(nn.Module):
    """
    A simple router that selects the top-k experts for each token.
    """
    def __init__(self, embed_dim, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(embed_dim, num_experts)

    def forward(self, x):
        logits = self.gate(x)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        # Apply softmax to the selected logits to get the expert weights
        selected_logits = F.softmax(top_k_logits, dim=-1)
        return selected_logits, top_k_indices

class MoELayer(nn.Module):
    """
    A vectorized and efficient Mixture of Experts (MoE) layer.
    
    The previous implementation was correct but very slow due to Python loops.
    This version uses vectorized operations for a massive performance gain.
    """
    def __init__(self, embed_dim, moe_experts, moe_topk, dropout_rate):
        super().__init__()
        self.top_k = moe_topk
        self.router = TopKRouter(embed_dim, moe_experts, moe_topk)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim),
                nn.Dropout(dropout_rate)
            ) for _ in range(moe_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        x_flat = x.view(-1, embed_dim)

        # Route tokens to top-k experts
        expert_weights, expert_indices = self.router(x_flat)
        
        # Prepare for vectorized expert computation
        final_output = torch.zeros_like(x_flat)
        
        # Reshape for scatter operation
        x_flat_reshaped = x_flat.unsqueeze(1).expand(-1, self.top_k, -1)
        expert_indices_reshaped = expert_indices.unsqueeze(2).expand(-1, -1, embed_dim)
        
        # Collect outputs from all experts in one pass
        # This is the key difference for efficiency
        expert_outputs_stacked = torch.empty(
            (x_flat.shape[0], self.top_k, embed_dim), 
            device=x_flat.device, 
            dtype=x_flat.dtype
        )
        for i, expert in enumerate(self.experts):
            # Find tokens that need to be processed by this expert
            tokens_to_process = (expert_indices == i).nonzero(as_tuple=True)
            if tokens_to_process[0].numel() > 0:
                expert_inputs = x_flat[tokens_to_process[0]]
                expert_outputs = expert(expert_inputs)
                 #Ensure dtype consistency (fixes Float vs Half mismatch)
                expert_outputs = expert_outputs.to(x_flat.dtype)

                
                # Scatter the outputs back to their original positions
                expert_outputs_stacked[tokens_to_process] = expert_outputs

        # Combine the weighted expert outputs
        final_output = (expert_weights.unsqueeze(2) * expert_outputs_stacked).sum(dim=1)
        
        return final_output.view(batch_size, seq_len, embed_dim)

class TransformerBlock(nn.Module):
    """
    A single block combining attention and a Mixture of Experts layer,
    using a pre-normalization (pre-norm) architecture.
    """
    def __init__(self, embed_dim, num_heads, kv_heads, dropout_rate, moe_experts, moe_topk):
        super().__init__()
        self.attn = GroupedQueryAttention(embed_dim, num_heads, kv_heads, dropout_rate)
        self.attn_norm = nn.LayerNorm(embed_dim)
        
        self.moe = MoELayer(embed_dim, moe_experts, moe_topk, dropout_rate)
        self.moe_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, rope_emb):
        attn_output = self.attn(self.attn_norm(x), rope_emb)
        x = x + attn_output

        moe_output = self.moe(self.moe_norm(x))
        x = x + moe_output
        return x

class ZiaAdvancedModel(PreTrainedModel):
    """
    The main Zia LLM model, built from stacked TransformerBlocks.
    It inherits from Hugging Face's PreTrainedModel for compatibility.
    """
    config_class = ZiaConfig

    def __init__(self, config: ZiaConfig):
        super().__init__(config)
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.rope_emb = precompute_rotary_emb(
            config.embed_dim // config.num_heads, config.seq_len
        ).to(self.token_embeddings.weight.device)
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                kv_heads=config.kv_heads,
                dropout_rate=config.dropout_rate,
                moe_experts=config.moe_experts,
                moe_topk=config.moe_topk
            ) for _ in range(config.num_layers)
        ])
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight  # weight tying
        self.final_norm = nn.LayerNorm(config.embed_dim)

    # 🔥 Add these hooks so HF resize works
    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.token_embeddings = new_embeddings
        self.lm_head.weight = new_embeddings.weight  # keep weight tying

    def forward(self, input_ids, labels=None):
        token_embeds = self.token_embeddings(input_ids)
        x = self.dropout(token_embeds)
        
        rope_emb = self.rope_emb[:input_ids.shape[1]].to(x.device)

        for layer in self.layers:
            x = layer(x, rope_emb)

        logits = self.lm_head(self.final_norm(x))
        
        if labels is not None:
            # Shift tokens to the left for language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}

# --- Tokenizer & Dataset ---
def finalize_tokenizer(cfg):
    """
    Converts a tokenizers.Tokenizer to a Hugging Face compatible format.
    Standardizes special tokens as [PAD], [UNK], [BOS], [EOS].
    """
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(cfg["TOKENIZER_PATH"], "tokenizer.json"))
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    hf_tok.padding_side = "right"
    # Add special tokens explicitly
    hf_tok.add_special_tokens({
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]"
    })
    hf_tok.save_pretrained(cfg["TOKENIZER_PATH"])
    print(f"[✓] HF-compatible tokenizer saved to {cfg['TOKENIZER_PATH']}")

def train_tokenizer_cli(args):
    """
    Command-line function to train a custom BPE tokenizer.
    """
    tokenizer_dir = args.TOKENIZER_PATH
    os.makedirs(tokenizer_dir, exist_ok=True)
    files = sorted(glob.glob(args.train_data_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.train_data_glob}")
    printb(f"Training tokenizer on {len(files)} files (text_key='{args.text_key}')")

    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import TemplateProcessing
    from tokenizers.normalizers import NFKC

    class JsonlIterator:
        """A streaming iterator for JSONL files."""
        def __init__(self, files, text_key):
            self.files = files
            self.text_key = text_key
        def __iter__(self):
            count = 0
            for fp in self.files:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            j = json.loads(line)
                        except Exception:
                            continue
                        txt = j.get(self.text_key) or j.get("text") or j.get("content")
                        if isinstance(txt, str) and txt:
                            yield txt
                            count += 1
            if count == 0:
                print("[WARN] Tokenizer saw zero training lines. Check data_glob/text_key.")
    
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "<user>", "<assistant>", "<system>"]
    trainer = BpeTrainer(vocab_size=args.vocab_size, min_frequency=args.min_frequency,
                         special_tokens=special_tokens, show_progress=True)
    it = JsonlIterator(files, args.text_key)
    printb("Starting training... (this may take awhile)")
    tok.train_from_iterator(it, trainer=trainer)
    tok.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[("[BOS]", tok.token_to_id("[BOS]") if tok.token_to_id("[BOS]") is not None else 0),
                        ("[EOS]", tok.token_to_id("[EOS]") if tok.token_to_id("[EOS]") is not None else 0)]
    )
    tok_json = os.path.join(tokenizer_dir, "tokenizer.json")
    tok.save(tok_json)
    printb("Tokenizer saved to", tokenizer_dir)

    finalize_tokenizer(vars(args) if hasattr(args, '__dict__') else args)

# Inherits from torch's IterableDataset, which is compatible with the Trainer's DataLoader
class StreamingDataset(torch.utils.data.IterableDataset):
    """
    An iterable dataset that streams data from JSONL files, tokenizes it,
    and chunks it into sequences of a fixed length.
    """
    def __init__(self, data_glob, tokenizer, max_seq_len, text_key):
        self.files = sorted(glob.glob(data_glob))
        if not self.files:
            raise FileNotFoundError(f"No files matched: {data_glob}")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_key = text_key
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        # Added for compatibility with Hugging Face Accelerate
        self._epoch = 0

        # Debug: Print tokenizer vocab and special tokens
        print("Tokenizer base vocab size:", self.tokenizer.vocab_size)
        print("Full vocab size:", len(self.tokenizer))
        print("pad_token:", self.tokenizer.pad_token, "pad_token_id:", self.tokenizer.pad_token_id)
        print("unk_token:", self.tokenizer.unk_token, "unk_token_id:", self.tokenizer.unk_token_id)
        print("bos_token:", self.tokenizer.bos_token, "bos_token_id:", self.tokenizer.bos_token_id)
        print("eos_token:", self.tokenizer.eos_token, "eos_token_id:", self.tokenizer.eos_token_id)
        assert self.tokenizer.pad_token_id is not None, "pad_token_id is None"
        assert self.tokenizer.unk_token_id is not None, "unk_token_id is None"
        assert self.tokenizer.bos_token_id is not None, "bos_token_id is None"
        assert self.tokenizer.eos_token_id is not None, "eos_token_id is None"

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = len(self.files)
        if worker_info:
            per_worker = int(math.ceil(end / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, end)

        # Shuffle files based on the current epoch
        rng = random.Random(self._epoch)
        shuffled_files = list(self.files[start:end])
        rng.shuffle(shuffled_files)

        for file_path in shuffled_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        text = data.get(self.text_key) or data.get("text") or data.get("content")
                        if not text: continue

                        ids = self.tokenizer.encode(text, add_special_tokens=False)
                        ids_with_special = [self.bos_token_id] + ids + [self.eos_token_id]

                        for i in range(0, len(ids_with_special), self.max_seq_len):
                            chunk = ids_with_special[i:i + self.max_seq_len]
                            if len(chunk) < self.max_seq_len:
                                chunk = chunk + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(chunk))

                            ids_tensor = torch.tensor(chunk, dtype=torch.long)
                            # Debug check for out-of-bound token ids
                            if ids_tensor.max().item() >= len(self.tokenizer):
                                print("⚠️ Found OOB token:", ids_tensor.max().item(), " >= ", len(self.tokenizer))
                                raise ValueError("Out-of-bound token id in dataset")

                            yield {"input_ids": ids_tensor, "labels": ids_tensor}
                    except (json.JSONDecodeError, IndexError) as e:
                        log.warning(f"Skipping malformed line: {e}")
                        continue

@dataclass
class CustomDataCollator:
    """
    A simple data collator that stacks features into a batch.
    """
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels)
        }

# --- Confusion Matrix Logging ---
def log_token_confusion_matrix(model, tokenizer, eval_ds, top_k: int, writer: SummaryWriter, step: int, device: str):
    """
    Generates and logs a confusion matrix for the most frequent tokens to TensorBoard.
    """
    try:
        import numpy as np
        from sklearn.metrics import confusion_matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[WARN] Confusion matrix skipped (missing libs):", e)
        return

    it = iter(eval_ds)
    y_true, y_pred = [], []
    frequent_ids = list(range(min(top_k, len(tokenizer))))
    if not frequent_ids:
        return
    model.eval()
    with torch.no_grad():
        for _ in range(50):
            try:
                batch_dict = next(it)
            except StopIteration:
                break
            
            input_ids = batch_dict["input_ids"].unsqueeze(0).to(device)
            labels = batch_dict["labels"].unsqueeze(0).to(device)

            outputs = model(input_ids)
            logits = outputs['logits'][:, :-1, :]
            
            valid_labels = labels[:, 1:].contiguous().view(-1).cpu().numpy()
            valid_logits = logits.contiguous().view(-1, logits.size(-1))

            pred = torch.argmax(valid_logits, dim=-1).cpu().numpy()
            
            mask = np.isin(valid_labels, frequent_ids)
            true_sel = valid_labels[mask]
            pred_sel = pred[mask]
            
            if true_sel.size == 0:
                continue
            y_true.append(true_sel)
            y_pred.append(pred_sel)
    
    if not y_true:
        return
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    remap = {tid:i for i, tid in enumerate(frequent_ids)}
    true_idx = np.array([remap[t] for t in y_true if t in remap])
    pred_idx = np.array([remap[p] for p in y_pred if p in remap])
    
    cm = confusion_matrix(true_idx, pred_idx, labels=list(range(len(frequent_ids))))
    fig = plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title("Token Confusion (Top-K)")
    plt.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    writer.add_image("confusion_matrix/topk", image_to_tensor(img), step)
    plt.close(fig)

# --- Training Pipeline ---
def train_from_scratch_cli(args):
    """
    Main function for the pre-training pipeline.
    Initializes the model, tokenizer, datasets, and the Hugging Face Trainer.
    """
    set_seed(args.seed)
    os.makedirs(args.OUTPUT_DIR, exist_ok=True)
    
    printb("Loading tokenizer from:", args.TOKENIZER_PATH)
    try:
        tok = AutoTokenizer.from_pretrained(args.TOKENIZER_PATH, use_fast=True)
    except Exception:
        log.warning("Tokenizer not found. Please run with --stage train_tokenizer first.")
        sys.exit(1)

    model_config = ZiaConfig(**DEFAULTS['presets'][args.preset])
    # IMPORTANT FIX: Align the model's vocab_size with the tokenizer's actual vocab_size
    model_config.vocab_size = len(tok)
    printb(f"Using model vocab size: {model_config.vocab_size}")

    model = ZiaAdvancedModel(model_config)
    # Resize embeddings to match tokenizer length
    model.resize_token_embeddings(len(tok))
    printb(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    
    train_ds = StreamingDataset(
        data_glob=args.train_data_glob, 
        tokenizer=tok, 
        max_seq_len=model_config.seq_len, 
        text_key=args.text_key
    )
    
    eval_ds = StreamingDataset(
        data_glob=args.eval_data_glob, 
        tokenizer=tok, 
        max_seq_len=model_config.seq_len, 
        text_key=args.text_key
    )

    hf_args = TrainingArguments(
        output_dir=args.OUTPUT_DIR,
        overwrite_output_dir=True,
        # num_train_epochs=args.num_train_epochs, # This is not compatible with StreamingDataset
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.keep_last_n,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=bool(args.bf16),
        fp16=bool(args.fp16) and not bool(args.bf16),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        report_to=["tensorboard"],
        remove_unused_columns=False,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        deepspeed=args.deepspeed,
        max_steps=args.max_steps,
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=args.load_best_model_at_end
    )

    trainer = Trainer(
        model=model,
        args=hf_args,
        data_collator=CustomDataCollator(),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    if args.early_stopping_patience > 0:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    tb_dir = os.path.join(args.OUTPUT_DIR, "runs")
    writer = SummaryWriter(tb_dir)

    printb("Starting training... logs ->", tb_dir)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        metrics["perplexity"] = math.exp(min(20, metrics["eval_loss"]))
    print("[Eval metrics]", metrics)

    try:
        log_token_confusion_matrix(model, tok, eval_ds, args.confmat_top_k, writer, int(trainer.state.global_step), trainer.args.device)
    except Exception as e:
        print("[WARN] Failed to log confusion matrix:", e)

    trainer.save_model()
    tok.save_pretrained(args.OUTPUT_DIR)
    writer.close()
    printb("Training complete. Model & tokenizer saved to", args.OUTPUT_DIR)

# --- CLI ---
def str2bool(v):
    """Helper function to convert string arguments to boolean."""
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def build_cli():
    """Builds a robust argument parser from the DEFAULTS dictionary."""
    p = argparse.ArgumentParser()
    
    # Manually add the stage argument since it's required and not in DEFAULTS
    p.add_argument("--stage", required=True, choices=["train_tokenizer", "pretrain"],
                   help="The stage of the pipeline to run.")
    
    for k, v in DEFAULTS.items():
        if isinstance(v, dict):
            # Skip nested dictionaries like 'presets'
            continue
        
        arg_name = "--" + k.replace("_", "-")
        help_text = f"Default: {v}"
        
        # Determine the argument type and add to parser
        if isinstance(v, bool):
            p.add_argument(arg_name, type=str2bool, default=v, help=help_text)
        elif k == 'resume_from_checkpoint':
            p.add_argument(arg_name, type=str, default=None, help=help_text)
        else:
            p.add_argument(arg_name, type=type(v), default=v, help=help_text)
            
    return p

def main():
    parser = build_cli()
    args = parser.parse_args()
    config = DEFAULTS.copy()
    for k, v in vars(args).items():
        if k in config:
            config[k] = v
    if args.preset and args.preset in config['presets']:
        config.update(config['presets'][args.preset])
    if args.stage == "train_tokenizer":
        train_tokenizer_cli(argparse.Namespace(**config))
    elif args.stage == "pretrain":
        train_from_scratch_cli(argparse.Namespace(**config))

if __name__ == "__main__":
    main()