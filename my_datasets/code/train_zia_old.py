# Finalize and save HuggingFace-compatible tokenizer
from transformers import PreTrainedTokenizerFast

def finalize_tokenizer(cfg):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(cfg["tokenizer_dir"], "tokenizer.json"))

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    hf_tok.save_pretrained(cfg["tokenizer_dir"])
    print(f"[✓] HF-compatible tokenizer saved to {cfg['tokenizer_dir']}")
#!/usr/bin/env python3
"""
train_from_scratch.py

A self-contained, configurable script to:
 - Train a tokenizer (Byte-level BPE)
 - Train a GPT-NeoX-style decoder-only model from scratch on JSONL corpora
 - Includes sensible in-file defaults (edit DEFAULTS) + CLI overrides
 - Checkpointing, tensorboard logging, confusion-matrix (top-K token proxy)
 - Designed to be robust for mixed-language corpora (English, Tamil, Tanglish)
 
Usage (PowerShell example):
python datasets/code/train_from_scratch.py --stage train_tokenizer
python datasets/code/train_from_scratch.py --stage pretrain --preset 0p5B

Edit DEFAULTS below to change behavior without any YAML.
"""

import os, sys, time, glob, json, random, math, io
from typing import Iterator, Optional
from dataclasses import dataclass

import joblib
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets import IterableDataset
from transformers import (
    AutoTokenizer,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# Optional performance libs
try:
    import bitsandbytes as bnb  # noqa: F401
    HAS_BNB = True
except Exception:
    HAS_BNB = False
try:
    from flash_attn import __version__ as _fa_v  # noqa: F401
    HAS_FLASH_ATTN = True
except Exception:
    HAS_FLASH_ATTN = False

# --------------------------
# INTERNAL DEFAULTS (edit here if you prefer)
# --------------------------
DEFAULTS = {
    "data_glob": "datasets/organized/*chunk_*.jsonl",
    "text_key": "text",          # key in JSONL; many of our files use "text" or "content"
    "tokenizer_dir": "artifacts/tokenizer_32k",
    "vocab_size": 32000,
    "min_frequency": 2,
    "seq_len": 2048,
    "preset": "0p5B",            # tiny, 0p5B, 1B
    "output_dir": "artifacts/aria-0p5B",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 64,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_ratio": 0.02,
    "lr_scheduler_type": "cosine",
    "logging_steps": 50,
    "eval_steps": 1000,
    "save_steps": 1000,
    "keep_last_n": 3,
    "dataloader_num_workers": 8,
    "early_stopping_patience": 5,
    "max_eval_samples": 50000,
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    "use_8bit_optimizer": False,
    "flash_attn": True,
    "compile": False,
    "max_grad_norm": 1.0,
    "confmat_top_k": 64,
    "seed": 42,
}

# --------------------------
# Utilities
# --------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def image_to_tensor(img: np.ndarray):
    # returns CHW
    return np.transpose(img, (2,0,1))

def printb(*a, **k):
    print("[i]", *a, **k)

# --------------------------
# Tokenizer training (Byte-level BPE recommended)
# --------------------------
def train_tokenizer_cli(args):
    # args is a simple namespace-like object (we accept dict too)
    tokenizer_dir = args.tokenizer_dir
    os.makedirs(tokenizer_dir, exist_ok=True)

    files = sorted(glob.glob(args.data_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.data_glob}")
    printb(f"Training tokenizer on {len(files)} files (text_key='{args.text_key}')")

    # Local import to avoid heavy deps on runs that won't do tokenization
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import TemplateProcessing
    from tokenizers.normalizers import NFKC

    class JsonlIterator:
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
                        txt = j.get(args.text_key) or j.get("text") or j.get("content")
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
    # Post-processor to add BOS/EOS tokens when encoding single strings
    tok.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[("[BOS]", tok.token_to_id("[BOS]") if tok.token_to_id("[BOS]") is not None else 0),
                        ("[EOS]", tok.token_to_id("[EOS]") if tok.token_to_id("[EOS]") is not None else 0)]
    )
    tok_json = os.path.join(tokenizer_dir, "tokenizer.json")
    tok.save(tok_json)
    printb("Tokenizer saved to", tokenizer_dir)

    # Finalize into HuggingFace format
    finalize_tokenizer(vars(args) if hasattr(args, '__dict__') else args)

# --------------------------
# Dataset streaming & tokenization helpers
# --------------------------

def build_iterable_dataset(data_glob: str, text_key: str):
    files = sorted(glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"No files match {data_glob}")
    def gen():
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                        txt = j.get(text_key) or j.get("text") or j.get("content")
                        if isinstance(txt, str) and txt.strip():
                            yield {"text": txt}
                    except Exception:
                        continue
    return IterableDataset.from_generator(gen)

def tokenize_streaming(dataset, tokenizer, seq_len: int, num_proc: int):
    # Convert stream of free text into fixed-length token chunks (seq_len)
    bos = tokenizer.bos_token or "[BOS]"
    eos = tokenizer.eos_token or "[EOS]"

    def tok_fn(ex):
        ids = tokenizer(ex["text"], add_special_tokens=False, truncation=False)["input_ids"]
        chunk = [tokenizer.convert_tokens_to_ids(bos)] + ids + [tokenizer.convert_tokens_to_ids(eos)]
        out = []
        for i in range(0, len(chunk) - 1, seq_len):
            x = chunk[i : i + seq_len]
            if len(x) < seq_len:
                x = x + [tokenizer.pad_token_id] * (seq_len - len(x))
            out.append({"input_ids": x})
        return {"chunks": out}

    def explode(batch):
        rows = []
        for lst in batch["chunks"]:
            rows.extend(lst)
        return {"input_ids": [r["input_ids"] for r in rows]}

    ds = dataset.map(tok_fn, batched=False)
    ds = ds.map(explode, batched=True, batch_size=64)
    return ds

# --------------------------
# Confusion matrix logging (token-level proxy)
# --------------------------
def log_token_confusion_matrix(model, tokenizer, eval_ds, top_k: int, writer: SummaryWriter, step: int, device: str):
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
                batch = [next(it) for _ in range(8)]
            except StopIteration:
                break
            input_ids = torch.tensor([ex["input_ids"] for ex in batch], device=device)
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = input_ids.cpu().numpy()
            # collect only positions where true token in frequent_ids
            mask = np.isin(labels, frequent_ids)
            true_sel = labels[mask]
            pred_sel = pred[mask]
            if true_sel.size == 0: 
                continue
            y_true.append(true_sel)
            y_pred.append(pred_sel)
    if not y_true:
        return
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    # remap and compute cm
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

# --------------------------
# Model config presets
# --------------------------
def build_config(vocab_size: int, seq_len: int, preset: str, flash_attn: bool) -> GPTNeoXConfig:
    presets = {
        "0p5B": dict(hidden_size=1536, num_hidden_layers=24, num_attention_heads=12),
        "1B":   dict(hidden_size=2048, num_hidden_layers=24, num_attention_heads=16),
        "tiny": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12),
    }
    if preset not in presets:
        raise ValueError("Unknown preset: " + preset)
    base = presets[preset]
    attn_impl = "flash_attention_2" if (flash_attn and HAS_FLASH_ATTN) else "eager"
    cfg = GPTNeoXConfig(
        vocab_size=vocab_size,
        hidden_size=base["hidden_size"],
        num_hidden_layers=base["num_hidden_layers"],
        num_attention_heads=base["num_attention_heads"],
        intermediate_size=base["hidden_size"] * 4,
        max_position_embeddings=seq_len,
        tie_word_embeddings=False,
        use_cache=False,
        attn_implementation=attn_impl,
    )
    return cfg

# --------------------------
# Training pipeline
# --------------------------
def train_from_scratch_cli(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    printb("Loading tokenizer from:", args.tokenizer_dir)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token":"[PAD]"})
    if tok.bos_token is None:
        tok.add_special_tokens({"bos_token":"[BOS]"})
    if tok.eos_token is None:
        tok.add_special_tokens({"eos_token":"[EOS]"})

    # Build model from scratch (no checkpoint)
    cfg = build_config(vocab_size=len(tok), seq_len=args.seq_len, preset=args.preset, flash_attn=args.flash_attn)
    model = GPTNeoXForCausalLM(cfg)
    model.resize_token_embeddings(len(tok))

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    raw_stream = build_iterable_dataset(args.data_glob, args.text_key)
    token_stream = tokenize_streaming(raw_stream, tok, args.seq_len, args.dataloader_num_workers)

    # simple eval splitter: 1% by default
    eval_every = max(20, int(100 / max(1, args.eval_fraction*100)))
    token_stream = token_stream.enumerate().map(lambda ex: {**ex[1], **{"is_eval": (ex[0] % eval_every == 0)}})
    train_ds = token_stream.filter(lambda ex: not ex["is_eval"]).remove_columns(["is_eval"])
    eval_ds = token_stream.filter(lambda ex: ex["is_eval"]).remove_columns(["is_eval"])

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # Joblib meta for bookkeeping
    meta = {"vocab_size": len(tok), "seq_len": args.seq_len, "preset": args.preset, "files": sorted(glob.glob(args.data_glob)), "start_time": time.time()}
    joblib.dump(meta, os.path.join(args.output_dir, "run_meta.joblib"))

    hf_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
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
    )

    trainer = Trainer(
        model=model,
        args=hf_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    if args.early_stopping_patience > 0:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    tb_dir = os.path.join(args.output_dir, "runs")
    writer = SummaryWriter(tb_dir)

    printb("Starting training... logs ->", tb_dir)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # eval & perplexity
    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        metrics["perplexity"] = math.exp(min(20, metrics["eval_loss"]))
    print("[Eval metrics]", metrics)

    # confusion matrix
    try:
        log_token_confusion_matrix(model, tok, eval_ds, args.confmat_top_k, writer, int(trainer.state.global_step), trainer.args.device)
    except Exception as e:
        print("[WARN] Failed to log confusion matrix:", e)

    trainer.save_model()
    tok.save_pretrained(args.output_dir)
    writer.close()
    printb("Training complete. Model & tokenizer saved to", args.output_dir)

# --------------------------
# CLI (internal defaults from DEFAULTS)
# --------------------------
def build_cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["train_tokenizer","pretrain"])
    # load defaults from DEFAULTS
    for k,v in DEFAULTS.items():
        arg_name = "--" + k.replace("_","-")
        # infer type
        if isinstance(v,bool):
            p.add_argument(arg_name, type=str, default=str(v))
        else:
            p.add_argument(arg_name, type=type(v), default=v)
    # keep eval_fraction separate
    p.add_argument("--eval-fraction", type=float, default=0.01)
    # resume
    p.add_argument("--resume-from-checkpoint", type=str, default=None)
    return p

def str2bool(v):
    return str(v).lower() in ("1","true","t","yes","y")

def main():
    p = build_cli()
    ns = p.parse_args()
    # convert namespace to simple args object (and cast bools)
    class A: pass
    args = A()
    for k,v in vars(ns).items():
        key = k.replace("-","_")
        if isinstance(DEFAULTS.get(key, None), bool):
            setattr(args, key, str2bool(v))
        else:
            setattr(args, key, v)
    # keep eval_fraction
    if hasattr(ns, "eval_fraction"):
        args.eval_fraction = ns.eval_fraction
    else:
        args.eval_fraction = DEFAULTS["eval_fraction"]

    if args.stage == "train_tokenizer":
        train_tokenizer_cli(args)
    elif args.stage == "pretrain":
        train_from_scratch_cli(args)
    else:
        raise RuntimeError("Unknown stage")

if __name__ == "__main__":
    main()
