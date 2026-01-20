#!/usr/bin/env python3
"""
train_and_kd_full.py

Usage examples:
# 1) Continue / resume base training
python train_and_kd_full.py --phase base_train --student_base_dir artifacts/zia_dense_runs --tokenizer_path artifacts/zia_tokenizer_60k --dataset my_datasets/processed/arrow_dataset --epochs 1 --batch_size 8

# 2) Run KD after base training (teacher: HuggingFace model)
python train_and_kd_full.py --phase distill --teacher bigscience/bloom-1b1 --student_base_dir artifacts/zia_dense_runs --tokenizer_path artifacts/zia_tokenizer_60k --dataset my_datasets/processed/arrow_dataset --steps 10000 --batch_size 2

"""

import os
import sys
import time
import json
import random
import argparse
import shutil
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

# Make sure project root is importable (adjust if needed)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import your TinyGPT implementation
try:
    from train_zia_dense import TinyGPT
except Exception:
    # fallback to package path if using modules layout
    from datasets.code.train_zia_dense import TinyGPT

# -------------------------
# Utilities
# -------------------------
def now(): return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def save_ckpt(path: str, state: dict, tmp_name="checkpoint.pt.tmp"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)

def find_latest_ckpt_dir(out_dir: str) -> Optional[str]:
    if not os.path.isdir(out_dir):
        return None
    dirs = [os.path.join(out_dir, d) for d in os.listdir(out_dir) if d.startswith("step_") or d=="best_val"]
    if not dirs: return None
    dirs = [d for d in dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, "checkpoint.pt"))]
    if not dirs: return None
    dirs.sort(key=lambda p: os.path.getmtime(os.path.join(p, "checkpoint.pt")), reverse=True)
    return os.path.join(dirs[0], "checkpoint.pt")

def sample_from_topk_top_p(probs: torch.Tensor, top_k:int=50, top_p:float=0.9) -> int:
    """
    probs: 1D tensor of probabilities (already temperature & softmaxed)
    returns selected token id (python int)
    """
    # convert to CPU numpy
    probs = probs.detach().cpu()
    # Top-k filter
    if top_k is not None and top_k > 0:
        topk_vals, topk_idx = torch.topk(probs, min(top_k, probs.size(-1)))
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask[topk_idx] = False
        probs = probs.clone()
        probs[mask] = 0.0
    # Top-p (nucleus)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        # mask tokens beyond cumulative > top_p
        cutoff = torch.where(cumulative > top_p)[0]
        if cutoff.numel()>0:
            cutoff_idx = cutoff[0].item()
            # zero out tokens after cutoff_idx
            indices_to_zero = sorted_idx[cutoff_idx+1:]
            probs[indices_to_zero] = 0.0
    # Renormalize and sample
    total = probs.sum().item()
    if total <= 0 or not torch.isfinite(torch.tensor(total)):
        # fallback to uniform over nonzero
        nonzero = (probs > 0).nonzero(as_tuple=False)
        if nonzero.numel() == 0:
            return int(torch.randint(0, probs.size(-1), (1,)).item())
        idx = int(nonzero[torch.randint(0, nonzero.size(0), (1,))].item())
        return idx
    probs = probs / probs.sum()
    idx = torch.multinomial(probs, num_samples=1).item()
    return idx

def sample_generate_custom(model, tokenizer, input_ids: torch.LongTensor, max_new_tokens=64,
                           temperature=1.0, top_k=50, top_p=0.9, device="cpu"):
    """
    Custom next-token sampling loop for TinyGPT.
    input_ids: (1, seq_len)
    returns generated token ids (1, seq_len + new)
    """
    model.eval()
    generated = input_ids.to(device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(generated, attention_mask=None)  # logits shape (B, T, V)
        last_logits = logits[:, -1, :].squeeze(0)  # (V,)
        # temperature and softmax
        if temperature != 1.0:
            last_logits = last_logits / (temperature + 1e-9)
        probs = F.softmax(last_logits, dim=-1)
        # sample
        next_token = sample_from_topk_top_p(probs, top_k=top_k, top_p=top_p)
        next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        generated = torch.cat([generated, next_token_tensor], dim=1)
        # break on eos
        if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
            break
    return generated

# -------------------------
# Model/Tokenizer builders
# -------------------------
def build_student(tokenizer_path: str, device: torch.device, defaults: dict, base_ckpt_path: Optional[str]=None):
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    vocab_size = len(tok)
    print(f"[i] Tokenizer loaded: vocab_size={vocab_size} pad_id={tok.pad_token_id} eos={tok.eos_token_id}")

    student = TinyGPT(
        vocab_size=vocab_size,
        d_model=defaults.get("d_model", 384),
        n_layers=defaults.get("n_layers", 8),
        n_heads=defaults.get("n_heads", 6),
        mlp_ratio=defaults.get("mlp_ratio", 4),
        max_len=defaults.get("max_len", 256),
        dropout=defaults.get("dropout", 0.1),
    ).to(device)

    resume_step = 0
    if base_ckpt_path and os.path.exists(base_ckpt_path):
        print(f"[i] Loading student base checkpoint from {base_ckpt_path}")
        ck = torch.load(base_ckpt_path, map_location=device)
        # support both formats (model_state_dict or model)
        if "model_state_dict" in ck:
            student.load_state_dict(ck["model_state_dict"], strict=False)
            resume_step = ck.get("step", 0)
        elif "model" in ck:
            student.load_state_dict(ck["model"], strict=False)
            resume_step = ck.get("step", 0)
        else:
            # older-style saving whole state
            try:
                student.load_state_dict(ck, strict=False)
            except Exception as e:
                print("[!] Could not interpret checkpoint format:", e)
    return student, tok, resume_step

def build_teacher(teacher_name: str, tokenizer: AutoTokenizer, device: torch.device, teacher_on_cpu: bool = False):
    """
    Build teacher model and ensure vocab is compatible with tokenizer.
    If teacher vocab != tokenizer vocab, we resize teacher embeddings to tokenizer vocab size.
    """
    t_device = torch.device("cpu") if teacher_on_cpu or device.type == "cpu" else device
    print(f"[i] Loading teacher '{teacher_name}' -> device {t_device}")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        low_cpu_mem_usage=True,
        device_map=None,
        torch_dtype=torch.float32 if t_device.type == "cpu" else torch.float16,
    )
    # If tokenizer vocab differs from teacher config, resize teacher embeddings to match tokenizer
    tok_vocab = len(tokenizer)
    teacher_vocab = teacher.get_input_embeddings().num_embeddings
    if teacher_vocab != tok_vocab:
        print(f"[!] Teacher vocab ({teacher_vocab}) != tokenizer vocab ({tok_vocab}). Resizing teacher embeddings to match tokenizer.")
        teacher.resize_token_embeddings(tok_vocab)
    teacher.to(t_device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher, t_device

# -------------------------
# Dataset utils
# -------------------------
def build_dataloaders(dataset_path: str, tokenizer: AutoTokenizer, seq_len: int, batch_size: int, num_workers: int = 0):
    # We assume dataset is HF dataset on disk with 'train' and 'test' splits and each example has 'messages'.
    ds = load_from_disk(dataset_path)
    train_ds = ds["train"]
    val_ds = ds.get("test") or ds.get("validation") or None

    def collate_fn(batch):
        texts = []
        for ex in batch:
            # join messages into single string
            msgs = ex.get("messages") or []
            parts = []
            for m in msgs:
                c = (m.get("content") or "").strip()
                if c:
                    parts.append(c)
            texts.append("\n".join(parts))
        enc = tokenizer(texts, truncation=True, padding=True, max_length=seq_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, drop_last=True)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False)
    return train_loader, val_loader

# -------------------------
# Training / Val loops
# -------------------------
@torch.no_grad()
def evaluate_student(student, dataloader, device, max_batches: int = 200):
    student.eval()
    losses = []
    n = 0
    for i, (input_ids, attn) in enumerate(dataloader, 1):
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        logits = student(input_ids, attn)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0)
        losses.append(loss.item())
        n += 1
        if max_batches and n >= max_batches:
            break
    student.train()
    return float(sum(losses) / max(1, len(losses)))

def train_base(student, tokenizer, train_loader, val_loader, device, out_dir, epochs=1, lr=3e-4, accum_steps=1,
               save_every_steps=500, keep_last=2, max_val_batches=200):
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb", now()))
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    total_steps = (len(train_loader) * epochs) // accum_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.03*total_steps), num_training_steps=total_steps)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    global_step = 0
    best_val = float("inf")
    step_since_save = 0

    for epoch in range(1, epochs+1):
        pbar = tqdm(train_loader, desc=f"BaseTrain epoch {epoch}/{epochs}")
        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (input_ids, attn) in enumerate(pbar, 1):
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            with autocast(enabled=(device.type == "cuda")):
                logits = student(input_ids, attn)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=tokenizer.pad_token_id)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
            accum_loss += loss.item()
            if (batch_idx) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                pbar.set_postfix(step=global_step, loss=f"{accum_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                # checkpointing
                if global_step % save_every_steps == 0:
                    ckdir = os.path.join(out_dir, f"step_{global_step:06d}")
                    ckpath = os.path.join(ckdir, "checkpoint.pt")
                    os.makedirs(ckdir, exist_ok=True)
                    state = {"model_state_dict": student.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(), "step": global_step}
                    save_ckpt(ckpath, state)
                    # cleanup keep_last
                    ckpts = sorted([d for d in os.listdir(out_dir) if d.startswith("step_")], reverse=True)
                    for old in ckpts[keep_last:]:
                        try: shutil.rmtree(os.path.join(out_dir, old))
                        except: pass
                    print(f"\n[i] Saved base checkpoint -> {ckpath}")
                # validation
                if val_loader is not None and global_step % (save_every_steps) == 0:
                    val_loss = evaluate_student(student, val_loader, device, max_batches=max_val_batches)
                    writer.add_scalar("val/loss", val_loss, global_step)
                    print(f"\n[✓] Step {global_step} | Val loss: {val_loss:.4f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        best_path = os.path.join(out_dir, "best_val")
                        os.makedirs(best_path, exist_ok=True)
                        save_ckpt(os.path.join(best_path, "checkpoint.pt"), {"model_state_dict": student.state_dict(), "step": global_step})
                        print(f"[🏆] New best val saved @ step {global_step}")
                accum_loss = 0.0
                step_since_save = 0
    # final save
    final_path = os.path.join(out_dir, f"step_final_{now()}")
    os.makedirs(final_path, exist_ok=True)
    save_ckpt(os.path.join(final_path, "checkpoint.pt"), {"model_state_dict": student.state_dict(), "step": global_step})
    writer.close()
    print("[i] Base training complete.")

def distill_training(student, tokenizer, teacher_name: str, teacher_on_cpu: bool, train_loader, val_loader,
                     device, out_dir, steps=10000, batch_size=2, grad_accum=4, lr=2e-4, seq_len=256,
                     save_every=500, alpha_kd=1.0, T=2.0, keep_last=2):
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb", now()))
    # Build teacher and ensure tokenization alignment
    teacher, teacher_device = build_teacher(teacher_name, tokenizer, device, teacher_on_cpu)
    # optimizer/scheduler/scaler
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.03*steps), num_training_steps=steps)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    student.train()
    global_step = 0
    pbar = tqdm(range(steps), desc="Distillation")
    for step in pbar:
        # sample a small batch of prompts
        batch_texts = []
        # Simple reservoir sampling from dataloader (we have train_loader)
        try:
            # get one batch from train_loader iterator
            batch = next(iter(train_loader))
            input_ids_batch, attn_batch = batch
            # convert to text via tokenizer.decode for each row
            for row in input_ids_batch:
                txt = tokenizer.decode(row.tolist(), skip_special_tokens=True)
                batch_texts.append(txt)
        except Exception:
            # fallback: random picks
            batch_texts = []
            for _ in range(batch_size):
                # get a random doc from dataset via sampling
                idx = random.randint(0, len(train_loader.dataset)-1) if hasattr(train_loader.dataset, "__len__") else 0
                # best effort: pick empty
                batch_texts.append("")

        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=seq_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # feed teacher (on teacher_device)
        teacher_inputs = {"input_ids": enc["input_ids"].to(teacher_device), "attention_mask": enc["attention_mask"].to(teacher_device)}
        with torch.no_grad():
            teacher_out = teacher(**teacher_inputs, return_dict=True)
            teacher_logits = teacher_out.logits  # (B, L, V_tok)
            # move to student device
            teacher_logits = teacher_logits.to(device)

        # student forward and KD loss
        with autocast(enabled=(device.type == "cuda")):
            student_logits = student(input_ids, attention_mask)  # (B, L, V_student)
            # align vocab dims (teacher logits may be bigger or equal)
            v_s = student_logits.size(-1)
            v_t = teacher_logits.size(-1)
            if v_t != v_s:
                # slice teacher logits/pad student if needed
                if v_t >= v_s:
                    t_logits = teacher_logits[..., :v_s].contiguous()
                else:
                    # teacher vocab smaller -> pad teacher logits with -inf for extra tokens
                    pad = torch.full((*teacher_logits.size()[:-1], v_s - v_t), -1e9, device=device, dtype=teacher_logits.dtype)
                    t_logits = torch.cat([teacher_logits, pad], dim=-1)
            else:
                t_logits = teacher_logits

            # align seq length - take min L
            L = min(student_logits.size(1), t_logits.size(1))
            s_logits = student_logits[:, :L, :].contiguous()
            t_logits = t_logits[:, :L, :].contiguous()

            # shift for next token prediction
            s_logits = s_logits[:, :-1, :].contiguous()
            t_logits = t_logits[:, :-1, :].contiguous()

            s_log_probs = F.log_softmax(s_logits / T, dim=-1)
            t_probs = F.softmax(t_logits / T, dim=-1)
            kd_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)
            hard_labels = torch.argmax(t_logits, dim=-1)
            ce_loss = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), hard_labels.view(-1), ignore_index=tokenizer.pad_token_id)
            loss = (alpha_kd * kd_loss + 0.1 * ce_loss) / float(grad_accum)

        scaler.scale(loss).backward()
        if (step + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        # logging / sample decode every N steps
        if (step % 50) == 0:
            try:
                # deterministic sample from a short prompt
                prompt = "Hello Zia!"
                encp = tokenizer(prompt, return_tensors="pt").to(device)
                gen_ids = sample_generate_custom(student, tokenizer, encp["input_ids"], max_new_tokens=32, temperature=0.9, top_k=50, top_p=0.9, device=device)
                decoded = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
                print(f"\n[i] Sample @ step {step} -> {decoded}")
                if writer is not None:
                    writer.add_text("sample/generated", decoded, step)
            except Exception as e:
                print("[!] sample generation failed:", e)

        # tensorboard writes
        try:
            if writer is not None:
                writer.add_scalar("loss/total", (alpha_kd * kd_loss + 0.1 * ce_loss).item(), step)
                writer.add_scalar("loss/kd", kd_loss.item(), step)
                writer.add_scalar("loss/ce", ce_loss.item(), step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)
        except Exception:
            pass

        # periodic checkpoint save
        if (step + 1) % save_every == 0 or (step + 1) == steps:
            ckdir = os.path.join(out_dir, f"step_{step+1:06d}")
            os.makedirs(ckdir, exist_ok=True)
            state = {"model_state_dict": student.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(), "step": step+1}
            save_ckpt(os.path.join(ckdir, "checkpoint.pt"), state)
            # cleanup old
            ckpts = sorted([d for d in os.listdir(out_dir) if d.startswith("step_")], reverse=True)
            for old in ckpts[keep_last:]:
                try: shutil.rmtree(os.path.join(out_dir, old))
                except: pass
            print(f"\n[i] Saved KD checkpoint -> {ckdir}")

    writer.close()
    print("[i] Distillation finished.")

# -------------------------
# CLI / main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["base_train", "distill"], required=True)
    parser.add_argument("--student_base_dir", type=str, default="artifacts/zia_dense_runs")
    parser.add_argument("--tokenizer_path", type=str, default="artifacts/zia_tokenizer_60k")
    parser.add_argument("--dataset", type=str, default="my_datasets/processed/arrow_dataset")
    parser.add_argument("--teacher", type=str, default="bigscience/bloom-1b1")
    parser.add_argument("--teacher_on_cpu", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[i] Running on device: {device} | CUDA available: {torch.cuda.is_available()}")

    # defaults for model
    defaults = {"d_model": 384, "n_layers": 8, "n_heads": 6, "mlp_ratio": 4, "max_len": args.seq_len, "dropout": 0.1}

    # Build student model and tokenizer
    student_ckpt = find_latest_ckpt_dir(os.path.join(args.student_base_dir))  # e.g. artifacts/zia_dense_runs/best_val or latest step
    student, tokenizer, resume_step = build_student(args.tokenizer_path, device, defaults, base_ckpt_path=student_ckpt)
    print(f"[i] Student params: {sum(p.numel() for p in student.parameters()):,}")

    # Quick tokenizer + model sanity checks
    # round-trip examples
    print("[i] Tokenizer sample:", tokenizer.decode(tokenizer("Hello world!")["input_ids"]))
    # sample model forward shape test
    sample_ids = tokenizer("Hello Zia!", return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        logits = student(sample_ids, None)
    print("[i] Forward logits shape:", logits.shape)

    # Build dataset loaders
    train_loader, val_loader = build_dataloaders(args.dataset, tokenizer, args.seq_len, args.batch_size, num_workers=args.num_workers)
    print(f"[i] train size: {len(train_loader.dataset)}")

    if args.phase == "base_train":
        out_dir = args.student_base_dir
        train_base(student, tokenizer, train_loader, val_loader, device, out_dir, epochs=args.epochs,
                   lr=args.lr, accum_steps=args.grad_accum_steps, save_every_steps=args.save_every)
    else:
        # phase distill uses teacher
        distill_out = os.path.join(args.student_base_dir, "distillation_checkpoints")
        os.makedirs(distill_out, exist_ok=True)
        distill_training(student, tokenizer, args.teacher, args.teacher_on_cpu, train_loader, val_loader,
                         device, distill_out, steps=args.steps, batch_size=args.batch_size,
                         grad_accum=args.grad_accum_steps, lr=args.lr, seq_len=args.seq_len,
                         save_every=args.save_every)

if __name__ == "__main__":
    main()
