# kd_live_distill.py
import os
import time
import argparse
import random
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

# optional libs
try:
    import bitsandbytes as bnb
    BNB = True
except Exception:
    BNB = False

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT = True
except Exception:
    PEFT = False

def join_messages(messages):
    # same simple join you used earlier
    parts = []
    for m in messages:
        c = (m.get("content") or "").strip()
        if c:
            parts.append(c)
    return "\n".join(parts)

def sample_prompts_from_dataset(ds, n):
    # ds is a HF Dataset (assumes features['messages'])
    idxs = random.sample(range(len(ds)), k=n)
    texts = []
    for i in idxs:
        ex = ds[i]
        texts.append(join_messages(ex["messages"]))
    return texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, default="bigscience/bloom-1b1")
    parser.add_argument("--student", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--dataset", type=str, default="datasets/processed/arrow_dataset")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher_on_cpu", action="store_true", help="Force teacher to CPU to save GPU memory")
    parser.add_argument("--use_8bit", action="store_true", help="Load student in 8-bit (requires bitsandbytes)")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    print("[i] options:", args)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print("[i] using device:", device)

    # load dataset
    print("[i] loading dataset:", args.dataset)
    ds = load_from_disk(args.dataset)
    train_ds = ds["train"]
    print(f"[i] train len: {len(train_ds)}")

    # tokenizer: use teacher tokenizer to avoid mismatch, we'll resize student embeddings
    print("[i] loading tokenizer from teacher:", args.teacher)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.model_max_length = args.seq_len

    # load teacher (CPU by default)
    print("[i] loading teacher:", args.teacher)
    teacher_device = "cpu" if args.teacher_on_cpu or device.type == "cpu" else device
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    teacher.to(teacher_device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # load student
    print("[i] loading student:", args.student)
    if args.use_8bit and BNB:
        # 8-bit + prepare for kbit training
        print("[i] Loading student in 8-bit (bnb) and preparing for LoRA")
        student = AutoModelForCausalLM.from_pretrained(
            args.student,
            load_in_8bit=True,
            device_map="auto"
        )
        if PEFT:
            student = prepare_model_for_kbit_training(student)
    else:
        student = AutoModelForCausalLM.from_pretrained(args.student)
        student.to(device)

    # ensure tokenizer size matches student embeddings
    try:
        student.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        print("[!] resize_token_embeddings failed:", e)

    # apply LoRA
    if not PEFT:
        raise RuntimeError("peft is required for efficient LoRA training. pip install peft")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "proj"],  # best-effort targets; may vary by model
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    student = get_peft_model(student, lora_config)
    student.print_trainable_parameters()

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr)
    total_steps = args.steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*total_steps), num_training_steps=total_steps)

    student.train()
    step = 0
    start_time = time.time()
    T = 2.0  # temperature for KD; tune as needed
    alpha_kd = 1.0  # weight for KD loss

    while step < args.steps:
        # sample a batch of prompts
        texts = sample_prompts_from_dataset(train_ds, args.batch_size)
        enc = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=args.seq_len).to(device)
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        # teacher forward (no grad) on teacher_device; move inputs appropriately
        teacher_in = {k: v.to(teacher_device) for k, v in enc.items()}
        with torch.no_grad():
            teacher_out = teacher(**teacher_in, return_dict=True)
            teacher_logits = teacher_out.logits.detach().to(device)  # bring back to student device

        # student forward
        student_out = student(**enc, return_dict=True)
        student_logits = student_out.logits  # on device

        # shift: predict next token
        t_logits = teacher_logits[:, :-1, :].contiguous()
        s_logits = student_logits[:, :-1, :].contiguous()

        # KD loss (KL divergence between teacher prob and student log-prob)
        s_log_probs = F.log_softmax(s_logits / T, dim=-1)
        t_probs = F.softmax(t_logits / T, dim=-1)
        kd_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)

        # Optional: also include CE to teacher argmax (hard labels)
        hard_labels = torch.argmax(t_logits, dim=-1)
        ce_loss = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), hard_labels.view(-1), ignore_index=tokenizer.pad_token_id)

        loss = alpha_kd * kd_loss + 0.1 * ce_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        step += 1
        if step % 5 == 0:
            elapsed = time.time() - start_time
            print(f"[step {step}/{args.steps}] loss={loss.item():.4f} (kd={kd_loss.item():.4f}, ce={ce_loss.item():.4f}) elapsed={elapsed:.1f}s")

        if step % args.save_every == 0:
            ckpt_dir = f"artifacts/online_kd_ckpt/step_{step:06d}"
            os.makedirs(ckpt_dir, exist_ok=True)
            # save only peft adapter weights
            student.save_pretrained(ckpt_dir)
            print(f"[i] saved LoRA adapters to {ckpt_dir}")

    print("[i] training finished")

if __name__ == "__main__":
    main()
