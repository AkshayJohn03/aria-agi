#!/usr/bin/env python3
"""
realign_retrain.py

Refined alignment fine-tuning for ZIA.

- Loads a TinyGPT checkpoint (e.g., zia_ift_v4_cursor best/latest)
- Loads the cleaned dataset (tokenized with the same 60k tokenizer)
- Fine-tunes to realign instruction→response behavior
- FP16 enabled, 30-min auto checkpoints (keeps 2 latest + best)
- Adjustable via TrainingConfig

Usage:
  python my_datasets/code/realign_retrain.py
  or override params, e.g.:
  python my_datasets/code/realign_retrain.py --batch_size 4 --grad_accum 8
"""

import os, glob, argparse, math, time
from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_from_disk

# ----------------------------- CONFIG -----------------------------
@dataclass
class TrainingConfig:
    ckpt: str = "artifacts/zia_ift_v4_cursor/best_val/checkpoint.pt"
    tokenizer: str = "artifacts/zia_tokenizer_60k"
    dataset: str = "artifacts/tokenized_dataset/zia_ift_v3_cleaned"
    output: str = "artifacts/zia_ift_v4_realign"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 2e-5
    epochs: int = 1
    max_len: int = 1024
    fp16_mode: str = "force"  # auto / force / off
    eval_every_steps: int = 400
    ckpt_interval_sec: int = 1800  # auto-save every 30 mins
    num_workers: int = 0           # keep 0 for Windows to avoid pickle issues

# ----------------------------- MODEL -----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, mlp_ratio, dropout):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mlp_ratio, dropout)
    def forward(self, x, key_padding_mask=None):
        h = self.ln1(x)
        T = h.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(h, h, h, attn_mask=causal_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + out
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model=384, n_layers=8, n_heads=6, mlp_ratio=4, max_len=1024, dropout=0.1, pad_token_id=0):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.max_len = max_len
        self.pad_token_id = pad_token_id

    def forward(self, input_ids, attention_mask=None, labels=None):
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
            x = blk(x, key_padding_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), ignore_index=-100)
        return logits, loss

# ----------------------------- HELPERS -----------------------------
def collate_fn(batch, pad_token_id, tokenizer):
    max_len = max(len(x['input_ids']) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        ids = x['input_ids']
        pad_needed = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_needed)
        attention_mask.append([1] * len(ids) + [0] * pad_needed)
        labels.append([*ids, *([-100] * pad_needed)])
    return (torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long))

@torch.no_grad()
def evaluate(model, dl, device, max_batches=200):
    model.eval()
    tot, n = 0.0, 0
    for i, (ids, mask, labels) in enumerate(dl):
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        _, loss = model(ids, attention_mask=mask, labels=labels)
        tot += loss.item()
        n += 1
        if n >= max_batches: break
    model.train()
    return tot / max(1, n)

@torch.no_grad()
def sample_generate(student, tokenizer, input_ids, device, max_length=100, temperature=0.3, top_k=20, top_p=0.8):
    student.eval()
    generated = input_ids.clone()
    for _ in range(max_length):
        logits, _ = student(generated)
        next_token_logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(next_token_logits, top_k)
            min_v = v[:, -1].unsqueeze(-1)
            next_token_logits[next_token_logits < min_v] = -float("inf")
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break
    return generated

def expand_pos_and_vocab(state_dict, model, tokenizer):
    sd = model.state_dict()
    if "pos.weight" in state_dict and "pos.weight" in sd:
        src, dst = state_dict["pos.weight"], sd["pos.weight"]
        if dst.size(0) > src.size(0):
            last = src[-1:].repeat(dst.size(0) - src.size(0), 1)
            state_dict["pos.weight"] = torch.cat([src, last], dim=0)
        else:
            state_dict["pos.weight"] = src[:dst.size(0)]
    for key in ("tok.weight", "head.weight"):
        if key in state_dict and key in sd:
            src, dst = state_dict[key], sd[key]
            dst_copy = dst.clone()
            n_copy = min(src.size(0), dst.size(0))
            dst_copy[:n_copy] = src[:n_copy]
            if dst.size(0) > src.size(0):
                nn.init.normal_(dst_copy[src.size(0):], std=0.02)
            state_dict[key] = dst_copy
    return state_dict

def find_latest_checkpoint(folder):
    if os.path.isdir(folder) and os.path.exists(os.path.join(folder, "checkpoint.pt")):
        return os.path.join(folder, "checkpoint.pt")
    if os.path.isdir(folder):
        cands = sorted(glob.glob(os.path.join(folder, "checkpoints", "step_*.pt")))
        return cands[-1] if cands else None
    if os.path.isfile(folder):
        return folder
    return None

# ----------------------------- MAIN -----------------------------
def main():
    cfg = TrainingConfig()
    parser = argparse.ArgumentParser()
    for k, v in vars(cfg).items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    print(f"[i] tokenizer vocab_size={len(tok)} pad={tok.pad_token_id}")

    ds = load_from_disk(args.dataset)
    train_ds = ds["train"]
    val_ds = ds.get("validation", None)
    print(f"[i] train={len(train_ds)} val={len(val_ds) if val_ds else 0}")

    device = torch.device(args.device)
    model = TinyGPT(vocab=len(tok), max_len=args.max_len).to(device)

    ckpt_path = find_latest_checkpoint(args.ckpt)
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck.get("model", ck)
    state = expand_pos_and_vocab(state, model, tok)
    model.load_state_dict({k: v for k, v in state.items() if k in model.state_dict()}, strict=False)
    print("[✓] model loaded")

    allow_fp16 = (args.fp16_mode == "force") or (args.fp16_mode == "auto" and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=allow_fp16)
    print(f"[i] fp16 enabled: {allow_fp16}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, tok.pad_token_id, tok),
                          num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, tok.pad_token_id, tok),
                        num_workers=args.num_workers, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(len(train_dl) * args.epochs / args.grad_accum)
    sched = get_linear_schedule_with_warmup(opt, 50, total_steps)
    print(f"[i] total_steps_est={total_steps}")

    # --- Training ---
    best_val = float("inf")
    step = 0
    micro = 0
    running_loss = 0.0
    last_ckpt_time = time.time()
    recent_ckpts = []

    canonical_prompts = [
        "Explain the importance of democracy.",
        "Why do humans need medicines?",
        "Describe artificial intelligence in a sentence."
    ]

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for ids, mask, labels in pbar:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=allow_fp16):
                _, loss = model(ids, attention_mask=mask, labels=labels)
                if loss is None:
                    continue
                loss = loss / args.grad_accum

            if allow_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro += 1
            running_loss += loss.item() * args.grad_accum

            if micro % args.grad_accum == 0:
                if allow_fp16:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()

                step += 1
                avg_loss = running_loss / args.grad_accum
                pbar.set_postfix({"step": step, "loss": f"{avg_loss:.4f}"})
                running_loss = 0.0

                # Auto checkpoint every 30 mins
                if time.time() - last_ckpt_time > args.ckpt_interval_sec:
                    os.makedirs(args.output, exist_ok=True)
                    ckpt_path = os.path.join(args.output, f"checkpoint_auto_step{step}.pt")
                    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                                "step": step, "metric": best_val}, ckpt_path)
                    recent_ckpts.append(ckpt_path)
                    print(f"[💾] Auto-saved checkpoint -> {ckpt_path}")
                    last_ckpt_time = time.time()
                    if len(recent_ckpts) > 2:
                        old_ckpt = recent_ckpts.pop(0)
                        try:
                            os.remove(old_ckpt)
                            print(f"[🧹] Removed old checkpoint: {old_ckpt}")
                        except OSError:
                            pass

                # Evaluate every N steps
                if step % args.eval_every_steps == 0:
                    val_loss = evaluate(model, val_dl, device, max_batches=200)
                    print(f"\n[Eval] step={step} val_loss={val_loss:.4f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        os.makedirs(args.output, exist_ok=True)
                        torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                    "step": step, "metric": best_val},
                                   os.path.join(args.output, "checkpoint_best.pt"))
                        print(f"[🏆] Saved best -> {args.output}/checkpoint_best.pt")

                    # sample generation
                    print("[GEN] Sample generations:")
                    for p in canonical_prompts:
                        inp = f"Instruction: {p}\nResponse:"
                        enc = tok(inp, return_tensors="pt").to(device)
                        gen = sample_generate(model, tok, enc["input_ids"], device)
                        text = tok.decode(gen[0], skip_special_tokens=True)
                        out = text.split("Response:", 1)[-1].strip() if "Response:" in text else text
                        print(f"Q: {p}\nA: {out}\n")

    # final save
    os.makedirs(args.output, exist_ok=True)
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "step": step, "metric": best_val},
               os.path.join(args.output, "checkpoint_final.pt"))
    print("[✓] Training finished, saved final checkpoint.")


if __name__ == "__main__":
    main()
