#!/usr/bin/env python3
"""
train_realign_v7.py — Long-context ZIA continuation training (resume-safe)
----------------------------------------------------------
✅ Step-based eval (~5–10 min per val)
✅ Auto-resume from last shard/context
✅ Step counter resets cleanly on resume (for new logs)
✅ Skips already trained shards — no retraining from start
✅ Lightweight validation subset from unseen shard
✅ Grad accumulation + checkpointing preserved
"""

import os, glob, time, random, re, torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_from_disk
from tqdm import tqdm

# --- environment
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ==================== CONFIG ====================

@dataclass
class Config:
    tokenizer: str = "artifacts/zia_tokenizer_60k"
    data_glob: str = "artifacts/processed/wiki_tokenized/wiki_chunk_*/"
    output: str = "artifacts/zia_ift_v4_longctx"

    # model
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1

    # optimization
    lr: float = 5e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    grad_accum: int = 80
    eval_every_steps: int = 1000
    autosave_minutes: int = 25

    # context schedule
    progressive_steps = [1024, 4096, 8192]

    # runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True
    seed: int = 42
    resume: bool = True
    num_workers: int = 0
    val_subset_size: int = 400


# ==================== HELPERS ====================

def set_seed(s=42):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def latest_checkpoint(folder):
    ckpts = sorted(glob.glob(os.path.join(folder, "checkpoints", "checkpoint_auto_step*.pt")), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None

def extract_step_from_ckpath(path):
    m = re.search(r"step(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0

@torch.no_grad()
def validate(model, val_loader, device, fp16):
    model.eval()
    total_loss, n = 0.0, 0
    for ids, lbls in val_loader:
        ids, lbls = ids.to(device), lbls.to(device)
        with torch.amp.autocast(device_type="cuda", enabled=fp16):
            _, loss = model(ids, lbls)
        total_loss += float(loss.item())
        n += 1
    model.train()
    return total_loss / max(1, n)


# ==================== MODEL ====================

class FeedForward(nn.Module):
    def __init__(self, d_model, mlp_ratio, dropout):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mlp_ratio, dropout)
    def forward(self, x):
        h = self.ln1(x)
        T = h.size(1)
        mask = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + out
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model, n_layers, n_heads, mlp_ratio, max_len, dropout, pad_token_id):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.max_len = max_len
        self.pad_token_id = pad_token_id

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        T = min(T, self.max_len)
        input_ids = input_ids[:, :T]
        pos_ids = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.tok(input_ids) + self.pos(pos_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100
            )
        return logits, loss


# ==================== DATA ====================

class WikiShard(Dataset):
    def __init__(self, path):
        self.ds = load_from_disk(path)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        it = self.ds[int(i)]
        return it["input_ids"], it["labels"]

class Collate:
    def __init__(self, pad_id): self.pad_id = pad_id
    def __call__(self, batch):
        maxlen = max(len(x[0]) for x in batch)
        ids = [b[0] + [self.pad_id]*(maxlen - len(b[0])) for b in batch]
        lbls = [b[1] + [-100]*(maxlen - len(b[1])) for b in batch]
        return torch.tensor(ids), torch.tensor(lbls)


# ==================== TRAIN LOOP ====================

def train():
    cfg = Config()
    set_seed(cfg.seed)
    ensure_dir(cfg.output)
    ensure_dir(os.path.join(cfg.output, "checkpoints"))

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    vocab = len(tok)

    shards = sorted(glob.glob(cfg.data_glob))
    if not shards:
        raise RuntimeError(f"No shards found: {cfg.data_glob}")

    val_shard = shards[min(len(shards)-1, 60)] if len(shards) > 60 else shards[-1]
    train_shards = [s for s in shards if s != val_shard]
    val_ds_full = WikiShard(val_shard)

    model = TinyGPT(vocab, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio,
                    cfg.progressive_steps[0], cfg.dropout, tok.pad_token_id).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optim, 500, 20000)
    scaler = torch.amp.GradScaler(enabled=cfg.fp16)
    collate = Collate(tok.pad_token_id)

    # --- Resume logic ---
    resume_state = {"ctx_stage": 0, "last_shard_idx": 0, "best_val": float("inf")}
    best_ckpath = os.path.join(cfg.output, "checkpoint_best.pt")

    if cfg.resume and os.path.exists(best_ckpath):
        ck = torch.load(best_ckpath, map_location="cpu")
        model.load_state_dict(ck["model"], strict=False)
        if "optim" in ck: optim.load_state_dict(ck["optim"])
        if "scheduler" in ck: scheduler.load_state_dict(ck["scheduler"])
        if "scaler" in ck:
            try: scaler.load_state_dict(ck["scaler"])
            except: pass
        resume_state["best_val"] = ck.get("best_val", float("inf"))
        resume_state["ctx_stage"] = ck.get("ctx_stage", 0)
        resume_state["last_shard_idx"] = ck.get("last_shard_idx", 0)
        print(f"[i] Resumed: ctx_stage={resume_state['ctx_stage']} | last shard={resume_state['last_shard_idx']}")
    else:
        print("[i] Fresh training start")

    best_val = resume_state["best_val"]
    ctx_stage = resume_state["ctx_stage"]
    start_shard = resume_state["last_shard_idx"]
    step, accum_step_counter = 0, 0
    last_save_time = time.time()

    print(f"[i] Training shards={len(train_shards)} | Val shard={os.path.basename(val_shard)}")

    # --- Training Loop ---
    for stage_i, ctx in enumerate(cfg.progressive_steps[ctx_stage:], start=ctx_stage):
        # expand context if needed
        if ctx > model.max_len:
            old = model.pos.weight.data
            new = torch.zeros((ctx, old.size(1)), dtype=old.dtype)
            new[:old.size(0)] = old
            new[old.size(0):] = old[-1].repeat(ctx - old.size(0), 1)
            model.pos = nn.Embedding(ctx, old.size(1)).to(cfg.device)
            model.pos.weight.data.copy_(new)
            model.max_len = ctx
            print(f"[i] Expanded positional embedding → {ctx}")

        for shard_idx, shard_path in enumerate(train_shards[start_shard:], start=start_shard):
            ds = WikiShard(shard_path)
            dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate)
            prog = tqdm(dl, desc=f"[ctx={ctx}] {os.path.basename(shard_path)}", leave=False)
            batch_counter = 0

            for ids, lbls in prog:
                ids, lbls = ids.to(cfg.device), lbls.to(cfg.device)
                with torch.amp.autocast(device_type="cuda", enabled=cfg.fp16):
                    _, loss = model(ids, lbls)
                    loss = loss / cfg.grad_accum
                scaler.scale(loss).backward()
                batch_counter += 1

                if batch_counter % cfg.grad_accum == 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)
                    scheduler.step()
                    step += 1
                    accum_step_counter += 1

                    # --- Step-based evaluation ---
                    if step > 0 and step % cfg.eval_every_steps == 0:
                        subset_size = min(cfg.val_subset_size, len(val_ds_full))
                        start_idx = random.randint(0, max(0, len(val_ds_full) - subset_size))
                        subset = Subset(val_ds_full, list(range(start_idx, start_idx + subset_size)))
                        val_loader = DataLoader(subset, batch_size=cfg.batch_size, shuffle=False,
                                                num_workers=0, pin_memory=True, collate_fn=collate)
                        val_loss = validate(model, val_loader, cfg.device, cfg.fp16)
                        if val_loss < best_val:
                            best_val = val_loss
                            ck = {
                                "model": model.state_dict(),
                                "optim": optim.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "scaler": scaler.state_dict(),
                                "ctx_stage": stage_i,
                                "last_shard_idx": shard_idx,
                                "best_val": best_val,
                                "cfg": asdict(cfg),
                            }
                            atomic_save(ck, best_ckpath)
                            print(f"[VAL] step={step} | new best val_loss={best_val:.4f}")

                    # --- Autosave ---
                    if time.time() - last_save_time > cfg.autosave_minutes * 60:
                        ckpath = os.path.join(cfg.output, "checkpoints", f"checkpoint_auto_step{step}.pt")
                        ck = {
                            "model": model.state_dict(),
                            "optim": optim.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict(),
                            "ctx_stage": stage_i,
                            "last_shard_idx": shard_idx,
                            "best_val": best_val,
                            "cfg": asdict(cfg),
                        }
                        atomic_save(ck, ckpath)
                        last_save_time = time.time()

            prog.close()

        start_shard = 0  # after finishing resumed shards, reset to full list for next ctx expansion

    print("[✓] Training complete.")


if __name__ == "__main__":
    train()
