#!/usr/bin/env python3
"""
train_realign_v16_8k_expand_stable.py

- Resume-safe TinyGPT continuation
- Resets optimizer + scaler on resume (model only loaded)
- Cosine LR decay scheduler
- Safe positional-expansion from smaller ckpt (4k -> 8k)
- Time-based checkpointing & exact batch resume
- Visible tqdm with loss + lr
"""

import os, glob, time, random, itertools, torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

# ---------------- Environment ----------------
if "TRANSFORMERS_CACHE" in os.environ:
    del os.environ["TRANSFORMERS_CACHE"]
os.environ.setdefault("HF_HOME", os.path.abspath("artifacts/hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("artifacts/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.abspath("artifacts/hf_metrics_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

# ---------------- Config ----------------
tokenizer_path = "artifacts/zia_tokenizer_60k"
data_glob = "artifacts/processed/wiki_tokenized/wiki_chunk_*/"
output_dir = "artifacts/zia_ift_v4_longctx"
resume_ckpt = os.path.join(output_dir, "checkpoints", "checkpoint_auto_step5432.pt")  # change if needed

# model hyperparams
d_model, n_layers, n_heads, mlp_ratio, dropout = 384, 8, 6, 4, 0.1

# training hyperparams; adjust as you like
desired_context_len = 8192    # target context for this run (will expand if ckpt smaller)
lr = 5e-5  #8.0e-05                  # lower LR for stable fine-tuning after expansion
weight_decay = 0.01
batch_size = 4
grad_accum = 80
autosave_minutes = 30
num_shards = 6                # random shards per run
num_workers = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
fp16 = True
seed = 42

# scheduler params
scheduler_tmax = 5000         # number of optimizer steps for cosine period
scheduler_eta_min = 3e-6

# ---------------- Helpers ----------------
def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
set_seed(seed)

def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def infer_ckpt_context(path):
    """Return context length inside checkpoint if available (pos.weight rows)."""
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location="cpu")
    mdl = ck.get("model", {})
    pos = mdl.get("pos.weight", None)
    if pos is not None:
        return pos.shape[0]
    return None

def expand_pos_embedding(model, new_ctx):
    """Expand positional embeddings safely (copy + extend)."""
    old = model.pos.weight.data
    old_n, dim = old.shape
    if new_ctx <= old_n:
        # nothing to do
        return
    new = torch.zeros((new_ctx, dim), device=old.device, dtype=old.dtype)
    new[:old_n] = old
    last = old[-1:].clone()
    diff = new_ctx - old_n
    noise = (torch.linspace(0, 1e-5, diff, device=old.device).unsqueeze(1) *
             (torch.arange(dim, device=old.device).float().unsqueeze(0) % 7 - 3))
    new[old_n:] = last.repeat(diff, 1) + noise
    model.pos = nn.Embedding(new_ctx, dim).to(device)
    model.pos.weight.data.copy_(new)
    model.max_len = new_ctx
    tqdm.write(f"[i] Expanded positional embedding {old_n} → {new_ctx}")

# ---------------- Model ----------------
class FeedForward(nn.Module):
    def __init__(self, d, m, drop):
        super().__init__()
        hidden = d * m
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
            nn.Dropout(drop)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, d, h, m, drop):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ff = FeedForward(d, m, drop)
    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         attn_mask=mask, need_weights=False)
        return x + a + self.ff(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self, vocab, d, l, h, m, max_len, drop, pad):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, h, m, drop) for _ in range(l)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.max_len = max_len
        self.pad_token_id = pad

    def forward(self, ids, lbls=None):
        T = min(ids.size(1), self.max_len)
        ids = ids[:, :T]
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device).unsqueeze(0))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if lbls is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                lbls[:, 1:T].reshape(-1),
                ignore_index=-100
            )
        return logits, loss

# ---------------- Data ----------------
class WikiShard(torch.utils.data.Dataset):
    def __init__(self, path): self.ds = load_from_disk(path)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        it = self.ds[int(i)]
        return it["input_ids"], it["labels"]

class Collate:
    def __init__(self, pad): self.pad = pad
    def __call__(self, batch):
        maxlen = max(len(x[0]) for x in batch)
        ids = [b[0] + [self.pad]*(maxlen - len(b[0])) for b in batch]
        lbl = [b[1] + [-100]*(maxlen - len(b[1])) for b in batch]
        return torch.tensor(ids), torch.tensor(lbl)

# ---------------- Train ----------------
def train():
    print(f"[i] Device: {device} | FP16: {fp16}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    vocab = len(tok)

    # shards selection (exclude problematic ones; adjust suffixes if different)
    all_shards = sorted(glob.glob(data_glob))
    # exclude 000 and validation shards 061/062 (you used earlier)
    train_pool = [s for s in all_shards if not s.rstrip('/').endswith(("000","061","062","059"))]
    if len(train_pool) == 0:
        raise RuntimeError(f"No training shards found at {data_glob}")
    selected_shards = random.sample(train_pool, min(num_shards, len(train_pool)))
    tqdm.write(f"[i] Found {len(train_pool)} shards | Using {len(selected_shards)} random shards")

    # infer checkpoint context if exists
    ckpt_ctx = infer_ckpt_context(resume_ckpt)
    base_ctx = ckpt_ctx if ckpt_ctx is not None else min(1024, desired_context_len)

    # If checkpoint context is larger than desired target, adapt target up to ckpt
    target_ctx = desired_context_len
    if ckpt_ctx is not None and ckpt_ctx > desired_context_len:
        tqdm.write(f"[i] Checkpoint context ({ckpt_ctx}) > desired ({desired_context_len}), using {ckpt_ctx} to avoid shrink.")
        target_ctx = ckpt_ctx

    model = TinyGPT(vocab, d_model, n_layers, n_heads, mlp_ratio, base_ctx, dropout, tok.pad_token_id).to(device)

    # --- load model weights only, keep optimizer/scaler fresh (reset) ---
    step, resume_shard, resume_batch = 0, 0, 0
    if os.path.exists(resume_ckpt):
        tqdm.write(f"[i] Loading checkpoint (model only): {resume_ckpt}")
        ck = torch.load(resume_ckpt, map_location="cpu")
        model.load_state_dict(ck["model"], strict=False)
        step = ck.get("step", 0)
        resume_shard = ck.get("current_shard_idx", 0)
        resume_batch = ck.get("current_batch_idx", 0)
        tqdm.write(f"[i] Resumed step={step} | shard={resume_shard} | batch={resume_batch}")
    else:
        tqdm.write("[i] No checkpoint found — starting new training.")

    # expand pos embeddings only if required (safe)
    if target_ctx > base_ctx:
        expand_pos_embedding(model, target_ctx)

    # reset optimizer and scaler (intentional)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=fp16)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=scheduler_tmax, eta_min=scheduler_eta_min)
    tqdm.write("[i] Optimizer & scaler initialized (model-only resume). Cosine scheduler attached.")

    collate = Collate(tok.pad_token_id)
    last_save = time.time()
    tqdm.write(f"[i] Context={target_ctx} | LR={lr:.2e} | GradAccum={grad_accum}")

    # training loop across selected shards (resume support)
    for shard_idx, shard in enumerate(selected_shards):
        if shard_idx < resume_shard:
            tqdm.write(f"[i] Skipping shard {shard_idx} (already completed in ckpt)")
            continue

        ds = WikiShard(shard)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=False,  # set False if your PyTorch doesn't like persistent workers
            prefetch_factor=2,
            collate_fn=collate
        )

        total_batches = len(dl)
        prog = tqdm(dl, total=total_batches,
                    desc=f"[shard {shard_idx+1}/{len(selected_shards)}]",
                    leave=True, dynamic_ncols=True)
        acc = 0

        for batch_idx, (ids, lbls) in enumerate(prog):
            # skip until resume batch when resuming inside this shard
            if shard_idx == resume_shard and batch_idx < resume_batch:
                continue

            ids, lbls = ids.to(device), lbls.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=fp16):
                _, loss = model(ids, lbls)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            acc += 1

            if acc % grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1

                # update tqdm postfix: show last loss + lr
                try:
                    current_lr = scheduler.get_last_lr()[0]
                except Exception:
                    current_lr = optim.param_groups[0]["lr"]
                prog.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}", "step": step})

            # time-based checkpointing (save model + resume indices; optimizer intentionally omitted)
            if time.time() - last_save > autosave_minutes * 60:
                ckpath = os.path.join(output_dir, "checkpoints", f"checkpoint_auto_step{step}.pt")
                atomic_save({
                    "model": model.state_dict(),
                    "step": step,
                    "current_shard_idx": shard_idx,
                    "current_batch_idx": batch_idx,
                    "context_len": target_ctx
                }, ckpath)
                last_save = time.time()
                tqdm.write(f"[💾] Saved step={step}, shard={shard_idx}, batch={batch_idx}")

        prog.close()
        tqdm.write(f"[✓] Shard {shard_idx+1}/{len(selected_shards)} complete | step={step}")

    # final save (model only)
    ck_final = os.path.join(output_dir, "checkpoints", f"checkpoint_auto_step{step}_final.pt")
    atomic_save({
        "model": model.state_dict(),
        "step": step,
        "context_len": target_ctx
    }, ck_final)
    tqdm.write(f"[✓] Training complete | Final step={step}")

if __name__ == "__main__":
    train()
