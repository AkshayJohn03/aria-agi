# train_realign_rope.py
import os, glob, random, time, math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer
from my_model_rope import TinyGPT_RoPE

# ---------------- Config ----------------
tokenizer_path = "artifacts/zia_tokenizer_60k"
data_glob = "artifacts/processed/wiki_tokenized/wiki_chunk_*/"  # same as your pipeline
output_dir = "artifacts/zia_ift_v4_longctx_realign_rope"
resume_ckpt = "artifacts/zia_ift_v4_longctx/checkpoints/checkpoint_auto_step5255.pt"  # your 4k good checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
fp16 = True
d_model = 384
n_layers = 8
n_heads = 6
mlp_ratio = 4
target_max_len = 8192    # RoPE supports high lengths; set to 8k for re-align
batch_size = 4
grad_accum = 32
realign_steps = 600
lr = 2e-5
weight_decay = 0.01
num_shards = 4
num_workers = 4
seed = 42

os.makedirs(output_dir, exist_ok=True)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def load_shards(glob_pat, exclude_suffixes=("000","061","062","059")):
    all_shards = sorted(glob.glob(glob_pat))
    pool = [s for s in all_shards if not s.rstrip("/").endswith(exclude_suffixes)]
    if not pool:
        raise RuntimeError("No shards found")
    return pool

class WikiShard(torch.utils.data.Dataset):
    def __init__(self, path):
        self.ds = load_from_disk(path)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        it = self.ds[int(i)]
        return torch.tensor(it["input_ids"], dtype=torch.long), torch.tensor(it["labels"], dtype=torch.long)

def collate_pad(batch, pad_id):
    maxlen = max(x[0].shape[0] for x in batch)
    ids = []
    lbls = []
    for a,b in batch:
        padlen = maxlen - a.shape[0]
        ids.append(torch.cat([a, torch.full((padlen,), pad_id, dtype=torch.long)]))
        lbls.append(torch.cat([b, torch.full((padlen,), -100, dtype=torch.long)]))
    return torch.stack(ids, dim=0), torch.stack(lbls, dim=0)

def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def train():
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    vocab = len(tok)

    pool = load_shards(data_glob)
    selected = random.sample(pool, min(num_shards, len(pool)))
    print(f"[i] selected shards: {selected}")

    # model init
    model = TinyGPT_RoPE(vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads, mlp_ratio=mlp_ratio, max_len=target_max_len, dropout=0.1, pad_token_id=tok.pad_token_id)
    model = model.to(device)

    # load existing checkpoint weights (model-only)
    if os.path.exists(resume_ckpt):
        ck = torch.load(resume_ckpt, map_location="cpu")
        print(f"[i] Loading checkpoint (model only) {resume_ckpt}")
        model.load_state_dict(ck.get("model", ck), strict=False)
        print("[i] weights loaded (partial load allowed).")

    # Freeze token embeddings (important) & lower layers optionally
    for param in model.tok.parameters():
        param.requires_grad = False
    # Optionally freeze lower half of blocks to preserve base features
    freeze_lower = True
    n_freeze = n_layers // 2 if freeze_lower else 0
    for i in range(n_freeze):
        for p in model.blocks[i].parameters():
            p.requires_grad = False
    print(f"[i] frozen token embedding and first {n_freeze} blocks.")

    # optimizer (only trainable params)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, realign_steps), eta_min=1e-6)

    last_save = time.time()
    step = 0
    micro_acc = 0
    total_micro = re_align_micro = re_align_steps = realign_steps * grad_accum

    # DataLoaders
    collate = lambda b: collate_pad(b, tok.pad_token_id)
    shards = selected
    for shard_idx, shard in enumerate(shards):
        ds = WikiShard(shard)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
        pbar = tqdm(dl, desc=f"shard {shard_idx+1}/{len(shards)}", dynamic_ncols=True)
        for (ids, lbls) in pbar:
            ids = ids.to(device)
            lbls = lbls.to(device)
            with torch.cuda.amp.autocast(enabled=fp16):
                _, loss = model(ids, labels=lbls)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            micro_acc += 1
            if micro_acc % grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                pbar.set_postfix({"step": step, "loss": f"{(loss.item()*grad_accum):.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
            # short-run stop condition
            if step >= realign_steps:
                break
        pbar.close()
        if step >= realign_steps:
            break

    # final save
    ckpath = os.path.join(output_dir, f"checkpoint_realign_rope_step{step}.pt")
    atomic_save({"model": model.state_dict(), "step": step, "context_len": target_max_len}, ckpath)
    print(f"[✓] Saved re-align checkpoint: {ckpath}")

if __name__ == "__main__":
    train()
