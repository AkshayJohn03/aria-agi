# train_dialogue_finetune_freeze.py
import os, glob, time, random, math
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
from my_model_rope import TinyGPT_RoPE

# ---------------- Config ----------------
tokenizer_path = "artifacts/zia_tokenizer_60k"
data_dir = "artifacts/dialogue_finetune"  # expected to contain shards or a HuggingFace dataset dir
resume_ckpt = "artifacts/zia_ift_v4_longctx_realign_rope/checkpoint_realign_rope_step600.pt"  # output of re-align
output_dir = "artifacts/zia_dialogue_finetune"
device = "cuda" if torch.cuda.is_available() else "cpu"
fp16 = True
d_model = 384
n_layers = 8
n_heads = 6
mlp_ratio = 4
max_len = 2048
batch_size = 4
grad_accum = 16
num_steps = 3000
lr = 2e-5
weight_decay = 0.01
eval_every = 500
freeze_first_n = 4   # freeze lower 4 layers
num_workers = 4
seed = 42

os.makedirs(output_dir, exist_ok=True)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def load_data(path):
    # try load dataset dir / shards; if not exists, raise
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dialogue dataset not found: {path}")
    try:
        ds = load_from_disk(path)
        return ds
    except Exception:
        # maybe shards inside
        subs = sorted([os.path.join(path, d) for d in os.listdir(path)])
        shards = [s for s in subs if os.path.isdir(s)]
        if shards:
            from datasets import concatenate_datasets
            arr = [load_from_disk(s) for s in shards]
            return concatenate_datasets(arr)
    raise RuntimeError("Could not load dataset from path")

class CollateDialogue:
    def __init__(self, tokenizer, max_len):
        self.tok = tokenizer
        self.max_len = max_len
    def __call__(self, batch):
        texts = []
        for ex in batch:
            if isinstance(ex, dict):
                # try messages -> flatten
                if "messages" in ex and isinstance(ex["messages"], list):
                    parts = []
                    for m in ex["messages"]:
                        role = (m.get("role","") + ": ").strip() if isinstance(m, dict) else ""
                        content = m.get("content") if isinstance(m, dict) else (m if isinstance(m,str) else "")
                        parts.append(f"{role}{content}")
                    texts.append("\n".join(parts))
                    continue
                if "text" in ex:
                    texts.append(ex["text"])
                    continue
                if "content" in ex:
                    texts.append(ex["content"])
                    continue
            texts.append(str(ex))
        enc = self.tok(texts, truncation=True, padding=True, max_length=self.max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=200):
    model.eval()
    total_loss = 0.0
    count = 0
    for i,(ids,mask) in enumerate(val_loader,1):
        ids = ids.to(device)
        _, loss = model(ids, labels=ids)
        total_loss += loss.item()
        count += 1
        if max_batches and i >= max_batches:
            break
    model.train()
    return total_loss / max(1,count)

def train():
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token":"<pad>"})
    vocab = len(tok)

    # model load
    model = TinyGPT_RoPE(vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads, mlp_ratio=mlp_ratio, max_len=max_len, dropout=0.1, pad_token_id=tok.pad_token_id)
    model = model.to(device)

    if os.path.exists(resume_ckpt):
        ck = torch.load(resume_ckpt, map_location="cpu")
        model.load_state_dict(ck.get("model", ck), strict=False)
        print("[i] loaded re-align checkpoint (partial).")

    # Freeze lower layers and token embedding
    for p in model.tok.parameters():
        p.requires_grad = False
    for i in range(freeze_first_n):
        for p in model.blocks[i].parameters():
            p.requires_grad = False
    print(f"[i] froze token embedding and first {freeze_first_n} blocks.")

    train_ds = load_data(os.path.join(data_dir, "train"))
    val_ds = load_data(os.path.join(data_dir, "validation")) if os.path.exists(os.path.join(data_dir, "validation")) else None

    collate = CollateDialogue(tok, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=min(4,num_workers), collate_fn=collate) if val_ds is not None else None

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, num_steps // 10), eta_min=1e-6)

    step = 0
    micro = 0
    raw_loss = 0.0
    best_val = float("inf")

    pbar = tqdm(total=num_steps, desc="dialogue-ft", dynamic_ncols=True)
    try:
        while step < num_steps:
            for ids,mask in train_loader:
                ids = ids.to(device)
                with torch.cuda.amp.autocast(enabled=fp16):
                    _, loss = model(ids, labels=ids)
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("[!] Bad loss, skipping batch")
                        continue
                    raw_loss += loss.item()
                    loss = loss / grad_accum
                scaler.scale(loss).backward()
                micro += 1
                if micro % grad_accum == 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)
                    scheduler.step()
                    step += 1
                    avg_loss = raw_loss / (grad_accum)
                    pbar.update(1)
                    pbar.set_postfix({"step": step, "loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
                    raw_loss = 0.0

                    # eval
                    if val_loader is not None and (step % eval_every == 0):
                        val_loss = evaluate(model, val_loader, device, max_batches=200)
                        print(f"[Eval] step={step} val_loss={val_loss:.4f}")
                        if val_loss < best_val:
                            best_val = val_loss
                            atomic_save({"model": model.state_dict(), "step": step, "best_val": best_val}, os.path.join(output_dir, "best_val.pt"))
                            print("[🏆] saved best_val")
                if step >= num_steps:
                    break
            if step >= num_steps:
                break
    except KeyboardInterrupt:
        print("[!] Interrupted - saving")
        atomic_save({"model": model.state_dict(), "step": step}, os.path.join(output_dir, f"interrupt_step{step}.pt"))
    finally:
        pbar.close()
        atomic_save({"model": model.state_dict(), "step": step}, os.path.join(output_dir, f"final_step{step}.pt"))
        print("[✓] training finished")

if __name__ == "__main__":
    train()
