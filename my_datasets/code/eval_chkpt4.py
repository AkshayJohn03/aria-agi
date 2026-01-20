#!/usr/bin/env python3
"""
eval_fast.py
Efficient evaluation harness:
 - evaluates only a chosen set of checkpoints (best N, last M, or a list)
 - uses fp16/no_grad and batched eval
 - supports sampling per shard to limit eval time
"""
import os, glob, time, torch
import math
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from datasets import load_from_disk

CHECKPOINT_DIR = "artifacts/zia_ift_fixed4k/checkpoints"
TOKENIZED_VAL_DIR = "artifacts/tokenized_datasets/zia_ift_v3/validation"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
MAX_BATCHES_PER_CHECKPOINT = 200   # limit per checkpoint for speed
CHECKPOINT_SELECTION = "best_n"    # options: "best_n", "last_n", "list"
BEST_N = 3

def list_checkpoints(mode="best_n", n=3):
    # If you keep "best_val" ckpt at known place, add manually
    all_ck = sorted(glob.glob(os.path.join(CHECKPOINT_DIR,"checkpoint*.pt")), key=os.path.getmtime)
    if mode=="last_n":
        return all_ck[-n:]
    if mode=="best_n":
        # find best_val if saved as checkpoint_best.pt
        best = os.path.join(os.path.dirname(CHECKPOINT_DIR),"checkpoint_best.pt")
        res = []
        if os.path.exists(best):
            res.append(best)
        # add last (n-1)
        res += all_ck[-(n-1):] if len(all_ck) >= (n-1) else all_ck
        return res
    return all_ck

@torch.no_grad()
def evaluate_ckpt(ckpt_path, model, val_loader, max_batches=None):
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"], strict=False)
    model.to(DEVICE).eval()
    total_loss=0.0; count=0
    for i, (ids, lbls) in enumerate(tqdm(val_loader, desc=f"Eval {os.path.basename(ckpt_path)}")):
        ids = ids.to(DEVICE); lbls = lbls.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=True if DEVICE.startswith("cuda") else False):
            logits, loss = model(ids, lbls)
        if not torch.isfinite(loss):
            continue
        total_loss += float(loss.item())
        count += 1
        if max_batches and count >= max_batches: break
    avg_loss = (total_loss / max(1,count))
    ppl = math.exp(min(avg_loss, 50))
    return avg_loss, ppl

def main():
    # load a lightweight model shell for the same architecture
    # You can import your real model builder; for demo we assume it exists as build_model()
    from train_realign_v11_memopt import TinyGPT
    # simple tokenizer read to get vocab/ pad - quick approach
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(); sp.load("artifacts/zia_tokenizer_v2_clean/zia_spm.model")
    vocab = sp.get_piece_size()
    pad_id = 0
    model = TinyGPT(vocab, 384, 8, 6, 4, 4096, 0.1, pad_id)

    # load tokenized validation dataset (or a prebuilt tensor dataset)
    # We expect a folder with tokenized pt shards OR use HF arrow -> collate into ids,labels
    # For safety, use small subset sampling to avoid long runs
    val_ds = load_from_disk(TOKENIZED_VAL_DIR)
    val_ds = val_ds.select(range(min(len(val_ds), 2000)))  # sample top 2000 items
    # create DataLoader quickly
    def collate(batch):
        maxlen = max(len(x["input_ids"]) for x in batch)
        ids = [x["input_ids"] + [pad_id]*(maxlen-len(x["input_ids"])) for x in batch]
        labels = [x["labels"] + [-100]*(maxlen-len(x["labels"])) for x in batch]
        return torch.tensor(ids,dtype=torch.long), torch.tensor(labels,dtype=torch.long)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=True)

    checkpoints = list_checkpoints(CHECKPOINT_SELECTION, BEST_N)
    results = {}
    for ck in checkpoints:
        print("[i] Evaluating:", ck)
        avg_loss, ppl = evaluate_ckpt(ck, model, val_loader, max_batches=MAX_BATCHES_PER_CHECKPOINT)
        results[ck] = {"loss": avg_loss, "ppl": ppl}
        print(f"[i] {os.path.basename(ck)} -> loss {avg_loss:.4f} ppl {ppl:.2f}")

    print("[✓] Eval done. Summary:")
    for k,v in results.items():
        print(k, v)

if __name__=="__main__":
    main()
