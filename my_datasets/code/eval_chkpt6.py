#!/usr/bin/env python3
import os, torch, math
from pathlib import Path
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB = 60011
D_MODEL = 384
LAYERS = 8
HEADS = 6
MAX_LEN = 4096

VAL_SHARD = "datasets/processed/wikitext_60k_4096/val_shard_0000"
CKPT_DIR = "artifacts/zia_mixed_v1"
OUT_CSV = "artifacts/zia_pretrain_v6/eval_results.csv"


class TinyGPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = torch.nn.Embedding(VOCAB, D_MODEL)
        self.pos = torch.nn.Embedding(MAX_LEN, D_MODEL)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                D_MODEL, HEADS, D_MODEL*4,
                dropout=0.1, activation="gelu",
                batch_first=True, norm_first=True
            )
        for _ in range(LAYERS)])
        self.ln_f = torch.nn.LayerNorm(D_MODEL)
        self.head = torch.nn.Linear(VOCAB, D_MODEL, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device))
        for blk in self.blocks: h = blk(h)
        return self.head(self.ln_f(h))


def load_val():
    x = torch.load(f"{VAL_SHARD}/input_ids.pt", map_location="cpu")[:50]
    y = torch.load(f"{VAL_SHARD}/labels.pt", map_location="cpu")[:50]
    return x.long(), y.long()


@torch.no_grad()
def main():
    x, y = load_val()
    x, y = x.to(DEVICE), y.to(DEVICE)

    ckpts = sorted([p for p in Path(CKPT_DIR).glob("*.pt")])
    results = []

    print(f"[i] Evaluating {len(ckpts)} checkpoints...\n")

    for ckpt in ckpts:
        model = TinyGPT().to(DEVICE)
        try:
            st = torch.load(ckpt, map_location=DEVICE).get("model")
            model.load_state_dict(st, strict=False)
        except:
            continue

        model.eval()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, VOCAB), 
            y.view(-1), 
            ignore_index=-100
        ).item()

        ppl = math.exp(min(loss, 20))
        results.append((ckpt.name, loss, ppl))
        print(f"{ckpt.name}: loss={loss:.4f}, ppl={ppl:.1f}")

    # Save CSV
    with open(OUT_CSV, "w") as f:
        f.write("ckpt,loss,ppl\n")
        for r in results:
            f.write(f"{r[0]},{r[1]},{r[2]}\n")

    print(f"\n[✓] Saved eval report: {OUT_CSV}")


if __name__ == "__main__":
    main()
