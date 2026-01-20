#!/usr/bin/env python3
# Deep ZIA checkpoint diagnostic – v3 (patched)
import os, glob, datetime, torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from save_model_for_inference import load_student, sample_generate

# === CONFIG ===
TOKENIZER_PATH = "artifacts/zia_tokenizer_60k"
DATASET_PATH = "artifacts/tokenized_dataset/zia_ift_v3_verified"
CHECKPOINT_DIR = "artifacts/zia_ift_v4_cursor"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_BATCHES = 100
PROMPTS = [
    "Explain the importance of democracy.",
    "Why do humans need medicines?",
    "Describe artificial intelligence in a sentence."
]

# === UTILITIES ===
def collate_fn(batch):
    return (
        torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
        torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
        torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    )

@torch.no_grad()
def compute_loss(model, dl, device, max_batches=100):
    model.eval()
    tot, n = 0.0, 0
    for i, (ids, mask, labels) in enumerate(dl):
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        _, loss = model(ids, attention_mask=mask, labels=labels)
        tot += loss.item()
        n += 1
        if n >= max_batches:
            break
    return tot / max(1, n)

def list_checkpoints():
    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "**", "*.pt"), recursive=True))
    for f in ckpts:
        st = os.stat(f)
        print(f"📦 {f} | {st.st_size/1e6:.1f} MB | {datetime.datetime.fromtimestamp(st.st_mtime)}")
    return ckpts

def run_diagnostics():
    print("\n=== 🔍 ZIA Deep Diagnostics ===\n")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    ds = load_from_disk(DATASET_PATH)

    val_ds = ds["validation"]
    test_ds = ds.get("test", val_ds)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=4, collate_fn=collate_fn)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=4, collate_fn=collate_fn)

    results = []
    for ckpt in list_checkpoints():
        print(f"\n--- Checking {ckpt} ---")
        try:
            model, tokenizer = load_student(ckpt, TOKENIZER_PATH, DEVICE)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[✓] Model loaded | params={n_params:,}")

            val_loss = compute_loss(model, val_dl, DEVICE, VAL_BATCHES)
            test_loss = compute_loss(model, test_dl, DEVICE, VAL_BATCHES // 2)
            print(f"[VAL] {val_loss:.4f} | [TEST] {test_loss:.4f}")

            # Generation using sample_generate()
            print("[GEN] Sample generations:")
            for p in PROMPTS:
                enc = tokenizer(f"Instruction: {p}\nResponse:", return_tensors="pt").to(DEVICE)
                out = sample_generate(model, tokenizer, enc["input_ids"], DEVICE,
                                      max_length=80, temperature=0.7, top_k=40, top_p=0.85)
                text = tokenizer.decode(out[0], skip_special_tokens=True)
                text = text.replace("Ġ", " ").replace("  ", " ")
                if "Response:" in text:
                    text = text.split("Response:")[-1].strip()
                print(f"Q: {p}\nA: {text.strip()}\n")

            results.append((ckpt, val_loss, test_loss))
        except Exception as e:
            print(f"[!] Error: {e}")
            continue

    if results:
        best_ckpt = min(results, key=lambda x: x[1])
        print("\n=== ✅ Summary ===")
        for ck, v, t in results:
            print(f"{ck} -> val={v:.4f} | test={t:.4f}")
        print(f"\n🏆 Best Checkpoint: {best_ckpt[0]} (val={best_ckpt[1]:.4f})")

if __name__ == "__main__":
    run_diagnostics()
