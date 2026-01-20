import torch
from transformers import PreTrainedTokenizerFast
from pathlib import Path

# ========= CONFIG =========
VAL_SHARD = "datasets/processed/wikitext_60k_4096/val_shard_0000"
TOKENIZER_FILE = "artifacts/zia_tokenizer_60k_clean/tokenizer.json"
SAMPLE_IDX = 0
# ==========================

def inspect():
    print(f"[i] Loading tokenizer: {TOKENIZER_FILE}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)

    inp_path = Path(VAL_SHARD) / "input_ids.pt"
    lbl_path = Path(VAL_SHARD) / "labels.pt"

    print(f"[i] Loading val shard from: {VAL_SHARD}")
    x = torch.load(inp_path, map_location="cpu")
    y = torch.load(lbl_path, map_location="cpu")

    x_seq = x[SAMPLE_IDX].tolist()
    y_seq = y[SAMPLE_IDX].tolist()

    # Remove ignored indices for decoding
    y_clean = [t for t in y_seq if t != -100]

    print("\n==================== INPUT TEXT ====================")
    print(tokenizer.decode(x_seq[:200]))
    print("...(truncated)")

    print("\n==================== LABEL TEXT ====================")
    print(tokenizer.decode(y_clean[:200]))
    print("...(truncated)")

    print("\n==================== RAW TENSORS ====================")
    print("Input (first 20):", x_seq[:20])
    print("Label (first 20):", y_seq[:20])

    print("\n==================== SHIFT CHECK ====================")

    # shift should match: label[t] == input[t+1]  (except for -100 masking)
    # Find first valid label token
    first_valid = None
    for i, tok in enumerate(y_seq):
        if tok != -100:
            first_valid = i
            break

    if first_valid is None:
        print("[!] ERROR: Label sequence is entirely -100 masked! Model cannot learn.")
        return

    if first_valid + 1 < len(x_seq):
        if y_seq[first_valid] == x_seq[first_valid + 1]:
            print("[✓] Shifted correctly (label[t] == input[t+1])")
        elif y_seq[first_valid] == x_seq[first_valid]:
            print("[!] Incorrect: (label[t] == input[t]) → identity mapping, model learns nothing.")
        else:
            print("[?] Non-standard alignment — investigate tensor printouts.")
    else:
        print("[?] Could not compare shift — sequence too short or masking irregular.")

    print("\n==================== END ====================")

if __name__ == "__main__":
    inspect()
