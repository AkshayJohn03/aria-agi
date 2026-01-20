import torch
import torch.nn as nn
import os

# --- CONFIG ---
OLD_CKPT = "artifacts/zia_ift_fixed4k/checkpoint_step240.pt"
NEW_PATH = "artifacts/zia_ift_v5/zia_v5_ready.pt"

# Verified Tokenizer Truth
TARGET_VOCAB = 60011 
D_MODEL = 384
# --------------

def expand():
    print(f"Loading checkpoint: {OLD_CKPT}")
    if not os.path.exists(OLD_CKPT):
        print("❌ Error: Checkpoint not found.")
        return

    ck = torch.load(OLD_CKPT, map_location="cpu")

    # Handle structure
    state_dict = ck["model"] if "model" in ck else ck

    print(f"Expanding model to Vocab Size: {TARGET_VOCAB}")
    new_state_dict = {}

    for key, val in state_dict.items():
        if "tok.weight" in key or "head.weight" in key:
            old_rows = val.shape[0]
            print(f"  -> Resizing {key}: {val.shape} -> [{TARGET_VOCAB}, {D_MODEL}]")

            # 1. Create new larger matrix
            new_param = torch.zeros((TARGET_VOCAB, D_MODEL), dtype=val.dtype)

            # 2. Copy old trained weights (The Knowledge)
            new_param[:old_rows] = val

            # 3. Initialize new rows safely (The New Vocabulary)
            # Small random noise ensures they can learn
            nn.init.normal_(new_param[old_rows:], mean=0.0, std=0.02)

            new_state_dict[key] = new_param
        else:
            # Keep all other layers (The Brain) untouched
            new_state_dict[key] = val

    # Prepare final checkpoint package
    new_ckpt = {
        "model": new_state_dict,
        "step": 0,             # Reset steps
        "best_val": 999.0,     # Reset validation record
        "config": {
            "d_model": D_MODEL,
            "vocab_size": TARGET_VOCAB,
            "n_layers": 8,
            "n_heads": 6,
            "context_len": 4096
        }
    }

    os.makedirs(os.path.dirname(NEW_PATH), exist_ok=True)
    torch.save(new_ckpt, NEW_PATH)

    print("\n✅ SUCCESS: Model Expanded.")
    print(f"Saved to: {NEW_PATH}")
    print("You are ready to train.")

if __name__ == "__main__":
    expand()