import json
import os
from transformers import PreTrainedTokenizerFast

# Config
SRC_JSON = "artifacts/zia_tokenizer_60k/tokenizer.json"
OUT_DIR = "artifacts/zia_tokenizer_legacy"

def restore():
    print(f"[i] Loading original map from {SRC_JSON}...")
    
    # 1. Load the raw tokenizer
    tok = PreTrainedTokenizerFast(tokenizer_file=SRC_JSON)
    
    # 2. Force the Original Special Token Map (Based on your dump)
    # ID 0: [PAD], ID 1: [UNK], ID 2: [BOS], ID 3: [EOS]
    # ID 4: <user>, ID 5: <assistant>, ID 6: <system>
    
    tok.add_special_tokens({
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "additional_special_tokens": ["<user>", "<assistant>", "<system>"]
    })
    
    # 3. Verify IDs match the original dump
    print("\n--- Verifying ID Alignment ---")
    check_map = {
        "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3,
        "<user>": 4, "<assistant>": 5
    }
    
    all_good = True
    for token, expected_id in check_map.items():
        real_id = tok.convert_tokens_to_ids(token)
        print(f"Token '{token}' -> ID {real_id} (Expected: {expected_id})")
        if real_id != expected_id:
            all_good = False
            
    if all_good:
        print("\n✅ SUCCESS: Tokenizer aligned with Checkpoint step240!")
        os.makedirs(OUT_DIR, exist_ok=True)
        tok.save_pretrained(OUT_DIR)
        print(f"Saved to: {OUT_DIR}")
    else:
        print("\n❌ FAILURE: ID Mismatch. Do not train.")

if __name__ == "__main__":
    restore()