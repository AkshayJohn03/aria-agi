import json
import os

# Point this to the ORIGINAL 60k tokenizer file you used BEFORE cleaning
# If you deleted it, point to "artifacts/zia_tokenizer_60k/tokenizer.json"
ORIGINAL_PATH = "artifacts/zia_tokenizer_60k/tokenizer.json"

def inspect():
    if not os.path.exists(ORIGINAL_PATH):
        print(f"❌ Error: Original tokenizer not found at {ORIGINAL_PATH}")
        return

    print(f"Reading {ORIGINAL_PATH}...")
    with open(ORIGINAL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print("\n--- TOKENIZER METADATA ---")
    if "model" in data:
        print(f"Model Type: {data['model']['type']}")
        print(f"Vocab Size: {len(data['model']['vocab'])}")
        
        # Print first 10 tokens
        print("\n--- FIRST 10 TOKENS ---")
        for i, (token, score) in enumerate(list(data['model']['vocab'].items())[:10]):
            print(f"ID {i}: {repr(token)}")
            
        # Print last 10 tokens
        print("\n--- LAST 10 TOKENS ---")
        vocab_list = list(data['model']['vocab'].items())
        for i, (token, score) in enumerate(vocab_list[-10:], start=len(vocab_list)-10):
            print(f"ID {i}: {repr(token)}")
            
    if "added_tokens" in data:
        print("\n--- ADDED TOKENS ---")
        for t in data["added_tokens"]:
            print(f"ID {t['id']}: {t['content']}")

if __name__ == "__main__":
    inspect()