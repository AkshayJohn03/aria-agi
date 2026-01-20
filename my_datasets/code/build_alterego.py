import os
import json
from pathlib import Path

# === Config ===
INPUT_CHAT_FILE = r"D:\aria\aria_ai\aria_ai_assistant\datasets\raw\alterego_chats.json"  
OUTPUT_JSONL = r"D:\aria\aria_ai\aria_ai_assistant\datasets\organized\alterego.jsonl"  

# Expect input format:
# [
#   {"role": "user", "content": "I smoke a lot, should I quit?"},
#   {"role": "assistant", "content": "Yes. Smoking reduces endurance..."}
#   ...
# ]

def convert_to_alterego(samples):
    """
    Convert conversation logs into instruction-output pairs.
    """
    dataset = []
    user_msg = None

    for msg in samples:
        if msg["role"] == "user":
            user_msg = msg["content"].strip()
        elif msg["role"] == "assistant" and user_msg:
            dataset.append({
                "instruction": user_msg,
                "input": "",
                "output": msg["content"].strip()
            })
            user_msg = None

    return dataset

def main():
    if not os.path.exists(INPUT_CHAT_FILE):
        print(f"❌ No chat file found at {INPUT_CHAT_FILE}")
        return

    with open(INPUT_CHAT_FILE, "r", encoding="utf-8") as f:
        samples = json.load(f)

    dataset = convert_to_alterego(samples)

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(dataset)} Alter Ego pairs to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
