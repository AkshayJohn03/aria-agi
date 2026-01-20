import os
import json
from pathlib import Path
import hashlib

# === Paths ===
EXPORT_FILE = r"D:\aria\aria_ai\aria_ai_assistant\datasets\raw\chatgpt_export\conversations.json"
OUTPUT_DIR = r"D:\aria\aria_ai\aria_ai_assistant\datasets\raw\alterego"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Helpers ===
def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def clean_message(msg):
    if not msg or not msg.strip():
        return None
    return msg.strip()

def flatten_parts(parts):
    """Convert parts into a flat string (handles dicts, strings, lists)."""
    texts = []
    for p in parts:
        if isinstance(p, str):
            texts.append(p)
        elif isinstance(p, dict):
            # Sometimes export has dicts like {"text": "..."} or {"content": "..."}
            if "text" in p:
                texts.append(p["text"])
            elif "content" in p:
                texts.append(str(p["content"]))
            else:
                texts.append(json.dumps(p, ensure_ascii=False))  # fallback
        else:
            texts.append(str(p))
    return " ".join(texts)

# === Parse Conversations ===
def parse_conversations():
    with open(EXPORT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = []
    seen_hashes = set()

    for conv in data:
        if "mapping" not in conv:
            continue

        for node in conv["mapping"].values():
            msg = node.get("message")
            if not msg:
                continue

            author = msg.get("author", {}).get("role")
            content = msg.get("content", {}).get("parts", [])
            if not content or author not in ["user", "assistant"]:
                continue

            text = clean_message(flatten_parts(content))
            if not text:
                continue

            h = hash_text(author + text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            conversations.append({
                "role": author,
                "content": text
            })

    return conversations

# === Save Outputs ===
def save_datasets(conversations):
    jsonl_path = os.path.join(OUTPUT_DIR, "alterego.jsonl")
    json_path = os.path.join(OUTPUT_DIR, "alterego.json")
    txt_path = os.path.join(OUTPUT_DIR, "alterego.txt")

    # JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in conversations:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    # TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        for c in conversations:
            role = "User:" if c["role"] == "user" else "Zia:"
            f.write(f"{role} {c['content']}\n")

    print(f"✅ Saved {len(conversations)} messages:")
    print(f"   JSONL: {jsonl_path}")
    print(f"   JSON: {json_path}")
    print(f"   TXT: {txt_path}")

# === Main ===
if __name__ == "__main__":
    convos = parse_conversations()
    save_datasets(convos)
