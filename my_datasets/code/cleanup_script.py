import os
import re
import json
from bs4 import BeautifulSoup
from datasets import load_from_disk, concatenate_datasets, Dataset, DatasetDict

# --- Config ---
base_dir = "my_datasets/processed/arrow_normalized_v1"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test", "test")  # nested test/test
out_dir = "my_datasets/processed/arrow_cleaned_v1"
log_file = os.path.join(out_dir, "cleanup_log.json")

# --- Cleaner ---
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1. Remove HTML
    text = BeautifulSoup(text, "lxml").get_text(separator=" ")

    # 2. Remove boilerplate
    boilerplate_patterns = [
        r"(?i)click here.*", r"(?i)subscribe now.*", r"(?i)advertisement.*",
        r"(?i)all rights reserved.*", r"(?i)cookie policy.*",
        r"(?i)terms of service.*", r"(?i)privacy policy.*"
    ]
    for pat in boilerplate_patterns:
        text = re.sub(pat, " ", text)

    # 3. Normalize spaces/unicode
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)

    # 4. Remove weird symbols but keep punctuation
    text = re.sub(r"[^\w\s\.,!?;:'\"()\[\]{}<>\-\/=+*&%$#@^]", " ", text)

    return text.strip()

def process_batch(batch):
    return {"text": [clean_text(x) for x in batch["text"]]}

def save_log(log_data):
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

def main():
    os.makedirs(out_dir, exist_ok=True)
    cleaned_train_dir = os.path.join(out_dir, "train")
    cleaned_test_dir = os.path.join(out_dir, "test")
    os.makedirs(cleaned_train_dir, exist_ok=True)
    os.makedirs(cleaned_test_dir, exist_ok=True)

    log_data = {"train": {}, "test": {}}

    # --- Process train shards ---
    shard_names = sorted(os.listdir(train_dir))
    for shard_name in shard_names:
        shard_path = os.path.join(train_dir, shard_name)
        out_shard_path = os.path.join(cleaned_train_dir, shard_name)

        if not os.path.isdir(shard_path):
            continue

        # Skip already processed
        if os.path.exists(out_shard_path):
            print(f"[✓] Skipping {shard_name} (already cleaned)")
            continue

        print(f"[i] Loading {shard_name}...")
        shard_ds = load_from_disk(shard_path)

        print(f"[i] Cleaning {shard_name}...")
        cleaned = shard_ds.map(process_batch, batched=True, num_proc=os.cpu_count(), desc=f"Cleaning {shard_name}")

        print(f"[i] Saving cleaned {shard_name}...")
        cleaned.save_to_disk(out_shard_path)

        # Log shard stats
        log_data["train"][shard_name] = {
            "rows": len(cleaned),
        }
        save_log(log_data)

    # --- Process test dataset ---
    if not os.path.exists(cleaned_test_dir):
        print("[i] Loading test dataset...")
        test_ds = load_from_disk(test_dir)

        print("[i] Cleaning test dataset...")
        cleaned_test = test_ds.map(process_batch, batched=True, num_proc=os.cpu_count(), desc="Cleaning test")

        print("[i] Saving cleaned test dataset...")
        cleaned_test.save_to_disk(cleaned_test_dir)

        log_data["test"] = {"rows": len(cleaned_test)}
        save_log(log_data)
    else:
        print("[✓] Skipping test dataset (already cleaned)")

    print(f"[✓] Cleanup complete. Cleaned shards are in {out_dir}")

if __name__ == "__main__":
    main()
