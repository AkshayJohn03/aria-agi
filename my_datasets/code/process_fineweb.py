import os
import argparse
import json
from datasets import load_dataset
from tqdm import tqdm

def process_fineweb(output_path, max_items=4_000_000, offset=0):
    """
    Stream the full FineWeb dataset and save in JSONL chunks.
    Each run extracts up to max_items (≈8GB safe).
    """

    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

    count, saved = 0, 0
    with open(output_path, "w", encoding="utf-8") as out:
        for sample in tqdm(ds, desc="Streaming FineWeb", unit="docs"):
            if "text" not in sample:
                continue
            text = sample["text"].strip()
            if not text:
                continue

            if count >= offset:
                record = {
                    "prompt": "Summarize the following text:",
                    "completion": text
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved += 1
                if saved >= max_items:
                    break

            count += 1

    print(f"[✓] Saved {saved} docs to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="datasets/organized/fineweb_chunk.jsonl")
    parser.add_argument("--max-items", type=int, default=4_000_000,
                        help="Max docs per run (≈8GB safe)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Start index (for chunking)")
    args = parser.parse_args()

    base_out = os.path.splitext(args.output)[0]
    output_path = f"{base_out}_{args.offset}.jsonl"

    process_fineweb(output_path, args.max_items, args.offset)
