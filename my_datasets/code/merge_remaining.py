# datasets/code/merge_remaining.py
import os
import json
from glob import glob

INPUT_DIR = "datasets/organized"
OUTPUT_DIR = "datasets/organized"
CHUNK_SIZE = 10 * 1024 * 1024 * 1024  # 10GB like previous merge
EXISTING_CHUNKS = 5  # You already have chunk_0..chunk_4

def merge_remaining():
    # All jsonl files
    all_files = glob(os.path.join(INPUT_DIR, "*.jsonl")) + glob(os.path.join(INPUT_DIR, "split_chat", "*.jsonl"))

    # Already included in earlier merge
    already_merged = {
        "alterego.jsonl",
        "alterego_uncensored_personal.jsonl",
        "codealpaca.jsonl",
        "complete_document_datasets.jsonl",
        "complete_val.jsonl",
        "financial_news_articles.jsonl",
        "fineweb_chunk_0.jsonl",
        "fin_news.jsonl",
        "gsm8k.jsonl",
        "json_misc_datasets.jsonl",
        "mathqa.jsonl",
        "oasst1.jsonl",
        "openhermes_uncensored_general.jsonl",
        "reddit_tifu.jsonl",
        "sharegpt.jsonl",
        "ultrachat_chunk_0.jsonl",
        "wikipedia_chunk_0.jsonl",
        "zeroshot_financial_filtered_financial.jsonl",
    }

    # Filter remaining datasets
    remaining = [f for f in all_files if os.path.basename(f) not in already_merged]

    if not remaining:
        print("[i] No remaining datasets found.")
        return

    print(f"[i] Found {len(remaining)} remaining datasets to merge")
    out_index = EXISTING_CHUNKS
    out_path = os.path.join(OUTPUT_DIR, f"merged_chunk_{out_index}.jsonl")
    out_file = open(out_path, "w", encoding="utf-8")
    bytes_written = 0
    total_samples = 0

    for file in remaining:
        print(f"[→] Processing {os.path.basename(file)}")
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                out_file.write(line)
                bytes_written += len(line.encode("utf-8"))
                total_samples += 1

                if bytes_written >= CHUNK_SIZE:
                    out_file.close()
                    print(f"[✓] Saved chunk {out_index} ({bytes_written / (1024**3):.2f} GB, {total_samples} samples)")
                    out_index += 1
                    out_path = os.path.join(OUTPUT_DIR, f"merged_chunk_{out_index}.jsonl")
                    out_file = open(out_path, "w", encoding="utf-8")
                    bytes_written = 0
                    total_samples = 0

    out_file.close()
    print(f"[✓] Finalized. Remaining datasets merged into chunks starting from {EXISTING_CHUNKS}")

if __name__ == "__main__":
    merge_remaining()
