import os
import json

ORG_DIR = "datasets/organized"
OUT_DIR = "datasets/merged"
CHUNK_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB per chunk

os.makedirs(OUT_DIR, exist_ok=True)

def merge_datasets():
    datasets = []
    for root, _, files in os.walk(ORG_DIR):
        for f in files:
            if f.endswith(".jsonl"):
                datasets.append(os.path.join(root, f))

    print(f"[i] Found {len(datasets)} datasets to merge")

    chunk_index = 0
    current_size = 0
    out_file = open(os.path.join(OUT_DIR, f"chunk_{chunk_index}.jsonl"), "w", encoding="utf-8")
    total_samples = 0

    for ds_path in datasets:
        ds_name = os.path.basename(ds_path)
        print(f"[→] Processing {ds_name}")

        with open(ds_path, "r", encoding="utf-8") as f:
            for line in f:
                out_file.write(line)
                total_samples += 1
                current_size += len(line.encode("utf-8"))

                if current_size >= CHUNK_SIZE_BYTES:
                    out_file.close()
                    print(f"[✓] Saved chunk {chunk_index} ({CHUNK_SIZE_BYTES / 1e9:.2f} GB)")
                    chunk_index += 1
                    current_size = 0
                    out_file = open(os.path.join(OUT_DIR, f"chunk_{chunk_index}.jsonl"), "w", encoding="utf-8")

    out_file.close()
    print(f"[✓] Finalized. Total samples: {total_samples}, chunks: {chunk_index+1}")

if __name__ == "__main__":
    merge_datasets()
