# datasets/code/merge_datasets.py
import os, json, hashlib

ORG_DIR = "datasets/organized"
OUT_DIR = "datasets/organized/merged"
CHUNK_SIZE = 10 * 1024**3  # ~10 GB per output file

def file_md5(line: str):
    return hashlib.md5(line.encode("utf-8")).hexdigest()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    datasets = [f for f in os.listdir(ORG_DIR) if f.endswith(".jsonl")]
    print(f"[i] Found {len(datasets)} datasets to merge")

    seen = set()
    chunk_idx, total, chunk_bytes = 0, 0, 0
    out_path = os.path.join(OUT_DIR, f"merged_part_{chunk_idx}.jsonl")
    out = open(out_path, "w", encoding="utf-8")

    for ds in datasets:
        ds_path = os.path.join(ORG_DIR, ds)
        print(f"[→] Processing {ds}")
        with open(ds_path, "r", encoding="utf-8") as f:
            for line in f:
                h = file_md5(line)
                if h in seen:
                    continue
                seen.add(h)

                out.write(line)
                total += 1
                chunk_bytes += len(line.encode("utf-8"))

                if chunk_bytes > CHUNK_SIZE:
                    out.close()
                    print(f"[✓] Saved chunk {chunk_idx} ({chunk_bytes/1024**3:.2f} GB)")
                    chunk_idx += 1
                    out_path = os.path.join(OUT_DIR, f"merged_part_{chunk_idx}.jsonl")
                    out = open(out_path, "w", encoding="utf-8")
                    chunk_bytes = 0

    out.close()
    print(f"[✓] Finalized. Total samples: {total}, chunks: {chunk_idx+1}")

if __name__ == "__main__":
    main()
