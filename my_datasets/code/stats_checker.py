import os
import json

def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def get_stats(base_dir="datasets/organized"):
    stats = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith(".jsonl"):
                path = os.path.join(root, fname)
                size_gb = os.path.getsize(path) / (1024**3)
                try:
                    n_lines = count_lines(path)
                except Exception as e:
                    n_lines = f"error: {e}"
                stats.append({
                    "file": fname,
                    "lines": n_lines,
                    "size_gb": round(size_gb, 2)
                })
    return stats

if __name__ == "__main__":
    stats = get_stats()
    total_lines = 0
    total_size = 0
    for s in stats:
        print(f"{s['file']:40} {s['lines']:>12} lines  {s['size_gb']:>6} GB")
        if isinstance(s["lines"], int):
            total_lines += s["lines"]
        total_size += s["size_gb"]
    print("\nTOTAL:", total_lines, "lines,", round(total_size, 2), "GB")
