# inspect_arrow.py
import os
import json
from datasets import load_from_disk
from pathlib import Path
from statistics import mean
from tqdm.auto import tqdm # <- ADD THIS LINE

ROOT = Path(".")
NORMALIZED = ROOT / "my_datasets" / "processed" / "arrow_normalized_v1"
CLEANED = ROOT / "my_datasets" / "processed" / "arrow_cleaned_v1"
OUT = ROOT / "inspect_report.json"

def dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except Exception:
                pass
    return total

def inspect_dataset_root(root_path: Path):
    info = {"root": str(root_path), "exists": root_path.exists()}
    if not root_path.exists():
        return info

    # determine top-level children (train/test)
    children = {}
    for child in tqdm(sorted(p for p in root_path.iterdir() if p.is_dir()), desc="[i] Inspecting splits"): # <- ADD tqdm HERE
        children[child.name] = inspect_split(child)
    info["splits"] = children
    info["disk_bytes_total"] = dir_size(root_path)
    return info

def inspect_split(split_path: Path):
    # If split_path contains shards (shard_000 etc) -> iterate each
    items = sorted(p for p in split_path.iterdir() if p.is_dir())
    # If no subdirs but files exist (single dataset dir), treat as single
    if not items and any(split_path.iterdir()):
        # this is a dataset dir already
        return inspect_shard(split_path)
    shards = {}
    total_rows = 0
    for d in tqdm(items, desc=f"[i] Inspecting shards in {split_path.name}"): # <- ADD tqdm HERE
        try:
            sh = inspect_shard(d)
            shards[d.name] = sh
            total_rows += sh.get("num_rows", 0)
        except Exception as e:
            shards[d.name] = {"error": str(e)}
    return {"shards": shards, "total_rows": total_rows, "disk_bytes": dir_size(split_path)}

def inspect_shard(shard_dir: Path):
    # attempt to load
    info = {"shard_path": str(shard_dir)}
    ds = load_from_disk(str(shard_dir))
    info["num_rows"] = len(ds)
    info["column_names"] = ds.column_names
    info["features"] = str(ds.features) if hasattr(ds, "features") else None

    # get sample(s)
    sample = {}
    if len(ds) > 0:
        example = ds[0]
        # pick a couple of fields to show
        for k in ("text", "messages"):
            if k in example:
                sample[k] = example[k]
        # fallback: include first key
        if not sample:
            keys = list(example.keys())
            sample[keys[0]] = example[keys[0]]
    info["sample"] = sample

    # compute avg token/char length heuristics (chars)
    lengths = []
    # This loop is fast, no need for tqdm
    for i, ex in enumerate(ds.select(range(min(1000, len(ds))))):
        txt = ""
        if "messages" in ex:
            # join messages content
            try:
                txt = "\n".join(m.get("content", "") if isinstance(m, dict) else str(m) for m in ex["messages"])
            except Exception:
                txt = str(ex["messages"])
        elif "text" in ex:
            txt = ex["text"]
        else:
            # join all fields
            txt = " ".join(str(v) for v in ex.values())
        lengths.append(len(txt))
    info["sample_count_for_stats"] = len(lengths)
    info["avg_chars_sample"] = mean(lengths) if lengths else 0
    info["disk_bytes"] = dir_size(shard_dir)
    # do not keep heavy metadata
    return info

def main():
    report = {
        "normalized": inspect_dataset_root(NORMALIZED),
        "cleaned": inspect_dataset_root(CLEANED),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[✓] Saved inspection report to {OUT}")

if __name__ == "__main__":
    main()