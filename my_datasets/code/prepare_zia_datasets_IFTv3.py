import os, json, hashlib, math
from datasets import load_dataset, Dataset
from tqdm import tqdm

# ---------- CONFIG ----------
SAVE_PATH = "datasets/processed/zia_ift_v3_clean_v2"  # ✅ renamed folder
os.makedirs(SAVE_PATH, exist_ok=True)

# ---------- UTILITY ----------
def dedupe_and_filter(data_list):
    """
    Removes duplicates and overly short examples.
    Expects a standard Python list of dictionaries.
    """
    filtered = [
        ex for ex in data_list
        if len(ex["input"].strip()) > 3 and len(ex["output"].strip()) > 3
    ]
    seen = set()
    unique = []
    for ex in filtered:
        h = hashlib.md5((ex["input"] + ex["output"]).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    return unique

def clean_record(record, src):
    text_in = record.get("instruction") or record.get("input") or record.get("question") or ""
    text_out = record.get("output") or record.get("response") or record.get("answer") or ""
    return {"input": f"[{src}] " + text_in.strip(), "output": text_out.strip()}

# ---------- LOAD ----------
datasets_to_load = [
    ("yahma/alpaca-cleaned", None, 150000, "alpaca"), 
    ("OpenAssistant/oasst1", None, 150000, "oasst"),
    ("databricks/databricks-dolly-15k", None, 5000, "dolly"),
    ("Open-Orca/OpenOrca", None, 200000, "openorca"),
    ("ai4bharat/indic-align", 'Indic_ShareLlama', 150000, "indic_instruct"), 
    ("ai4bharat/samanantar", 'ta', 150000, "indic_parallel_ta"),
]

merged = []
for name, config_name, subset, tag in datasets_to_load:
    print(f"[+] Loading {name} (Config: {config_name or 'default'}, Tag: {tag}) ...")
    try:
        if config_name:
            ds = load_dataset(name, config_name, split=f"train[:{subset}]")
        else:
            ds = load_dataset(name, split=f"train[:{subset}]")

        cleaned = [clean_record(x, tag) for x in ds]
        merged.extend(cleaned)
    except Exception as e:
        print(f"[!] Failed to load {name} (Config: {config_name}): {e}")

# ---------- CLEAN ----------
merged = dedupe_and_filter(merged)
split_idx = int(0.90 * len(merged))
train_list, val_list = merged[:split_idx], merged[split_idx:]

train_ds = Dataset.from_list(train_list)
val_ds = Dataset.from_list(val_list)

# ---------- SAVE (multi-shard safe) ----------
def save_dataset_in_shards(ds, save_dir, num_shards=10):
    """
    Save HuggingFace dataset safely on Windows in smaller Arrow shards.
    """
    os.makedirs(save_dir, exist_ok=True)
    shard_size = math.ceil(len(ds) / num_shards)
    for i in tqdm(range(num_shards), desc=f"Saving → {save_dir}"):
        start, end = i * shard_size, min(len(ds), (i + 1) * shard_size)
        shard = ds.select(range(start, end))
        shard.save_to_disk(os.path.join(save_dir, f"shard_{i:02d}"))

print("[💾] Saving train + val datasets in multi-shard mode...")
save_dataset_in_shards(train_ds, os.path.join(SAVE_PATH, "train"), num_shards=20)
save_dataset_in_shards(val_ds, os.path.join(SAVE_PATH, "val"), num_shards=5)

# ---------- INFO ----------
info = {
    "total": len(merged),
    "train": len(train_list),
    "val": len(val_list),
    "format": "Arrow shards (safe multi-shard saving)"
}
with open(os.path.join(SAVE_PATH, "info.json"), "w") as f:
    json.dump(info, f, indent=2)

print(f"[✅] Saved cleaned dataset in Arrow shard format → {SAVE_PATH}")
print(f"[ℹ️] Stats: {info}")
