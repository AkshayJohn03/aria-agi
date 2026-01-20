import json, os
from tqdm import tqdm

def process_mathqa(output_dir, max_items=None, offset=0):
    base = "datasets/raw/MathQA"
    splits = {
        "train": "train.json",
        "valid": "dev.json",
        "test": "test.json",
    }

    data = []
    for split, fname in splits.items():
        fpath = os.path.join(base, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            rows = json.load(f)
            for r in tqdm(rows, desc=f"Processing {split}"):
                prob = r.get("Problem", "").strip()
                rationale = r.get("Rationale", "").strip()
                if not prob or not rationale:
                    continue
                data.append({
                    "prompt": prob,
                    "completion": rationale,
                    "meta": {"source": "MathQA", "split": split}
                })

    save_jsonl(data, f"{output_dir}/mathqa.jsonl", max_items, offset)
