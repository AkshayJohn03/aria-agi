# datasets/code/process_mathqa.py
import json
import os

RAW_DIR = "datasets/raw/MathQA"
OUT_FILE = "datasets/organized/mathqa.jsonl"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def convert_split(fname, split):
    path = os.path.join(RAW_DIR, fname)
    data = load_json(path)
    for i, row in enumerate(data):
        yield {
            "id": f"{split}_{i}",
            "problem": row.get("Problem", ""),
            "rationale": row.get("Rationale", ""),
            "options": row.get("options", ""),
            "correct": row.get("correct", ""),
            "annotated_formula": row.get("annotated_formula", ""),
            "linear_formula": row.get("linear_formula", ""),
            "category": row.get("category", ""),
            "split": split,
        }

def main():
    splits = {
        "train": "train.json",
        "validation": "dev.json",
        "test": "test.json",
        "challenge": "challenge_test.json"
    }
    
    total = 0
    with open(OUT_FILE, "w", encoding="utf-8") as out:
        for split, fname in splits.items():
            for row in convert_split(fname, split):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1
    print(f"[✓] Saved {total} samples to {OUT_FILE}")

if __name__ == "__main__":
    main()
