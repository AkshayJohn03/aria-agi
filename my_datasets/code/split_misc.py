# datasets/code/split_misc.py
import os, json, argparse
from tqdm import tqdm

def split_misc(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    writers = {}

    counts = {}

    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing misc dataset"):
            try:
                sample = json.loads(line)
            except:
                continue

            # Decide category
            if "source" in sample:
                source = sample["source"]
            elif "instruction" in sample:
                source = "alpaca_like"
            elif "messages" in sample:
                source = "chat_style"
            elif "question" in sample and "answer" in sample:
                source = "qa_style"
            else:
                source = "misc_other"

            # Lazy-open writers
            if source not in writers:
                writers[source] = open(os.path.join(output_dir, f"{source}.jsonl"), "w", encoding="utf-8")
                counts[source] = 0

            writers[source].write(json.dumps(sample, ensure_ascii=False) + "\n")
            counts[source] += 1

    # Close all files
    for w in writers.values():
        w.close()

    print("\n[✓] Split complete. Dataset counts:")
    for k, v in counts.items():
        print(f" - {k}: {v} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/organized/json_misc_datasets.jsonl")
    parser.add_argument("--output-dir", default="datasets/organized/split_misc")
    args = parser.parse_args()

    split_misc(args.input, args.output_dir)
