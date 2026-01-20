import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm

def split_datasets(input_file, output_dir, key_candidates=None):
    """
    Split a merged JSONL into per-dataset files based on a source key.
    """
    if key_candidates is None:
        key_candidates = ["source", "dataset", "origin", "from"]

    os.makedirs(output_dir, exist_ok=True)

    # Prepare file handles for writing
    writers = {}
    counts = defaultdict(int)

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Processing {input_file}"):
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Detect dataset key
            dataset_name = None
            for k in key_candidates:
                if k in sample:
                    dataset_name = str(sample[k]).replace("/", "_")
                    break

            if not dataset_name:
                dataset_name = "unknown"

            # Open file handle lazily
            if dataset_name not in writers:
                out_path = os.path.join(output_dir, f"{dataset_name}.jsonl")
                writers[dataset_name] = open(out_path, "w", encoding="utf-8")

            # Write line
            writers[dataset_name].write(json.dumps(sample, ensure_ascii=False) + "\n")
            counts[dataset_name] += 1

    # Close writers
    for w in writers.values():
        w.close()

    print("\n[✓] Split complete. Dataset counts:")
    for ds, c in counts.items():
        print(f" - {ds}: {c} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to merged JSONL (e.g., complete_chat_datasets.jsonl)")
    parser.add_argument("--output-dir", default="datasets/organized/split", help="Where to save split datasets")
    parser.add_argument("--keys", nargs="+", default=None, help="Candidate keys for dataset source (default: source, dataset, origin, from)")
    args = parser.parse_args()

    split_datasets(args.input, args.output_dir, args.keys)
