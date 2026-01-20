import os
import argparse
import json
import pyarrow.parquet as pq
from tqdm import tqdm

def process_parquet(parquet_path, output_path, max_items=2_000_000, offset=0):
    """
    Convert Parquet dataset (Ultrachat) into JSONL with chunking.
    Each run extracts up to max_items examples starting from offset.
    """

    table = pq.read_table(parquet_path)
    df = table.to_pandas()  # convert Arrow -> Pandas for row iteration

    count, saved = 0, 0
    with open(output_path, "w", encoding="utf-8") as out:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(parquet_path)}"):
            if count >= offset:
                sample = {
                    "prompt": row.get("prompt") or row.get("instruction") or "",
                    "completion": row.get("response") or row.get("output") or "",
                }
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                saved += 1
                if saved >= max_items:
                    break
            count += 1

    print(f"[✓] Saved {saved} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to Ultrachat .parquet file")
    parser.add_argument("--output", default="datasets/organized/ultrachat_chunk.jsonl", help="Output JSONL file")
    parser.add_argument("--max-items", type=int, default=2_000_000, help="Max rows per run (≈8GB safe)")
    parser.add_argument("--offset", type=int, default=0, help="Row offset for chunking")
    args = parser.parse_args()

    base_out = os.path.splitext(args.output)[0]
    output_path = f"{base_out}_{args.offset}.jsonl"

    process_parquet(args.input, output_path, args.max_items, args.offset)
