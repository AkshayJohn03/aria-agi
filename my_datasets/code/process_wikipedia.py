import os
import bz2
import argparse
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

def stream_wikipedia(input_path, output_path, max_items=4_000_000, offset=0):
    """
    Stream-parse Wikipedia XML dump (.xml or .bz2) and save chunks as JSONL.
    Each run extracts up to max_items (≈8GB safe per run).
    """

    # Handle compressed/uncompressed
    if input_path.endswith(".bz2"):
        f = bz2.open(input_path, "rb")
    else:
        f = open(input_path, "rb")

    count = 0
    saved = 0
    context = ET.iterparse(f, events=("end",))

    with open(output_path, "w", encoding="utf-8") as out, tqdm(unit="pages") as pbar:
        for event, elem in context:
            if elem.tag.endswith("page"):
                title_elem = elem.find("./{*}title")
                text_elem = elem.find("./{*}revision/{*}text")

                title = title_elem.text if title_elem is not None else None
                text = text_elem.text if text_elem is not None else None

                if title and text:
                    if count >= offset:
                        sample = {
                            "prompt": f"Write an encyclopedic article about {title}:",
                            "completion": text.strip()
                        }
                        out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        saved += 1
                        if saved >= max_items:
                            break
                    count += 1
                    pbar.update(1)

                elem.clear()

    print(f"[✓] Saved {saved} articles to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="my_datasets/raw/enwiki-latest-pages-articles.xml/enwiki-latest-pages-articles.xml")
    parser.add_argument("--output", default="my_datasets/organized/wikipedia_chunk.jsonl")
    parser.add_argument("--max-items", type=int, default=4_000_000, help="Max articles per run (≈8GB safe)")
    parser.add_argument("--offset", type=int, default=0, help="Start index (for chunking)")
    args = parser.parse_args()

    base_out = os.path.splitext(args.output)[0]
    output_path = f"{base_out}_{args.offset}.jsonl"

    stream_wikipedia(args.input, output_path, args.max_items, args.offset)
