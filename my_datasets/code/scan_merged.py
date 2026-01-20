import json
import glob
from tqdm import tqdm

input_files = glob.glob("datasets/organized/merged/*.jsonl")
output_file = "datasets/organized/unified/unified.jsonl"

with open(output_file, "w", encoding="utf-8") as fout:
    for f in input_files:
        print(f"Processing {f}")
        with open(f, "r", encoding="utf-8") as fin:
            for line in tqdm(fin):
                obj = json.loads(line)

                # Case 1: Already has messages
                if "messages" in obj:
                    norm = {
                        "messages": obj["messages"],
                        "source": obj.get("source", "unknown"),
                        "type": obj.get("type", "generic")
                    }

                # Case 2: Just role/content
                elif "role" in obj and "content" in obj:
                    norm = {
                        "messages": [{"role": obj["role"], "content": obj["content"]}],
                        "source": "unknown",
                        "type": "single"
                    }

                else:
                    continue  # skip malformed entries

                fout.write(json.dumps(norm, ensure_ascii=False) + "\n")

print(f"✅ Unified dataset saved at {output_file}")
