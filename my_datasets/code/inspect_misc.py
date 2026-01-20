# datasets/code/inspect_misc.py
import json

def inspect_file(path, n=20):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n: break
            try:
                sample = json.loads(line)
            except:
                continue
            print(f"\n--- Sample {i+1} ---")
            print(sample.keys())
            if "source" in sample:
                print("Source:", sample["source"])
            if "instruction" in sample:
                print("Instruction-style")
            elif "messages" in sample:
                print("Chat-style")
            elif "question" in sample and "answer" in sample:
                print("QA-style")

if __name__ == "__main__":
    inspect_file("datasets/organized/json_misc_datasets.jsonl")
