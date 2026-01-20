import json

path = "datasets/organized/unified/unified.jsonl"

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 5:  # print only first 5
            obj = json.loads(line)
            print(json.dumps(obj, indent=2, ensure_ascii=False))
