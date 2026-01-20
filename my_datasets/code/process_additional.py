import os
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset

def save_jsonl(data, output_path, max_items=None, offset=0):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count, saved = 0, 0
    with open(output_path, "w", encoding="utf-8") as out:
        for item in tqdm(data, desc=f"Processing {os.path.basename(output_path)}"):
            if count >= offset:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                saved += 1
                if max_items and saved >= max_items:
                    break
            count += 1
    print(f"[✓] Saved {saved} samples to {output_path}")


# --- FIXED DATASETS ---

def process_reddit_tifu(output_dir, max_items=None, offset=0):
    ds_long = load_dataset("carleslc/reddit-tifu", "long", split="train")
    data_long = [
        {"prompt": ex["documents"], "completion": ex["tldr"]}
        for ex in ds_long
    ]
    save_jsonl(data_long, f"{output_dir}/reddit_tifu_long.jsonl", max_items, offset)

    ds_short = load_dataset("carleslc/reddit-tifu", "short", split="train")
    data_short = [
        {"prompt": ex["documents"], "completion": ex["tldr"]}
        for ex in ds_short
    ]
    save_jsonl(data_short, f"{output_dir}/reddit_tifu_short.jsonl", max_items, offset)



def process_oasst1(output_dir, max_items=None, offset=0):
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    data = []
    for ex in ds:
        convo = ex.get("conversation", [])
        for i in range(len(convo) - 1):
            if convo[i]["role"] == "user" and convo[i + 1]["role"] == "assistant":
                data.append({
                    "prompt": convo[i]["content"],
                    "completion": convo[i + 1]["content"]
                })
    save_jsonl(data, f"{output_dir}/oasst1.jsonl", max_items, offset)


def process_sharegpt(output_dir, max_items=None, offset=0):
    ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", data_files="ShareGPT_V3_unfiltered_cleaned_split.json", split="train")

    data = []
    for ex in ds:
        conv = ex.get("conversations", [])
        if len(conv) >= 2:
            prompt = conv[0].get("value", "")
            completion = conv[1].get("value", "")
            if prompt and completion:
                data.append({"prompt": prompt, "completion": completion})

    save_jsonl(data, f"{output_dir}/sharegpt.jsonl", max_items, offset)


def process_codealpaca(output_dir, max_items=None, offset=0):
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    data = [{"prompt": ex["instruction"], "completion": ex["output"]} for ex in ds]
    save_jsonl(data, f"{output_dir}/codealpaca.jsonl", max_items, offset)


def process_gsm8k(output_dir, max_items=None, offset=0):
    ds = load_dataset("gsm8k", "main", split="train")
    data = [{"prompt": ex["question"], "completion": ex["answer"]} for ex in ds]
    save_jsonl(data, f"{output_dir}/gsm8k.jsonl", max_items, offset)


def process_mathqa(output_dir, max_items=None, offset=0):
    ds = load_dataset("lucasmccabe-lmi/MathQA", split="train")
    data = [{"prompt": ex["Problem"], "completion": ex["Rationale"]} for ex in ds]
    save_jsonl(data, f"{output_dir}/mathqa.jsonl", max_items, offset)


# --- MAIN ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["reddit_tifu", "oasst1", "sharegpt", "codealpaca", "gsm8k", "mathqa"]
    )
    parser.add_argument("--output-dir", default="datasets/organized")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    if args.dataset == "reddit_tifu":
        process_reddit_tifu(args.output_dir, args.max_items, args.offset)
    elif args.dataset == "oasst1":
        process_oasst1(args.output_dir, args.max_items, args.offset)
    elif args.dataset == "sharegpt":
        process_sharegpt(args.output_dir, args.max_items, args.offset)
    elif args.dataset == "codealpaca":
        process_codealpaca(args.output_dir, args.max_items, args.offset)
    elif args.dataset == "gsm8k":
        process_gsm8k(args.output_dir, args.max_items, args.offset)
    elif args.dataset == "mathqa":
        process_mathqa(args.output_dir, args.max_items, args.offset)
