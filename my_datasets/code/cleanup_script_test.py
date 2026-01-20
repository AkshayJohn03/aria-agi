# clean_test_and_save.py
import os, re, warnings
from pathlib import Path
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from datasets import load_from_disk

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

ROOT = Path(".")
SRC_TEST = ROOT / "my_datasets" / "processed" / "arrow_normalized_v1" / "test" / "test"
OUT_TEST = ROOT / "my_datasets" / "processed" / "arrow_cleaned_v1" / "test"

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # remove HTML safely
    text = BeautifulSoup(text, "lxml").get_text(separator=" ")
    # basic boilerplate removal
    patterns = [r"(?i)click here", r"(?i)subscribe now", r"(?i)advertisement", r"(?i)cookie policy", r"(?i)privacy policy"]
    for p in patterns:
        text = re.sub(p, " ", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # keep punctuation, code symbols & emojis generally intact
    return text

def detect_and_clean(ds):
    # returns cleaned ds (not saved)
    if "messages" in ds.column_names:
        def map_func(batch):
            cleaned = []
            for msgs in batch["messages"]:
                # msgs is expected list of dicts
                new_msgs = []
                if isinstance(msgs, list):
                    for m in msgs:
                        if isinstance(m, dict):
                            c = m.get("content", "")
                            m["content"] = clean_text(c)
                            new_msgs.append(m)
                        else:
                            new_msgs.append({"role":"", "content": clean_text(str(m))})
                    cleaned.append(new_msgs)
                else:
                    cleaned.append([])
            return {"messages": cleaned}
        return ds.map(map_func, batched=True, num_proc=os.cpu_count() or 1, desc="Cleaning test messages")
    elif "text" in ds.column_names:
        def map_text(batch):
            return {"text": [clean_text(t) for t in batch["text"]]}
        return ds.map(map_text, batched=True, num_proc=os.cpu_count() or 1, desc="Cleaning test text")
    else:
        # fallback: convert all fields to a text field, clean it, and save as text
        keys = ds.column_names
        def map_any(batch):
            out = []
            for i in range(len(batch[keys[0]])):
                pieces = []
                for k in keys:
                    pieces.append(str(batch[k][i]))
                txt = " ".join(pieces)
                out.append(clean_text(txt))
            return {"text": out}
        ds = ds.map(map_any, batched=True, num_proc=os.cpu_count() or 1, desc="Cleaning test (fallback)")
        return ds

def main():
    if not SRC_TEST.exists():
        print(f"[!] Source test dataset not found at {SRC_TEST}")
        return

    OUT_TEST.mkdir(parents=True, exist_ok=True)
    # if already saved, skip
    if any(OUT_TEST.iterdir()):
        print(f"[✓] Cleaned test directory already contains files — skipping: {OUT_TEST}")
        return

    print("[i] Loading source test dataset...")
    ds = load_from_disk(str(SRC_TEST))
    print(f"[i] test rows: {len(ds)} | columns: {ds.column_names}")

    print("[i] Cleaning test dataset in parallel...")
    cleaned = detect_and_clean(ds)

    print(f"[i] Saving cleaned test dataset to {OUT_TEST} ...")
    cleaned.save_to_disk(str(OUT_TEST))
    print(f"[✓] Done. Cleaned test saved at {OUT_TEST}")

if __name__ == "__main__":
    main()
