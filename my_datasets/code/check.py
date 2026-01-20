from datasets import load_from_disk
import hashlib

ds = load_from_disk("artifacts/tokenized_datasets/zia_ift_v3")
train, val = ds["train"], ds["validation"]

def hash_example(example):
    return hashlib.md5(str(example["input_ids"]).encode()).hexdigest()

train_hashes = set(map(hash_example, train))
val_hashes = set(map(hash_example, val))
overlap = len(train_hashes.intersection(val_hashes))
print(f"Overlap between train and val: {overlap} / {len(val)}")