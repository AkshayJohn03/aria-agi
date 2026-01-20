import torch
from datasets import load_from_disk

ds = load_from_disk("artifacts/tokenized_datasets/zia_ift_v3")["train"]
max_id = 0
min_id = 999999
for i in range(10):
    sample = ds[i]
    ids = torch.tensor(sample["labels"])
    valid = ids[ids != -100]
    max_id = max(max_id, valid.max().item())
    min_id = min(min_id, valid.min().item())

print(f"Label min: {min_id}, max: {max_id}")