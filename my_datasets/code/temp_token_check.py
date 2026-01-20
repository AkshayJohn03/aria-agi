from transformers import AutoTokenizer
from datasets import load_from_disk, concatenate_datasets
import os

# Define the paths
dataset_root_path = "my_datasets/processed/arrow_cleaned_v1"
train_path = os.path.join(dataset_root_path, "train")
tokenizer_path = "artifacts/zia_tokenizer_60k"

# Load the tokenizer
tok = AutoTokenizer.from_pretrained(tokenizer_path)

# List all shard directories inside the 'train' folder
shard_dirs = sorted([
    os.path.join(train_path, d) 
    for d in os.listdir(train_path) 
    if os.path.isdir(os.path.join(train_path, d))
])

# Load each shard and store them in a list
datasets_list = []
print(f"Loading {len(shard_dirs)} shards from '{train_path}'...")
for shard_path in shard_dirs:
    try:
        ds = load_from_disk(shard_path)
        datasets_list.append(ds)
        print(f"Successfully loaded {shard_path}")
    except FileNotFoundError as e:
        print(f"Skipping {shard_path}: {e}")

# Concatenate all the loaded datasets into a single Dataset object
if datasets_list:
    ds_full = concatenate_datasets(datasets_list)
    print(f"\nConcatenated all shards. Total samples in dataset: {len(ds_full)}")
else:
    print("\nNo datasets were loaded. Please check the directory structure.")
    exit()

# Define the function to count tokens
def count_tokens(dataset, tokenizer, max_samples=10000):
    total, seen = 0, 0
    # Use enumerate for tracking progress
    for i, ex in enumerate(dataset):
        # Stop after max_samples to save time
        if max_samples and i >= max_samples:
            break
        
        # Safely get the text content
        txt = ex.get("text") or ex.get("content") or str(ex)
        total += len(tokenizer.encode(txt, add_special_tokens=False))
        seen += 1
    
    # Calculate average and estimated total tokens
    avg = total / seen if seen > 0 else 0
    est_total = avg * len(dataset)
    return avg, est_total

# Run the token count function on the full dataset
avg, total = count_tokens(ds_full, tok)
print(f"Avg tokens per sample: {avg:.2f}")
print(f"Estimated total tokens: {total/1e9:.2f}B")