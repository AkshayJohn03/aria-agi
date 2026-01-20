import os
import torch
from transformers import PreTrainedTokenizerFast
from datasets import Dataset 
from tqdm import tqdm

# CONFIG
# Your existing Arrow files path
ARROW_DIR = r"D:\aria\aria_ai\aria_ai_assistant\datasets\raw\hf_cache\datasets\Open-Orca___open_orca\default\0.0.0\e9c87b4abb2609913751f9b26553fdb9c061796c"
TOKENIZER_DIR = "artifacts/hf_tokenizer_mistral"
OUTPUT_DIR = "datasets/processed/hf_chatml_4096_v2"
SEQ_LEN = 4096
SHARD_SIZE = 1000  # Sequences per shard

def prepare():
    # 1. Load Tokenizer
    print(f"[i] Loading Tokenizer from {TOKENIZER_DIR}...")
    tok = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    
    # Ensure IDs are correct
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else 1
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 2
    
    print(f"[i] Config: PAD={pad_id}, BOS={bos_id}, EOS={eos_id}")

    # 2. Find Arrow Files
    arrow_files = [
        os.path.join(ARROW_DIR, f) for f in os.listdir(ARROW_DIR) 
        if f.endswith(".arrow") and "dataset_info" not in f
    ]
    print(f"[i] Found {len(arrow_files)} Arrow files.")
    
    if not arrow_files:
        print("❌ Error: No .arrow files found. Check ARROW_DIR path.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Buffers
    current_tokens = []
    shard_count = 0
    total_seqs = 0

    # 3. Process Files
    for arrow_file in arrow_files:
        print(f"[i] Processing: {os.path.basename(arrow_file)}")
        
        # Safe loading using Datasets library (Memory Mapped)
        try:
            ds = Dataset.from_file(arrow_file)
        except Exception as e:
            print(f"⚠️ Could not load {arrow_file}: {e}")
            continue

        # Iterate over the dataset
        # We assume columns: 'system_prompt', 'question', 'response'
        for row in tqdm(ds, desc="Tokenizing"):
            sys = row.get('system_prompt', '')
            q = row.get('question', '')
            a = row.get('response', '')
            
            # Format: ChatML
            # <s> <|im_start|>system\n{sys}<|im_end|>\n <|im_start|>user\n{q}<|im_end|>\n <|im_start|>assistant\n{a}<|im_end|> </s>
            
            # Note: We add special tokens manually to prevent splitting issues, 
            # but since we validated the tokenizer is atomic, simple encoding works too.
            # We will construct the text and let tokenizer handle the IDs.
            
            prompt_text = f"<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            full_text = prompt_text + f"{a}<|im_end|>"
            
            # Tokenize (No BOS/EOS added by default in simple encode, so we add them)
            # Warning: check if your tokenizer adds BOS automatically. 
            # We will force add them to be safe.
            
            ids = tok.encode(full_text, add_special_tokens=False)
            ids = [bos_id] + ids + [eos_id]
            
            # Accumulate
            current_tokens.extend(ids)
            
            # 4. Pack into Chunks (Contiguous Packing)
            # This is "Streaming" packing.
            while len(current_tokens) >= SEQ_LEN:
                # Cut a chunk
                chunk = current_tokens[:SEQ_LEN]
                current_tokens = current_tokens[SEQ_LEN:]
                
                # We need a list of chunks to save a shard
                if 'shard_buffer' not in locals(): shard_buffer = []
                shard_buffer.append(chunk)
                
                if len(shard_buffer) >= SHARD_SIZE:
                    # Save Shard
                    save_shard(shard_buffer, shard_count, pad_id)
                    shard_buffer = []
                    shard_count += 1
                    total_seqs += SHARD_SIZE

    # Save remaining
    if 'shard_buffer' in locals() and shard_buffer:
        save_shard(shard_buffer, shard_count, pad_id)
        total_seqs += len(shard_buffer)

    print(f"\n✅ Done! Processed {total_seqs} sequences into {OUTPUT_DIR}")

def save_shard(buffer, idx, pad_id):
    # Convert to Tensor
    input_ids = torch.tensor(buffer, dtype=torch.long)
    
    # Create Labels (Shifted is handled in training, but we need to mask PADs)
    labels = input_ids.clone()
    # Mask PAD tokens with -100 so we don't learn from padding (if any)
    # Note: In contiguous packing, we rarely have pads except maybe last chunk.
    labels[labels == pad_id] = -100
    
    # Save
    out_path = os.path.join(OUTPUT_DIR, f"shard_{idx:04d}")
    os.makedirs(out_path, exist_ok=True)
    torch.save(input_ids, os.path.join(out_path, "input_ids.pt"))
    torch.save(labels, os.path.join(out_path, "labels.pt"))
    # print(f"Saved shard {idx}")

if __name__ == "__main__":
    prepare()