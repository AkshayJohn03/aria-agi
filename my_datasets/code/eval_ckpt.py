# checks/eval_ckpt.py
import torch, math
from datasets import load_from_disk
from transformers import AutoTokenizer
from train_zia_dense import TinyGPT # Relying on the original import
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ================================
# CONFIGURATION
# ================================
ckpt = "artifacts/zia_dense_runs/checkpoints/interrupt_step_189482.pt"  # choose candidate checkpoint
tokenizer_path = "artifacts/zia_tokenizer_60k"

# FIX: Point to the 'train' split directory directly. 
# This assumes 'train' contains the actual loadable Dataset object.
val_path = "my_datasets/processed/arrow_cleaned_v1/train" 

# ================================
# MODEL & DATA SETUP
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.add_special_tokens({"pad_token":"<pad>"})

model = TinyGPT(vocab_size=len(tokenizer), d_model=384, n_layers=8, n_heads=6, mlp_ratio=4, max_len=256, dropout=0.1).to(device)

# Standard torch.load (removed deprecated 'with' block)
ck = torch.load(ckpt, map_location=device)

model.load_state_dict(ck.get("model", ck.get("model_state_dict", ck)), strict=False)
model.eval()
pad = tokenizer.pad_token_id

# FIX: Load the single Dataset directly from the 'train' directory.
val = load_from_disk(val_path)
print(f"Successfully loaded Dataset from '{val_path}' (size: {len(val)}).")


# ================================
# EVALUATION LOOP
# ================================
batch_size = 8

def collate(batch):
    texts=[(x.get("text") if isinstance(x, dict) else str(x)) for x in batch]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]

dl = DataLoader(val, batch_size=batch_size, collate_fn=collate)
total_loss=0.0; n=0
for i,(ids,mask) in enumerate(dl):
    ids,mask = ids.to(device), mask.to(device)
    
    # Target creation (IDs shifted by 1 position for next token prediction)
    targets = ids[:,1:].contiguous().view(-1)
    
    with torch.no_grad():
        # model should return (logits, _)
        logits,_ = model(ids, attention_mask=mask) 
        
        # Cross Entropy Loss calculation
        loss = F.cross_entropy(
            logits[:,:-1].contiguous().view(-1, logits.size(-1)), # Logits for tokens [0] to [L-2]
            targets, # Targets for tokens [1] to [L-1]
            ignore_index=pad
        )
        
    total_loss += loss.item(); n+=1
    if i>=200: break
    
avg = total_loss/n
print(f"avg_val_loss={avg:.6f} ppl={math.exp(avg):.3f}")