import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

# --- CONFIG ---
# We use the best checkpoint you identified
CKPT_PATH = "artifacts/zia_ift_v5/checkpoint_autosave_step3316.pt"
TOKENIZER_DIR = "artifacts/zia_tokenizer_60k_clean"

# Must match training exactly
VOCAB_SIZE = 60011 
D_MODEL = 384
N_LAYERS = 8
N_HEADS = 6
MAX_LEN = 4096

# Generation Params
TEMP = 0.7
TOP_K = 50
MAX_NEW_TOKENS = 200
# ----------------

# --- MODEL DEFINITION (Must match training) ---
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, 
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)

def load_model(device):
    print(f"[i] Loading model architecture (Vocab: {VOCAB_SIZE})...")
    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, MAX_LEN).to(device)
    
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Error: Checkpoint not found at {CKPT_PATH}")
        exit()
        
    print(f"[i] Loading weights from {os.path.basename(CKPT_PATH)}...")
    ck = torch.load(CKPT_PATH, map_location=device)
    
    # Handle different checkpoint structures
    state_dict = ck.get("model", ck)
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print("[✅] Weights loaded successfully.")
    except Exception as e:
        print(f"[⚠️] Strict load failed: {e}")
        print("Attempting strict=False...")
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    return model

def generate(model, tok, prompt, device):
    # 1. Format as ChatML
    # <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # 2. Encode
    ids = tok.encode(formatted_prompt, add_special_tokens=False)
    # Add BOS if not present (Mistral/Llama usually like a BOS at start)
    if tok.bos_token_id:
        ids = [tok.bos_token_id] + ids
        
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    # 3. Generate Loop
    generated = []
    print("\nZia: ", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            # Crop context if too long
            x_cond = x if x.size(1) <= MAX_LEN else x[:, -MAX_LEN:]
            
            logits = model(x_cond)
            logits = logits[:, -1, :] / TEMP
            
            # Top-K Sampling
            if TOP_K > 0:
                v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop if EOS or <|im_end|> (ID 60008 based on your logs)
            idx = next_token.item()
            if idx == tok.eos_token_id or idx == 60008: 
                break
                
            # Decode and print stream
            decoded_token = tok.decode([idx], skip_special_tokens=True)
            print(decoded_token, end="", flush=True)
            
            generated.append(idx)
            x = torch.cat((x, next_token), dim=1)
            
    print("\n")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[i] Using device: {device}")
    
    # Load Tokenizer
    print(f"[i] Loading tokenizer from {TOKENIZER_DIR}...")
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    except Exception as e:
        print(f"❌ Tokenizer load failed: {e}")
        return

    model = load_model(device)
    
    print("="*40)
    print(" ZIA v5 Chat Interface (ChatML)")
    print(" Type 'exit' to quit.")
    print("="*40)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue
            
        generate(model, tok, user_input, device)

if __name__ == "__main__":
    main()