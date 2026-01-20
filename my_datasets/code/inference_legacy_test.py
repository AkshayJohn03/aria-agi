import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

CKPT_PATH = "artifacts/zia_ift_v5/zia_v5_aligned.pt"
TOKENIZER_DIR = "artifacts/zia_tokenizer_60k_clean"

VOCAB_SIZE = 60011
D_MODEL = 384
N_LAYERS = 8
N_HEADS = 6
MAX_LEN = 4096

TEMP = 0.8
TOP_K = 50
MAX_NEW_TOKENS = 150

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model*4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))


def generate_reply(model, tok, prompt, device):
    ids = tok.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    print("\nZia: ", end="", flush=True)

    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            logits = model(x)[:, -1, :] / TEMP

            # top-k filtering
            if TOP_K:
                v, _ = torch.topk(logits, TOP_K)
                logits[logits < v[:, -1].unsqueeze(-1)] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

        # stop if encountering <|im_end|>
        if next_id == tok.convert_tokens_to_ids("<|im_end|>"):
            break

        token = tok.decode([next_id], skip_special_tokens=True)
        print(token, end="", flush=True)

        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)

    print("\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)

    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, MAX_LEN).to(device)
    ck = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    print("ZIA v5 Inference Ready.\n")

    while True:
        user = input("You: ")
        if user.strip().lower() == "exit":
            break

        chatml_prompt = (
            "<|im_start|>user\n" +
            user.strip() +
            "\n<|im_end|>\n<|im_start|>assistant\n"
        )

        generate_reply(model, tok, chatml_prompt, device)

if __name__ == "__main__":
    main()
