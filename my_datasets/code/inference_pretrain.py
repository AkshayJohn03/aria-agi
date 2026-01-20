import torch
from transformers import PreTrainedTokenizerFast
from model_zia import TinyGPT

CKPT = "artifacts/zia_pretrain_v6/checkpoint_pretrain_step655_1764694406.pt"
TOKENIZER = "artifacts/zia_tokenizer_60k_clean"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# match training config
VOCAB = 60011
D_MODEL = 384
LAYERS = 8
HEADS = 6
MLP = 4
MAX_LEN = 4096
DROP = 0.1


def generate(model, tok, prompt, max_new=100):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)

    with torch.no_grad():
        for _ in range(max_new):
            logits = model(ids)[:, -1, :]
            next_id = torch.argmax(logits, dim=-1).view(1, 1)
            ids = torch.cat([ids, next_id], dim=1)

    return tok.decode(ids[0], skip_special_tokens=True)


def main():
    tok = PreTrainedTokenizerFast.from_pretrained(TOKENIZER, local_files_only=True)
    model = TinyGPT(VOCAB, D_MODEL, LAYERS, HEADS, MLP, MAX_LEN, DROP).to(DEVICE)

    ck = torch.load(CKPT, map_location=DEVICE)["model"]
    model.load_state_dict(ck, strict=False)
    model.eval()

    print("[Loaded]")
    print(CKPT)

    prompt = "Explain the importance of clean water."
    out = generate(model, tok, prompt, max_new=80)
    print(out)


if __name__ == "__main__":
    main()
