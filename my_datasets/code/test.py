import torch
from transformers import AutoTokenizer
from my_model import TinyGPT
import re

def clean_output(text):
    # strip control chars, weird unicode boxes, and normalise whitespace
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_from_ckpt(ckpt_path, prompt, max_new_tokens=200, temperature=0.9, top_k=40):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained("artifacts/zia_tokenizer_60k")
    if tok.pad_token is None: tok.add_special_tokens({"pad_token": "<pad>"})
    vocab = len(tok)
    ck = torch.load(ckpt_path, map_location='cpu')
    ctx_len = ck.get("context_len", 4096) if isinstance(ck, dict) else 4096
    model = TinyGPT(vocab, 384, 8, 6, 4, ctx_len, 0.1, tok.pad_token_id).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(ids)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            if top_k:
                values, indices = torch.topk(logits, top_k)
                probs = torch.softmax(values, dim=-1)
                next_token = indices[0, torch.multinomial(probs, 1)]
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_token.unsqueeze(0)], dim=1)
    out = tok.decode(ids[0].tolist(), skip_special_tokens=True)
    return clean_output(out)

if __name__ == "__main__":
    out = generate_from_ckpt("artifacts/zia_ift_v4_longctx/checkpoints/checkpoint_auto_step6004.pt",
                             "In summary, the best approach to handle long context is")
    print(out)
