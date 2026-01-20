# datasets/code/eval_quick.py
import math, torch, argparse
from datasets import load_from_disk
from transformers import AutoTokenizer
from train_zia_dense import TinyGPT

@torch.no_grad()
def compute_ppl(model, tokenizer, dataset, device="cuda", max_count=1000):
    model.eval()
    total_loss = 0.0
    n = 0
    for i, ex in enumerate(dataset):
        if i >= max_count:
            break
        text = "\n".join([m.get("content","") for m in ex["messages"]]) + (tokenizer.eos_token or "</s>")
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=model.max_len).to(device)
        logits = model(enc["input_ids"], attention_mask=enc["attention_mask"])
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = enc["input_ids"][:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0)
        total_loss += loss.item()
        n += 1
    avg_loss = total_loss / max(1, n)
    return avg_loss, math.exp(avg_loss)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="datasets/processed/arrow_dataset")
    parser.add_argument("--tokenizer_path", default="artifacts/zia_tokenizer_60k")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_eval_samples", type=int, default=500)
    args = parser.parse_args()

    ds = load_from_disk(args.dataset_path)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    ckpt_path = "artifacts/zia_dense_runs/best_val/checkpoint.pt"  # or use find_last_checkpoint
    import os
    if not os.path.exists(ckpt_path):
        # fallback to find most recent
        from save_model_for_inference import find_last_checkpoint
        ckpt_path = find_last_checkpoint("artifacts/zia_dense_runs")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = TinyGPT(vocab_size=len(tok), d_model=384, n_layers=8, n_heads=6, mlp_ratio=4, max_len=256, dropout=0.1)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)

    avg_loss, ppl = compute_ppl(model, tok, ds["test"], device=args.device, max_count=args.max_eval_samples)
    print(f"[i] avg_loss={avg_loss:.4f} | ppl={ppl:.2f} on {args.max_eval_samples} samples")

if __name__ == "__main__":
    main()
