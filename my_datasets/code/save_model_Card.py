import json, os
from datetime import datetime

out_dir = "artifacts/zia_dense_runs"
ckpt = os.path.join(out_dir, "best_val", "checkpoint.pt")
meta = {
    "model_name": "Zia",
    "organization": "ARIA",
    "founder": "Akshay John",
    "description": "Zia - dense TinyGPT trained on merged dataset (arrow format).",
    "date_saved": datetime.utcnow().isoformat() + "Z",
    "notes": "Trained with TinyGPT dense architecture. tokenizer: artifacts/zia_tokenizer_60k",
}
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "model_card.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("[i] model_card.json saved to", out_dir)
