import torch
from transformers import AutoTokenizer
from model.zia_model import ZIATransformerLM  # adjust if filename differs

# ---------------- CONFIG ----------------
CHECKPOINT_PATH = "artifacts/zia_lm_v20/v20_step977_t1765969109.pt"
TOKENIZER_PATH = "artifacts/zia_tokenizer_60k_clean"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
TOP_P = 0.9
REPETITION_PENALTY = 1.1

PROMPT = (
    "The theory of evolution explains that"
)

# ----------------------------------------

def main():
    print("[i] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        use_fast=True
    )

    print("[i] Loading model...")
    model = ZIATransformerLM.from_pretrained(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.eval()
    model.to(DEVICE)

    input_ids = tokenizer(
        PROMPT,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    output_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    print("\n===== MODEL OUTPUT =====\n")
    print(output_text)
    print("\n========================\n")

if __name__ == "__main__":
    main()
