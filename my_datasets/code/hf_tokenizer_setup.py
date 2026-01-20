# hf_tokenizer_setup_fixed.py
import os
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

TOKENIZER_JSON = "artifacts/hf_tokenizer_mistral/tokenizer.json"
OUT_DIR = "artifacts/hf_tokenizer_mistral"

def build_and_verify():
    tok_native = Tokenizer.from_file(TOKENIZER_JSON)
    print("Loaded native tokenizers.Tokenizer — vocab_size:", tok_native.get_vocab_size())

    hf = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_JSON)

    # Ensure special tokens exist and are registered
    to_add = {}
    if hf.pad_token is None:
        to_add["pad_token"] = "<pad>"
    if hf.bos_token is None:
        # some HF tokenizers use <s>
        if "<s>" in hf.get_vocab():
            to_add["bos_token"] = "<s>"
    if hf.eos_token is None:
        if "</s>" in hf.get_vocab():
            to_add["eos_token"] = "</s>"

    # Add role tokens as "additional_special_tokens" if they are missing
    role_tokens = ["<|user|>","<|assistant|>","<|system|>","<|im_start|>","<|im_end|>"]
    missing_roles = [t for t in role_tokens if hf.get_vocab().get(t) is None]
    if missing_roles:
        hf.add_special_tokens({"additional_special_tokens": missing_roles})
        print("Added missing role tokens:", missing_roles)

    if to_add:
        hf.add_special_tokens(to_add)
        print("Added special tokens:", to_add)

    hf.save_pretrained(OUT_DIR)
    print("Saved PreTrainedTokenizerFast to", OUT_DIR)
    print("Tokenizer config:", hf.special_tokens_map)

    # --- Verification (atomicity & round-trip)
    tests = role_tokens
    for t in tests:
        enc = hf.encode(t)
        # encode returns either list of ids or BatchEncoding — normalize to list
        ids = enc.ids if hasattr(enc, "ids") else (enc if isinstance(enc, list) else enc["input_ids"])
        print(f"{t} -> ids: {ids} | atomic: {len(ids) == 1}")

    s = "<|im_start|>user\nHello world<|im_end|>"
    enc = hf.encode(s)
    ids = enc.ids if hasattr(enc, "ids") else (enc if isinstance(enc, list) else enc["input_ids"])
    dec = hf.decode(ids)
    print("orig:", repr(s))
    print("decoded:", repr(dec))
    print("roundtrip exact:", s == dec)

if __name__ == "__main__":
    build_and_verify()
