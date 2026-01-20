import os, math, torch
import glob
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset, concatenate_datasets

def load_split_from_shards(split_dir: str):
    """
    Load dataset split from disk, supporting both single-directory
    and sharded-directory structures.
    """
    if not os.path.exists(split_dir):
        return None
    
    # Check if the split directory itself is a valid dataset directory
    if os.path.exists(os.path.join(split_dir, "dataset_info.json")):
        print(f"[i] Loading single dataset from {split_dir}")
        return load_from_disk(split_dir)
        
    # If not, assume it's a sharded dataset structure.
    # Find all data-*.arrow files recursively.
    data_files = glob.glob(os.path.join(split_dir, "**", "data-*.arrow"), recursive=True)
    
    if not data_files:
        print(f"[!] No .arrow files found in {split_dir}. Returning None.")
        return None
        
    print(f"[i] Loading dataset from {len(data_files)} .arrow files in {split_dir}")
    # Use the 'arrow' format to load a dataset from a list of files.
    ds = load_dataset("arrow", data_files={"train": data_files})
    
    # Since load_dataset returns a DatasetDict, we need to extract the 'train' split.
    return ds["train"]

def run_diagnostics():
    cfg = ZiaConfig()
    device = torch.device(cfg.device)
    print(f"[i] Device: {device}")

    # --- Tokenizer ---
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    eos_token = tok.eos_token or "</s>"
    print(f"[i] Tokenizer vocab={len(tok)} | pad_id={tok.pad_token_id} | eos_id={tok.eos_token_id}")

    # --- Dataset ---
    train_ds = load_split_from_shards(os.path.join(cfg.dataset_path, "train"))
    val_ds = load_split_from_shards(os.path.join(cfg.dataset_path, "validation"))
    if val_ds is None:
        val_ds = load_split_from_shards(os.path.join(cfg.dataset_path, "test"))
    
    if train_ds is None:
        raise FileNotFoundError("No train split found! Please check your dataset path.")
    if val_ds is None:
        raise FileNotFoundError("No validation or test split found! Please check your dataset path.")

    collate = CollateWrapper(tok, eos_token, cfg.max_len)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=2, collate_fn=collate)
    print(f"[i] Validation size={len(val_ds)}")

    # --- Model ---
    model = TinyGPT(
        vocab_size=len(tok),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        mlp_ratio=cfg.mlp_ratio,
        max_len=cfg.max_len,
        dropout=cfg.dropout,
    ).to(device)
    model.pad_token_id = tok.pad_token_id

    # --- Load checkpoint ---
    ckpt_path = os.path.join(cfg.output_dir, "best_val", "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"[i] Loaded checkpoint step={ckpt.get('step')}")
    model.load_state_dict(ckpt["model"])

    # --- Quick val eval ---
    small_val_loss = evaluate(model, val_loader, device, max_batches=50)
    print(f"[diag] quick eval loss={small_val_loss:.4f}, ppl={math.exp(small_val_loss):.2f}")

    # --- Decode sample (clean) ---
    ids, mask = next(iter(val_loader))
    decoded_text = tok.decode(ids[0].tolist()[:80], skip_special_tokens=True)
    print("[diag] decoded val sample:", decoded_text)
    print("[diag] attention mask sum:", mask[0].sum().item())

if __name__ == "__main__":
    from datasets import load_dataset
    from train_zia_dense_final import TinyGPT, ZiaConfig, CollateWrapper, evaluate
    try:
        run_diagnostics()
    except FileNotFoundError as e:
        print(f"Error: {e}")
