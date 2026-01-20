#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download missing datasets and a few recommended extras (<10GB each) into the existing cache.
It reuses the project's HF cache at datasets/raw/hf_cache.

Run:
  python datasets/code/download_missing_and_extra.py
"""
import os
from pathlib import Path
from typing import Optional

def set_hf_cache(project_root: Path) -> None:
    hf_home = project_root / "datasets" / "raw" / "hf_cache"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home))
    # Keep TRANSFORMERS_CACHE consistent as well
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home))
    print(f"[cache] HF_HOME set to: {os.environ['HF_HOME']}")


def download_dataset(name: str, config: Optional[str] = None, note: str = "") -> None:
    from datasets import load_dataset
    try:
        print(f"\n[download] {name} {f'({config})' if config else ''} {note}")
        if config:
            ds = load_dataset(name, config)
        else:
            ds = load_dataset(name)
        # Touch all splits to materialize cache
        if isinstance(ds, dict):
            for split, d in ds.items():
                print(f"  - split: {split}, num_rows: {len(d)}")
                # Read first row to ensure materialization
                _ = d[0] if len(d) else None
        else:
            print(f"  - split: train, num_rows: {len(ds)}")
            _ = ds[0] if len(ds) else None
        print(f"[ok] {name} done.")
    except Exception as e:
        print(f"[warn] Failed to download {name}{' ('+str(config)+')' if config else ''}: {e}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    set_hf_cache(project_root)

    # Missing datasets (small/medium)
    targets = [
        ("empathetic_dialogues", None, "small conversation dataset"),
        ("financial_phrasebank", "sentences_allagree", "finance sentiment (agreed)"),
        ("reddit_tifu", "short", "TIFU short stories"),
        ("HuggingFaceH4/ultrachat_200k", None, "UltraChat 200k (parquet shards)"),
    ]

    # Recommended extras (<10GB each)
    extras = [
        ("databricks/databricks-dolly-15k", None, "instruction dataset (15k)"),
        ("yahma/alpaca-cleaned", None, "cleaned alpaca 52k"),
        ("vicgalle/alpaca-gpt4", None, "alpaca-gpt4 small"),
    ]

    print("\n[*] Downloading missing datasets...")
    for n, c, note in targets:
        download_dataset(n, c, note)

    print("\n[*] Downloading recommended extras (small)...")
    for n, c, note in extras:
        download_dataset(n, c, note)

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
