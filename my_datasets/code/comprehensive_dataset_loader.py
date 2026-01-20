#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comprehensive_dataset_loader.py

Comprehensive dataset loader that:
1. Scans both D: and F: drives for all available datasets
2. Identifies duplicates and larger versions
3. Loads data from multiple formats (JSONL, Arrow, Parquet, XML, TXT)
4. Provides detailed analysis and storage estimates
5. Creates a unified dataset for SFT training

Run:
  python comprehensive_dataset_loader.py --base-dir "D:\aria\aria_ai\aria_ai_assistant" --f-drive "F:\aria_ai\aria_ai_assistant"
"""

import os, re, json, csv, sys, argparse, random, hashlib, math
from pathlib import Path
from collections import defaultdict

# Optional deps
try:
    from transformers import AutoTokenizer
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

try:
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except Exception:
    _HAS_PARQUET = False

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs): return it

# Configuration
DATASET_NAMES = {
    "ultrachat": "ultrachat_filtered_dialogue",
    "openorca": "openorca_instruction_following", 
    "openhermes": "openhermes_uncensored_general",
    "fineweb": "fineweb_filtered_general",
    "gutenberg": "gutenberg_literature",
    "wikipedia": "wikipedia_encyclopedia",
    "alterego": "alterego_uncensored_personal",
    "chatgpt_export": "chatgpt_export_personal",
    "financial_news": "financial_news_articles",
    "empathetic_dialogues": "empathetic_dialogues_emotional",
    "financial_phrasebank": "financial_phrasebank_sentiment",
    "reddit_tifu": "reddit_tifu_storytelling",
    "zeroshot_financial": "zeroshot_financial_filtered_financial",
}

def analyze_directories(base_dir: Path, f_drive: Path):
    """Analyze both directories for datasets and identify duplicates"""
    print("[*] Analyzing dataset directories...")
    
    # D Drive analysis
    d_raw = base_dir / "datasets" / "raw"
    d_datasets = {}
    
    if d_raw.exists():
        for item in d_raw.iterdir():
            if item.is_dir():
                size_mb = sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / (1024*1024)
                d_datasets[item.name] = {
                    "path": item,
                    "size_mb": size_mb,
                    "type": "directory"
                }
            elif item.is_file():
                size_mb = item.stat().st_size / (1024*1024)
                d_datasets[item.name] = {
                    "path": item,
                    "size_mb": size_mb,
                    "type": "file"
                }
    
    # F Drive analysis
    f_raw = f_drive / "datasets" / "raw"
    f_datasets = {}
    
    if f_raw.exists():
        for item in f_raw.iterdir():
            if item.is_dir():
                size_mb = sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / (1024*1024)
                f_datasets[item.name] = {
                    "path": item,
                    "size_mb": size_mb,
                    "type": "directory"
                }
            elif item.is_file():
                size_mb = item.stat().st_size / (1024*1024)
                f_datasets[item.name] = {
                    "path": item,
                    "size_mb": size_mb,
                    "type": "file"
                }
    
    # Identify duplicates and larger versions
    duplicates = []
    unique_datasets = {}
    
    for name, info in d_datasets.items():
        if name in f_datasets:
            # Duplicate found
            d_size = info["size_mb"]
            f_size = f_datasets[name]["size_mb"]
            
            if d_size > f_size:
                unique_datasets[name] = {"drive": "D", "info": info, "size_mb": d_size}
                duplicates.append({
                    "name": name,
                    "d_size": d_size,
                    "f_size": f_size,
                    "recommended": "D",
                    "reason": f"D drive larger ({d_size:.1f}MB vs {f_size:.1f}MB)"
                })
            else:
                unique_datasets[name] = {"drive": "F", "info": f_datasets[name], "size_mb": f_size}
                duplicates.append({
                    "name": name,
                    "d_size": d_size,
                    "f_size": f_size,
                    "recommended": "F",
                    "reason": f"F drive larger ({f_size:.1f}MB vs {d_size:.1f}MB)"
                })
        else:
            # Unique to D drive
            unique_datasets[name] = {"drive": "D", "info": info, "size_mb": info["size_mb"]}
    
    # Add F drive unique datasets
    for name, info in f_datasets.items():
        if name not in d_datasets:
            unique_datasets[name] = {"drive": "F", "info": info, "size_mb": info["size_mb"]}
    
    return unique_datasets, duplicates

def load_ultrachat_comprehensive(base_dir: Path):
    """Load UltraChat from all available sources"""
    print("  Loading UltraChat...")
    out = []
    
    # 1. Raw JSONL file
    raw_file = base_dir / "datasets" / "raw" / "ultrachat.jsonl"
    if raw_file.exists():
        print(f"    Processing raw JSONL: {raw_file}")
        raw_samples = load_ultrachat_raw(raw_file)
        out.extend(raw_samples)
        print(f"    Raw JSONL: {len(raw_samples)} samples")
    
    # 2. HF Cache arrow files
    hf_dir = base_dir / "datasets" / "raw" / "hf_cache" / "datasets" / "HuggingFaceH4___ultrachat_200k" / "default" / "0.0.0"
    if hf_dir.exists() and _HAS_PANDAS:
        print(f"    Processing HF cache: {hf_dir}")
        hf_samples = load_ultrachat_hf_cache(hf_dir)
        out.extend(hf_samples)
        print(f"    HF Cache: {len(hf_samples)} samples")
    
    print(f"    Total UltraChat: {len(out)} samples")
    return out

def load_ultrachat_raw(file_path: Path):
    """Load UltraChat from raw JSONL file"""
    out = []
    try:
        with file_path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                try:
                    row = json.loads(line)
                    if "text" in row:
                        text_content = row["text"]
                        if text_content.startswith("<EGO> "):
                            # Use regex to extract prompt and response
                            prompt_match = re.search(r'"prompt":\s*"([^"]*)"', text_content)
                            response_match = re.search(r'"response":\s*"([^"]*)"', text_content)
                            
                            if prompt_match and response_match:
                                prompt = normalize_text(prompt_match.group(1))
                                response = normalize_text(response_match.group(1))
                                if prompt and response and len(prompt) > 10 and len(response) > 10:
                                    out.append({
                                        "source": DATASET_NAMES["ultrachat"],
                                        "type": "chat",
                                        "messages": [
                                            {"role": "user", "content": prompt},
                                            {"role": "assistant", "content": response}
                                        ]
                                    })
                    
                    if i % 100000 == 0:
                        print(f"      Processed {i} lines, found {len(out)} samples...")
                        
                except Exception:
                    continue
    except Exception as e:
        print(f"    Error reading UltraChat file: {e}")
    
    return out

def load_ultrachat_hf_cache(hf_dir: Path):
    """Load UltraChat from HF cache arrow files"""
    out = []
    arrow_files = list(hf_dir.rglob("*.arrow"))
    
    for af in arrow_files:
        try:
            data = pd.read_feather(af)
            print(f"      Processing {af.name} with columns: {list(data.columns)}")
            
            if "messages" in data.columns:
                for _, row in data.iterrows():
                    msgs = row.get("messages", [])
                    if isinstance(msgs, list) and len(msgs) >= 2:
                        norm = []
                        for m in msgs:
                            role = (m.get("role") or "").lower()
                            if role in ("user", "assistant"):
                                content = normalize_text(m.get("content", ""))
                                if content:
                                    norm.append({"role": role, "content": content})
                        if len(norm) >= 2:
                            out.append({
                                "source": DATASET_NAMES["ultrachat"],
                                "type": "chat",
                                "messages": norm
                            })
            
            elif "prompt" in data.columns and "response" in data.columns:
                for _, row in data.iterrows():
                    prompt = normalize_text(row.get("prompt", ""))
                    response = normalize_text(row.get("response", ""))
                    if prompt and response:
                        out.append({
                            "source": DATASET_NAMES["ultrachat"],
                            "type": "chat",
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": response}
                            ]
                        })
        
        except Exception as e:
            print(f"      Error reading {af.name}: {e}")
            continue
    
    return out

def load_openorca_comprehensive(base_dir: Path):
    """Load Open-Orca from all available sources"""
    print("  Loading Open-Orca...")
    out = []
    
    # Raw JSONL file
    raw_file = base_dir / "datasets" / "raw" / "openorca.jsonl"
    if raw_file.exists():
        print(f"    Processing raw JSONL: {raw_file}")
        raw_samples = load_openorca_raw(raw_file)
        out.extend(raw_samples)
        print(f"    Raw JSONL: {len(raw_samples)} samples")
    
    # HF Cache
    hf_dir = base_dir / "datasets" / "raw" / "Open-Orca___open_orca" / "default" / "0.0.0"
    if hf_dir.exists() and _HAS_PANDAS:
        print(f"    Processing HF cache: {hf_dir}")
        hf_samples = load_openorca_hf_cache(hf_dir)
        out.extend(hf_samples)
        print(f"    HF Cache: {len(hf_samples)} samples")
    
    print(f"    Total Open-Orca: {len(out)} samples")
    return out

def load_openorca_raw(file_path: Path):
    """Load Open-Orca from raw JSONL file"""
    out = []
    try:
        with file_path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                try:
                    row = json.loads(line)
                    
                    if "question" in row and "response" in row:
                        question = normalize_text(row.get("question", ""))
                        response = normalize_text(row.get("response", ""))
                        if question and response and len(question) > 10 and len(response) > 10:
                            out.append({
                                "source": DATASET_NAMES["openorca"],
                                "type": "chat",
                                "messages": [
                                    {"role": "user", "content": question},
                                    {"role": "assistant", "content": response}
                                ]
                            })
                    
                    if i % 100000 == 0:
                        print(f"      Processed {i} lines, found {len(out)} samples...")
                        
                except Exception:
                    continue
    except Exception as e:
        print(f"    Error reading Open-Orca file: {e}")
    
    return out

def load_openorca_hf_cache(hf_dir: Path):
    """Load Open-Orca from HF cache arrow files"""
    out = []
    arrow_files = list(hf_dir.rglob("*.arrow"))
    
    for af in arrow_files:
        try:
            data = pd.read_feather(af)
            print(f"      Processing {af.name} with columns: {list(data.columns)}")
            
            if "instruction" in data.columns and "response" in data.columns:
                for _, row in data.iterrows():
                    instruction = normalize_text(row.get("instruction", ""))
                    response = normalize_text(row.get("response", ""))
                    if instruction and response:
                        out.append({
                            "source": DATASET_NAMES["openorca"],
                            "type": "chat",
                            "messages": [
                                {"role": "user", "content": instruction},
                                {"role": "assistant", "content": response}
                            ]
                        })
            
            elif "prompt" in data.columns and "completion" in data.columns:
                for _, row in data.iterrows():
                    prompt = normalize_text(row.get("prompt", ""))
                    completion = normalize_text(row.get("completion", ""))
                    if prompt and completion:
                        out.append({
                            "source": DATASET_NAMES["openorca"],
                            "type": "chat",
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": completion}
                            ]
                        })
        
        except Exception as e:
            print(f"      Error reading {af.name}: {e}")
            continue
    
    return out

def normalize_text(s: str) -> str:
    """Normalize text content"""
    if s is None:
        return ""
    # Strip control chars, collapse whitespace, normalize newlines
    s = re.sub(r'[\u0000-\u0008\u000B-\u000C\u000E-\u001F]', "", str(s))
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r'[ \t\u00A0\u200B\u200C\u200D]+', " ", s)
    return s.strip()

def main():
    parser = argparse.ArgumentParser(description="Comprehensive dataset loader for both D and F drives")
    parser.add_argument("--base-dir", type=str, default=str(Path(__file__).resolve().parents[2]),
                        help="Project root (aria_ai_assistant)")
    parser.add_argument("--f-drive", type=str, default="F:\\aria_ai\\aria_ai_assistant",
                        help="F drive project path")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze directories, don't load data")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    f_drive = Path(args.f_drive)
    
    print(f"[*] Comprehensive Dataset Analysis")
    print(f"    D Drive: {base_dir}")
    print(f"    F Drive: {f_drive}")
    
    # Analyze directories
    unique_datasets, duplicates = analyze_directories(base_dir, f_drive)
    
    # Report findings
    print(f"\n[*] Found {len(unique_datasets)} unique datasets:")
    total_size = 0
    for name, info in sorted(unique_datasets.items(), key=lambda x: x[1]["size_mb"], reverse=True):
        size_mb = info["size_mb"]
        total_size += size_mb
        drive = info["drive"]
        print(f"    {name:40} {size_mb:8.1f} MB ({drive} drive)")
    
    print(f"\n[*] Total dataset size: {total_size:.1f} MB ({total_size/1024:.1f} GB)")
    
    if duplicates:
        print(f"\n[*] Found {len(duplicates)} duplicate datasets:")
        for dup in duplicates:
            print(f"    {dup['name']:40} D: {dup['d_size']:6.1f}MB, F: {dup['f_size']:6.1f}MB -> Use {dup['recommended']} ({dup['reason']})")
    
    if args.analyze_only:
        return
    
    # Load datasets
    print(f"\n[*] Loading datasets...")
    
    # Load major datasets
    datasets = {}
    datasets["ultrachat"] = load_ultrachat_comprehensive(base_dir)
    datasets["openorca"] = load_openorca_comprehensive(base_dir)
    
    # Report loaded data
    print(f"\n[*] Dataset Summary:")
    total_samples = 0
    for name, samples in datasets.items():
        print(f"    {name:20}: {len(samples):8,} samples")
        total_samples += len(samples)
    
    print(f"\n[*] Total samples loaded: {total_samples:,}")
    
    # Storage estimate (rough: 1 sample ≈ 1KB)
    estimated_storage_mb = total_samples / 1024
    print(f"[*] Estimated storage for processed data: {estimated_storage_mb:.1f} MB ({estimated_storage_mb/1024:.1f} GB)")

if __name__ == "__main__":
    main() 