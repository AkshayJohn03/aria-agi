# -*- coding: utf-8 -*-
"""
process_missing_datasets.py

This script processes ONLY the missing datasets that weren't loaded in the main script:
- UltraChat (from raw JSONL)
- Open-Orca (from raw JSONL) 
- Empathetic Dialogues
- Financial Phrasebank
- Reddit TIFU
- ChatGPT Export
- Wikipedia (XML parsing)
- Gutenberg (RDF parsing)

Then merges them with existing organized data to create the final complete dataset.

Run:
  python process_missing_datasets.py --base-dir "D:\\aria\\aria_ai\\aria_ai_assistant" --f-drive "F:\\aria_ai\\aria_ai_assistant" --tokenizer gpt2 --context 8192
"""

import os, re, json, csv, sys, gzip, argparse, random, hashlib, math, bz2, tarfile
from pathlib import Path
import xml.etree.ElementTree as ET

# Optional deps: transformers, pandas, pyarrow, tqdm
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

# Hugging Face datasets (for API-based fallback downloads/loading)
try:
    from datasets import load_dataset
    _HAS_HF_DATASETS = True
except Exception:
    _HAS_HF_DATASETS = False

def ensure_hf_cache(base_dir: Path) -> None:
    """Point HF caches to the project's hf_cache folder so downloads go to datasets/raw/hf_cache."""
    hf_home = base_dir / "datasets" / "raw" / "hf_cache"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home))

# -----------------------------
# Config
# -----------------------------
DATASET_NAMES = {
    "ultrachat": "ultrachat_filtered_dialogue",
    "openorca": "openorca_instruction_following", 
    "empathetic_dialogues": "empathetic_dialogues_emotional",
    "financial_phrasebank": "financial_phrasebank_sentiment",
    "reddit_tifu": "reddit_tifu_storytelling",
    "chatgpt_export": "chatgpt_export_personal",
    "wikipedia": "wikipedia_encyclopedia",
    "gutenberg": "gutenberg_literature",
}

RNG_SEED = 42

# -----------------------------
# Utilities
# -----------------------------
WS_RE = re.compile(r"[ \t\u00A0\u200B\u200C\u200D]+")
CTRL_RE = re.compile(r"[\u0000-\u0008\u000B-\u000C\u000E-\u001F]")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    # Strip control chars, collapse whitespace, normalize newlines
    s = CTRL_RE.sub("", str(s))
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = WS_RE.sub(" ", s)
    return s.strip()

def hash_chat(messages) -> str:
    # Canonical hash for dedupe
    parts = []
    for m in messages:
        role = m.get("role","").lower()
        content = normalize_text(m.get("content",""))
        if not content:
            continue
        parts.append(f"{role}:{content}")
    h = hashlib.sha1(("||".join(parts)).encode("utf-8")).hexdigest()
    return h

def hash_doc(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_jsonl(path: Path, rows_iter):
    with path.open("w", encoding="utf-8") as f:
        for r in rows_iter:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def approx_token_count(text: str) -> int:
    # Fallback heuristic: ~4 chars/token (OpenAI rule of thumb)
    n = len(text)
    return max(1, n // 4)

class TokenEstimator:
    def __init__(self, name_or_path: str | None, context_len: int):
        self.context = int(context_len)
        self.name = name_or_path
        self.ok = False
        self.tokenizer = None
        if _HAS_TRANSFORMERS and name_or_path:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, local_files_only=False, use_fast=True)
                if not self.tokenizer.pad_token:
                    # Reasonable default
                    self.tokenizer.pad_token = self.tokenizer.eos_token or "<|endoftext|>"
                self.ok = True
            except Exception:
                self.ok = False

    def count_messages(self, messages) -> int:
        if self.ok:
            # Convert to a single prompt text (role-tagged)
            text = "\n".join([f"{m.get('role','')}: {m.get('content','')}" for m in messages])
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Fallback heuristic
        text = "\n".join([f"{m.get('role','')}: {m.get('content','')}" for m in messages])
        return approx_token_count(text)

# -----------------------------
# Missing Dataset Loaders
# -----------------------------
def load_ultrachat_raw(base_dir: Path):
    """
    Load UltraChat from raw JSONL file
    """
    raw_file = base_dir / "datasets" / "raw" / "ultrachat.jsonl"
    if not raw_file.exists():
        print(f"UltraChat file not found at: {raw_file}")
        return []
    
    print(f"Loading UltraChat from: {raw_file}")
    out = []
    try:
        with raw_file.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                try:
                    row = json.loads(line)
                    
                    # Handle the nested format: {"text": "<EGO> {...}"}
                    if "text" in row:
                        text_content = row["text"]
                        # Extract the JSON part after "<EGO> "
                        if text_content.startswith("<EGO> "):
                            try:
                                # Use regex to extract prompt and response more robustly
                                import re
                                
                                # Look for prompt field
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
                                
                                # Also try to parse as JSON if possible
                                ego_part = text_content[6:]  # Skip "<EGO> "
                                try:
                                    ego_data = json.loads(ego_part)
                                    prompt = normalize_text(ego_data.get("prompt", ""))
                                    response = normalize_text(ego_data.get("response", ""))
                                    if prompt and response and len(prompt) > 10 and len(response) > 10:
                                        out.append({
                                            "source": DATASET_NAMES["ultrachat"],
                                            "type": "chat",
                                            "messages": [
                                                {"role": "user", "content": prompt},
                                                {"role": "assistant", "content": response}
                                            ]
                                        })
                                except:
                                    pass  # JSON parsing failed, regex already tried
                                    
                            except Exception as e:
                                if i < 5:  # Only show first few errors
                                    print(f"    Error parsing EGO data on line {i+1}: {e}")
                                continue
                    
                    # Also try direct prompt/response format
                    elif "prompt" in row and "response" in row:
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
                    
                    # Progress indicator
                    if i % 10000 == 0:
                        print(f"  Processed {i} lines, found {len(out)} samples...")
                        
                except Exception as e:
                    if i < 5:  # Only show first few errors
                        print(f"    Error on line {i}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading UltraChat file: {e}")
    
    print(f"UltraChat: Loaded {len(out)} samples from {raw_file}")
    return out

def load_openorca_raw(base_dir: Path):
    """
    Load Open-Orca from raw JSONL file
    """
    raw_file = base_dir / "datasets" / "raw" / "openorca.jsonl"
    if not raw_file.exists():
        print(f"Open-Orca file not found at: {raw_file}")
        return []
    
    print(f"Loading Open-Orca from: {raw_file}")
    out = []
    try:
        with raw_file.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                try:
                    row = json.loads(line)
                    
                    # Handle the actual format: {"id": "...", "system_prompt": "", "question": "..."}
                    if "question" in row:
                        question = normalize_text(row.get("question", ""))
                        # Look for answer in various possible fields
                        answer = normalize_text(row.get("answer", row.get("response", row.get("completion", ""))))
                        
                        if question and answer:
                            out.append({
                                "source": DATASET_NAMES["openorca"],
                                "type": "chat",
                                "messages": [
                                    {"role": "user", "content": question},
                                    {"role": "assistant", "content": answer}
                                ]
                            })
                    
                    # Also try instruction/response format
                    elif "instruction" in row and "response" in row:
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
                    
                    # Progress indicator
                    if i % 10000 == 0:
                        print(f"  Processed {i} lines, found {len(out)} samples...")
                        
                except Exception as e:
                    if i < 5:  # Only show first few errors
                        print(f"    Error on line {i}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading Open-Orca file: {e}")
    
    print(f"Open-Orca: Loaded {len(out)} samples from {raw_file}")
    return out

def load_empathetic_dialogues(base_dir: Path):
    """
    Empathetic Dialogues dataset
    """
    root = base_dir / "datasets" / "raw" / "hf_cache" / "hub" / "datasets--empathetic_dialogues" / "snapshots"
    if not root.exists():
        print(f"Empathetic Dialogues directory not found at: {root}")
        return []
    
    print(f"Loading Empathetic Dialogues from: {root}")
    out = []
    
    # Look for data files in snapshots
    for snapshot_dir in root.iterdir():
        if snapshot_dir.is_dir():
            print(f"  Checking snapshot: {snapshot_dir.name}")
            # Look for arrow files or other data formats
            data_files = list(snapshot_dir.rglob("*.arrow"))
            if not data_files:
                print(f"    No arrow files found in {snapshot_dir.name}")
                continue
                
            print(f"    Found {len(data_files)} arrow files")
            for df in data_files:
                try:
                    if df.suffix == '.arrow':
                        data = pd.read_feather(df)
                    else:
                        continue
                except Exception as e:
                    print(f"    Error reading {df.name}: {e}")
                    continue
                
                print(f"    Processing {df.name} with columns: {list(data.columns)}")
                # Look for conversation columns
                if "context" in data.columns and "response" in data.columns:
                    for _, row in data.iterrows():
                        context = normalize_text(row.get("context", ""))
                        response = normalize_text(row.get("response", ""))
                        if context and response:
                            out.append({
                                "source": DATASET_NAMES["empathetic_dialogues"],
                                "type": "chat",
                                "messages": [
                                    {"role": "user", "content": context},
                                    {"role": "assistant", "content": response}
                                ]
                            })
                elif "input" in data.columns and "output" in data.columns:
                    for _, row in data.iterrows():
                        input_text = normalize_text(row.get("input", ""))
                        output = normalize_text(row.get("output", ""))
                        if input_text and output:
                            out.append({
                                "source": DATASET_NAMES["empathetic_dialogues"],
                                "type": "chat",
                                "messages": [
                                    {"role": "user", "content": input_text},
                                    {"role": "assistant", "content": output}
                                ]
                            })
    
    print(f"Empathetic Dialogues: Loaded {len(out)} samples")
    return out

def load_financial_phrasebank(base_dir: Path):
    """
    Financial Phrasebank dataset
    """
    root = base_dir / "datasets" / "raw" / "hf_cache" / "hub" / "datasets--financial_phrasebank" / "snapshots"
    if not root.exists():
        return []
    
    out = []
    
    # Look for data files in snapshots
    for snapshot_dir in root.iterdir():
        if snapshot_dir.is_dir():
            # Look for CSV files
            data_files = list(snapshot_dir.rglob("*.csv"))
            
            for df in data_files:
                try:
                    with df.open("r", encoding="utf-8", newline="") as fh:
                        reader = csv.DictReader(fh)
                        for row in reader:
                            # Look for text and sentiment columns
                            text = normalize_text(row.get("text", row.get("sentence", "")))
                            sentiment = normalize_text(row.get("sentiment", row.get("label", "")))
                            if text and sentiment:
                                out.append({
                                    "source": DATASET_NAMES["financial_phrasebank"],
                                    "type": "chat",
                                    "messages": [
                                        {"role": "user", "content": f"Classify the sentiment of this financial text:\n\n{text}"},
                                        {"role": "assistant", "content": sentiment}
                                    ]
                                })
                except Exception:
                    continue
            
            # Also look for arrow files
            arrow_files = list(snapshot_dir.rglob("*.arrow"))
            if arrow_files and _HAS_PANDAS:
                for af in arrow_files:
                    try:
                        if af.suffix == '.arrow':
                            data = pd.read_feather(af)
                        else:
                            continue
                    except Exception:
                        continue
                    
                    # Look for text and sentiment columns
                    if "text" in data.columns and "sentiment" in data.columns:
                        for _, row in data.iterrows():
                            text = normalize_text(row.get("text", ""))
                            sentiment = normalize_text(row.get("sentiment", ""))
                            if text and sentiment:
                                out.append({
                                    "source": DATASET_NAMES["financial_phrasebank"],
                                    "type": "chat",
                                    "messages": [
                                        {"role": "user", "content": f"Classify the sentiment of this financial text:\n\n{text}"},
                                        {"role": "assistant", "content": sentiment}
                                    ]
                                })
    
    return out

def load_reddit_tifu(base_dir: Path):
    """
    Reddit TIFU dataset - storytelling
    """
    root = base_dir / "datasets" / "raw" / "hf_cache" / "hub" / "datasets--reddit_tifu" / "snapshots"
    if not root.exists():
        return []
    
    out = []
    
    # Look for data files in snapshots
    for snapshot_dir in root.iterdir():
        if snapshot_dir.is_dir():
            data_files = list(snapshot_dir.rglob("*.arrow"))
            if not data_files and _HAS_PANDAS:
                continue
                
            for df in data_files:
                try:
                    if df.suffix == '.arrow':
                        data = pd.read_feather(df)
                    else:
                        continue
                except Exception:
                    continue
                
                # Look for story and title columns
                if "story" in data.columns and "title" in data.columns:
                    for _, row in data.iterrows():
                        title = normalize_text(row.get("title", ""))
                        story = normalize_text(row.get("story", ""))
                        if title and story:
                            out.append({
                                "source": DATASET_NAMES["reddit_tifu"],
                                "type": "chat",
                                "messages": [
                                    {"role": "user", "content": f"Tell me a story about: {title}"},
                                    {"role": "assistant", "content": story}
                                ]
                            })
                elif "text" in data.columns and "title" in data.columns:
                    for _, row in data.iterrows():
                        title = normalize_text(row.get("title", ""))
                        text = normalize_text(row.get("text", ""))
                        if title and text:
                            out.append({
                                "source": DATASET_NAMES["reddit_tifu"],
                                "type": "chat",
                                "messages": [
                                    {"role": "user", "content": f"Tell me a story about: {title}"},
                                    {"role": "assistant", "content": text}
                                ]
                            })
    
    return out

# -----------------------------
# API loaders (download+load via datasets library)
# -----------------------------
def load_ultrachat_api(base_dir: Path):
    out = []
    if not _HAS_HF_DATASETS:
        return out
    ensure_hf_cache(base_dir)
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k")
    except Exception:
        return out
    splits = [k for k in ds.keys()]
    for sp in splits:
        d = ds[sp]
        cols = set(d.column_names)
        if "messages" in cols:
            for rec in d:
                msgs = rec.get("messages") or []
                norm = []
                if isinstance(msgs, list):
                    for m in msgs:
                        role = (m.get("role") or "").lower()
                        if role in ("user","assistant"):
                            content = normalize_text(m.get("content",""))
                            if content:
                                norm.append({"role": role, "content": content})
                if len(norm) >= 2:
                    out.append({"source": DATASET_NAMES["ultrachat"], "type":"chat", "messages": norm})
        elif "prompt" in cols and "response" in cols:
            for rec in d:
                u = normalize_text(rec.get("prompt",""))
                a = normalize_text(rec.get("response",""))
                if u and a:
                    out.append({"source": DATASET_NAMES["ultrachat"], "type":"chat", "messages":[{"role":"user","content":u},{"role":"assistant","content":a}]})
    return out

def load_empathetic_dialogues_api(base_dir: Path):
    out = []
    if not _HAS_HF_DATASETS:
        return out
    ensure_hf_cache(base_dir)
    try:
        ds = load_dataset("empathetic_dialogues")
    except Exception:
        return out
    for sp, d in ds.items():
        cols = set(d.column_names)
        # Common fields: context, utterance (or 'response')
        ctx_key = "context" if "context" in cols else None
        rsp_key = "utterance" if "utterance" in cols else ("response" if "response" in cols else None)
        if not ctx_key or not rsp_key:
            continue
        for rec in d:
            u = normalize_text(rec.get(ctx_key, ""))
            a = normalize_text(rec.get(rsp_key, ""))
            if u and a:
                out.append({"source": DATASET_NAMES["empathetic_dialogues"], "type":"chat", "messages":[{"role":"user","content":u},{"role":"assistant","content":a}]})
    return out

def load_financial_phrasebank_api(base_dir: Path):
    out = []
    if not _HAS_HF_DATASETS:
        return out
    ensure_hf_cache(base_dir)
    # Prefer sentences_allagree config (smaller/cleaner)
    configs = ["sentences_allagree", "sentences_50agree", "sentences_66agree", "sentences_75agree"]
    ds = None
    for cfg in configs:
        try:
            ds = load_dataset("financial_phrasebank", cfg)
            break
        except Exception:
            continue
    if ds is None:
        return out
    for sp, d in ds.items():
        cols = set(d.column_names)
        text_key = "text" if "text" in cols else ("sentence" if "sentence" in cols else None)
        lab_key = "label" if "label" in cols else ("sentiment" if "sentiment" in cols else None)
        if not text_key or not lab_key:
            continue
        for rec in d:
            text = normalize_text(rec.get(text_key, ""))
            lab = rec.get(lab_key)
            # map labels to strings
            lab_str = str(lab)
            if isinstance(lab, int):
                lab_map = {0:"negative", 1:"neutral", 2:"positive"}
                lab_str = lab_map.get(lab, str(lab))
            elif isinstance(lab, str) and lab.lower() in ("positive","negative","neutral"):
                lab_str = lab.lower()
            if text and lab_str:
                out.append({"source": DATASET_NAMES["financial_phrasebank"], "type":"chat", "messages":[
                    {"role":"user","content": f"Classify the sentiment (negative/neutral/positive) of this financial text:\n\n{text}"},
                    {"role":"assistant","content": lab_str}
                ]})
    return out

def load_reddit_tifu_api(base_dir: Path):
    out = []
    if not _HAS_HF_DATASETS:
        return out
    ensure_hf_cache(base_dir)
    # Prefer 'short' variant
    try:
        ds = load_dataset("reddit_tifu", "short")
    except Exception:
        try:
            ds = load_dataset("reddit_tifu")
        except Exception:
            return out
    for sp, d in ds.items():
        cols = set(d.column_names)
        title_key = "title" if "title" in cols else None
        text_key = None
        for cand in ("document","story","text","selftext"):
            if cand in cols:
                text_key = cand
                break
        if not title_key or not text_key:
            continue
        for rec in d:
            title = normalize_text(rec.get(title_key, ""))
            body = normalize_text(rec.get(text_key, ""))
            if title and body:
                out.append({"source": DATASET_NAMES["reddit_tifu"], "type":"chat", "messages":[
                    {"role":"user","content": f"Tell me a story about: {title}"},
                    {"role":"assistant","content": body}
                ]})
    return out

def load_dolly_api(base_dir: Path):
    out = []
    if not _HAS_HF_DATASETS:
        return out
    ensure_hf_cache(base_dir)
    try:
        ds = load_dataset("databricks/databricks-dolly-15k")
    except Exception:
        return out
    d = ds.get("train", ds)
    for rec in d:
        instr = normalize_text(rec.get("instruction",""))
        ctx = normalize_text(rec.get("context",""))
        resp = normalize_text(rec.get("response",""))
        if instr and resp:
            user = instr if not ctx else f"{instr}\n\n{ctx}"
            out.append({"source": "dolly_15k", "type":"chat", "messages":[
                {"role":"user","content": user},
                {"role":"assistant","content": resp}
            ]})
    return out

def load_alpaca_cleaned_api(base_dir: Path):
    out = []
    if not _HAS_HF_DATASETS:
        return out
    ensure_hf_cache(base_dir)
    try:
        ds = load_dataset("yahma/alpaca-cleaned")
    except Exception:
        return out
    d = ds.get("train", ds)
    for rec in d:
        instr = normalize_text(rec.get("instruction",""))
        inp = normalize_text(rec.get("input",""))
        outp = normalize_text(rec.get("output",""))
        if instr and outp:
            user = instr if not inp else f"{instr}\n\n{inp}"
            out.append({"source": "alpaca_cleaned", "type":"chat", "messages":[
                {"role":"user","content": user},
                {"role":"assistant","content": outp}
            ]})
    return out

def load_alpaca_gpt4_api(base_dir: Path):
    out = []
    if not _HAS_HF_DATASETS:
        return out
    ensure_hf_cache(base_dir)
    try:
        ds = load_dataset("vicgalle/alpaca-gpt4")
    except Exception:
        return out
    d = ds.get("train", ds)
    for rec in d:
        instr = normalize_text(rec.get("instruction",""))
        outp = normalize_text(rec.get("output",""))
        if instr and outp:
            out.append({"source": "alpaca_gpt4", "type":"chat", "messages":[
                {"role":"user","content": instr},
                {"role":"assistant","content": outp}
            ]})
    return out

def load_chatgpt_export(base_dir: Path):
    """
    ChatGPT export conversations
    """
    root = base_dir / "datasets" / "raw" / "chatgpt_export"
    if not root.exists():
        print(f"ChatGPT export directory not found at: {root}")
        return []
    
    print(f"Loading ChatGPT export from: {root}")
    out = []
    
    # Try conversations.json
    conversations_file = root / "conversations.json"
    if conversations_file.exists():
        print(f"  Found conversations.json: {conversations_file}")
        try:
            with conversations_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    for conv in data:
                        messages = conv.get("messages", [])
                        norm = []
                        for msg in messages:
                            role = msg.get("role", "").lower()
                            if role in ("user", "assistant"):
                                content = normalize_text(msg.get("content", ""))
                                if content:
                                    norm.append({"role": role, "content": content})
                        if len(norm) >= 2:
                            out.append({
                                "source": DATASET_NAMES["chatgpt_export"],
                                "type": "chat",
                                "messages": norm
                            })
        except Exception as e:
            print(f"    Error reading conversations.json: {e}")
    
    # Also try to parse chat.html if it exists
    chat_file = root / "chat.html"
    if chat_file.exists():
        print(f"  Found chat.html: {chat_file}")
        try:
            # Simple HTML parsing to extract conversations
            with chat_file.open("r", encoding="utf-8") as fh:
                content = fh.read()
                # Look for conversation patterns in HTML
                # This is a basic approach - could be improved with proper HTML parsing
                # Find user messages (look for patterns like "User:" or similar)
                user_pattern = r'<[^>]*>([^<]*[Uu]ser[^<]*)</[^>]*>'
                assistant_pattern = r'<[^>]*>([^<]*[Aa]ssistant[^<]*)</[^>]*>'
                
                user_matches = re.findall(user_pattern, content)
                assistant_matches = re.findall(assistant_pattern, content)
                
                # Try to pair them up
                for i in range(min(len(user_matches), len(assistant_matches))):
                    user_text = normalize_text(user_matches[i])
                    assistant_text = normalize_text(assistant_matches[i])
                    if user_text and assistant_text:
                        out.append({
                            "source": DATASET_NAMES["chatgpt_export"],
                            "type": "chat",
                            "messages": [
                                {"role": "user", "content": user_text},
                                {"role": "assistant", "content": assistant_text}
                            ]
                        })
        except Exception as e:
            print(f"    Error reading chat.html: {e}")
    
    print(f"ChatGPT Export: Loaded {len(out)} samples")
    return out

def load_wikipedia_xml(base_dir: Path):
    """
    Wikipedia dump - parse XML with bz2 decompression
    """
    # Check for both compressed and uncompressed versions
    wiki_bz2 = base_dir / "datasets" / "raw" / "enwiki-latest-pages-articles.xml.bz2"
    wiki_xml = base_dir / "datasets" / "raw" / "enwiki-latest-pages-articles.xml"
    
    if not wiki_bz2.exists() and not wiki_xml.exists():
        print("Wikipedia file not found!")
        return []
    
    out = []
    
    if wiki_xml.exists():
        print(f"Loading Wikipedia from uncompressed XML: {wiki_xml}")
        try:
            with open(wiki_xml, 'r', encoding='utf-8', errors='ignore') as fh:
                # Read in chunks to handle large files
                chunk_size = 1024 * 1024  # 1MB chunks
                buffer = ""
                
                for chunk in iter(lambda: fh.read(chunk_size), ""):
                    buffer += chunk
                    
                    # Look for complete <page> elements
                    while "</page>" in buffer:
                        end_pos = buffer.find("</page>")
                        page_xml = buffer[:end_pos + 7]
                        buffer = buffer[end_pos + 7:]
                        
                        try:
                            # Parse the page XML
                            root = ET.fromstring(page_xml)
                            
                            # Extract title and text
                            title_elem = root.find(".//title")
                            text_elem = root.find(".//text")
                            
                            if title_elem is not None and text_elem is not None:
                                title = normalize_text(title_elem.text or "")
                                text = normalize_text(text_elem.text or "")
                                
                                # Filter out non-article pages and very short texts
                                if (title and text and 
                                    not title.startswith("Wikipedia:") and
                                    not title.startswith("Template:") and
                                    not title.startswith("File:") and
                                    not title.startswith("Category:") and
                                    len(text) > 100):
                                    
                                    out.append({
                                        "source": DATASET_NAMES["wikipedia"],
                                        "type": "doc",
                                        "text": text,
                                        "meta": {"title": title, "format": "wikipedia_xml"}
                                    })
                                    
                                    # Limit to first 10000 articles to avoid memory issues
                                    if len(out) >= 10000:
                                        print(f"Reached limit of 10000 Wikipedia articles")
                                        break
                        
                        except ET.ParseError:
                            # Skip malformed XML
                            continue
                    
                    if len(out) >= 10000:
                        break
                        
        except Exception as e:
            print(f"Error parsing Wikipedia XML: {e}")
    
    elif wiki_bz2.exists():
        print(f"Wikipedia BZ2 file found but XML parsing not implemented for compressed files")
        print("Please extract the BZ2 file first or implement BZ2 decompression")
    
    print(f"Wikipedia: Loaded {len(out)} articles")
    return out

def load_gutenberg_rdf(base_dir: Path):
    """
    Gutenberg RDF - parse tar.bz2 with basic text extraction
    """
    gutenberg_bz2 = base_dir / "datasets" / "raw" / "gutenberg_rdf.tar.bz2"
    gutenberg_dir = base_dir / "datasets" / "raw" / "gutenberg_rdf"
    
    if not gutenberg_bz2.exists() and not gutenberg_dir.exists():
        print("Gutenberg files not found!")
        return []
    
    out = []
    
    if gutenberg_dir.exists():
        print(f"Loading Gutenberg from extracted directory: {gutenberg_dir}")
        try:
            # Look for text files in the extracted directory
            text_files = list(gutenberg_dir.rglob("*.txt"))
            print(f"  Found {len(text_files)} text files")
            
            for i, txt_file in enumerate(text_files):
                try:
                    with open(txt_file, 'r', encoding='utf-8', errors='ignore') as fh:
                        content = fh.read()
                        text = normalize_text(content)
                        
                        if text and len(text) > 100:
                            out.append({
                                "source": DATASET_NAMES["gutenberg"],
                                "type": "doc",
                                "text": text,
                                "meta": {"filename": str(txt_file), "format": "gutenberg_txt"}
                            })
                            
                            # Limit to first 1000 texts to avoid memory issues
                            if len(out) >= 1000:
                                print(f"Reached limit of 1000 Gutenberg texts")
                                break
                    
                    if i % 100 == 0:
                        print(f"  Processed {i} files, found {len(out)} samples...")
                        
                except Exception as e:
                    if i < 5:  # Only show first few errors
                        print(f"    Error reading {txt_file}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error processing Gutenberg directory: {e}")
    
    elif gutenberg_bz2.exists():
        print(f"Gutenberg BZ2 file found but RDF parsing not implemented for compressed files")
        print("Please extract the BZ2 file first or implement BZ2 decompression")
    
    print(f"Gutenberg: Loaded {len(out)} texts")
    return out

def load_hf_cache_datasets(base_dir: Path):
    """
    Load datasets from hf_cache/datasets directory
    """
    hf_dir = base_dir / "datasets" / "raw" / "hf_cache" / "datasets"
    if not hf_dir.exists():
        print(f"HF cache datasets directory not found at: {hf_dir}")
        return {}
    
    print(f"Loading datasets from HF cache: {hf_dir}")
    datasets = {}
    
    # UltraChat from HF cache
    ultrachat_hf = hf_dir / "HuggingFaceH4___ultrachat_200k" / "default" / "0.0.0"
    if ultrachat_hf.exists():
        print(f"  Found UltraChat HF cache: {ultrachat_hf}")
        arrow_files = list(ultrachat_hf.rglob("*.arrow"))
        if arrow_files and _HAS_PANDAS:
            ultrachat_samples = []
            for af in arrow_files:
                try:
                    data = pd.read_feather(af)
                    print(f"    Processing {af.name} with columns: {list(data.columns)}")
                    
                    # Look for conversation columns
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
                                    ultrachat_samples.append({
                                        "source": DATASET_NAMES["ultrachat"],
                                        "type": "chat",
                                        "messages": norm
                                    })
                    
                    elif "prompt" in data.columns and "response" in data.columns:
                        for _, row in data.iterrows():
                            prompt = normalize_text(row.get("prompt", ""))
                            response = normalize_text(row.get("response", ""))
                            if prompt and response:
                                ultrachat_samples.append({
                                    "source": DATASET_NAMES["ultrachat"],
                                    "type": "chat",
                                    "messages": [
                                        {"role": "user", "content": prompt},
                                        {"role": "assistant", "content": response}
                                    ]
                                })
                
                except Exception as e:
                    print(f"    Error reading {af.name}: {e}")
                    continue
            
            if ultrachat_samples:
                datasets["ultrachat_hf"] = ultrachat_samples
                print(f"    HF UltraChat: {len(ultrachat_samples)} samples")
    
    # OpenHermes from HF cache
    openhermes_hf = hf_dir / "rombodawg___open_hermes-2.5-uncensored" / "default" / "0.0.0"
    if openhermes_hf.exists():
        print(f"  Found OpenHermes HF cache: {openhermes_hf}")
        arrow_files = list(openhermes_hf.rglob("*.arrow"))
        if arrow_files and _HAS_PANDAS:
            openhermes_samples = []
            for af in arrow_files:
                try:
                    data = pd.read_feather(af)
                    print(f"    Processing {af.name} with columns: {list(data.columns)}")
                    
                    # Look for instruction/response columns
                    if "instruction" in data.columns and "response" in data.columns:
                        for _, row in data.iterrows():
                            instruction = normalize_text(row.get("instruction", ""))
                            response = normalize_text(row.get("response", ""))
                            if instruction and response:
                                openhermes_samples.append({
                                    "source": DATASET_NAMES["openorca"], # Assuming openorca is the target for OpenHermes
                                    "type": "chat",
                                    "messages": [
                                        {"role": "user", "content": instruction},
                                        {"role": "assistant", "content": response}
                                    ]
                                })
                
                except Exception as e:
                    print(f"    Error reading {af.name}: {e}")
                    continue
            
            if openhermes_samples:
                datasets["openhermes_hf"] = openhermes_samples
                print(f"    HF OpenHermes: {len(openhermes_samples)} samples")
    
    return datasets

# -----------------------------
# Chunking for 8K SFT
# -----------------------------
def split_chat_to_windows(messages, tok: TokenEstimator, max_tokens: int):
    """
    Produce one or more samples <= max_tokens using a sliding window across messages.
    Keep at least one user->assistant pair per chunk.
    """
    chunks = []
    n = len(messages)
    start = 0
    while start < n:
        end = start
        best_end = None
        while end < n:
            cand = messages[start:end+1]
            if tok.count_messages(cand) <= max_tokens:
                # ensure it ends on an assistant message for SFT target
                if cand[-1]["role"] == "assistant":
                    best_end = end
                end += 1
            else:
                break
        if best_end is None:
            # Force a minimal pair if a single assistant reply already exceeds (rare)
            # Try to include the smallest user+assistant around 'start..start+1'
            if start+1 < n and messages[start]["role"]=="user" and messages[start+1]["role"]=="assistant":
                pair = [messages[start], messages[start+1]]
                chunks.append(pair)
                start = start + 2
            else:
                # Skip a single oversize message
                chunks.append([messages[start]])
                start += 1
        else:
            chunks.append(messages[start:best_end+1])
            start = best_end + 1
    # Remove trivial chunks without both roles
    filtered = []
    for c in chunks:
        roles = set(m["role"] for m in c)
        if "assistant" in roles and "user" in roles:
            filtered.append(c)
    return filtered

def test_file_formats(base_dir: Path):
    """
    Test function to check the format of dataset files
    """
    print("\n[*] Testing file formats...")
    
    # Test UltraChat
    ultrachat_file = base_dir / "datasets" / "raw" / "ultrachat.jsonl"
    if ultrachat_file.exists():
        print(f"\nUltraChat file exists: {ultrachat_file}")
        print(f"File size: {ultrachat_file.stat().st_size / (1024*1024):.1f} MB")
        try:
            with ultrachat_file.open("r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= 3:  # Only check first 3 lines
                        break
                    try:
                        data = json.loads(line)
                        print(f"  Line {i+1}: Keys = {list(data.keys())}")
                        if "prompt" in data:
                            print(f"    Prompt preview: {str(data['prompt'])[:100]}...")
                        if "response" in data:
                            print(f"    Response preview: {str(data['response'])[:100]}...")
                    except Exception as e:
                        print(f"  Line {i+1}: Error parsing JSON: {e}")
        except Exception as e:
            print(f"  Error reading file: {e}")
    
    # Test Open-Orca
    openorca_file = base_dir / "datasets" / "raw" / "openorca.jsonl"
    if openorca_file.exists():
        print(f"\nOpen-Orca file exists: {openorca_file}")
        print(f"File size: {openorca_file.stat().st_size / (1024*1024):.1f} MB")
        try:
            with openorca_file.open("r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= 3:  # Only check first 3 lines
                        break
                    try:
                        data = json.loads(line)
                        print(f"  Line {i+1}: Keys = {list(data.keys())}")
                        if "instruction" in data:
                            print(f"    Instruction preview: {str(data['instruction'])[:100]}...")
                        if "response" in data:
                            print(f"    Response preview: {str(data['response'])[:100]}...")
                    except Exception as e:
                        print(f"  Line {i+1}: Error parsing JSON: {e}")
        except Exception as e:
            print(f"  Error reading file: {e}")
    
    # Test ChatGPT Export
    chatgpt_dir = base_dir / "datasets" / "raw" / "chatgpt_export"
    if chatgpt_dir.exists():
        print(f"\nChatGPT export directory exists: {chatgpt_dir}")
        for file in chatgpt_dir.iterdir():
            if file.is_file():
                print(f"  File: {file.name} ({file.stat().st_size / 1024:.1f} KB)")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=str(Path(__file__).resolve().parents[2]),
                        help="Project root (aria_ai_assistant)")
    parser.add_argument("--f-drive", type=str, default="F:\\aria_ai\\aria_ai_assistant",
                        help="F drive project path")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        help="HF tokenizer name or local path (default: gpt2). If unavailable, uses heuristic.")
    parser.add_argument("--context", type=int, default=8192, help="Context tokens for SFT windows (default 8192)")
    parser.add_argument("--val-ratio", type=float, default=0.01, help="Validation split ratio")
    parser.add_argument("--test-formats", action="store_true", help="Test file formats before processing")
    args = parser.parse_args()

    random.seed(RNG_SEED)

    base_dir = Path(args.base_dir)
    f_drive = Path(args.f_drive)
    organized_dir = base_dir / "datasets" / "organized"
    missing_dir = base_dir / "datasets" / "missing"
    ensure_dir(missing_dir)

    # Test file formats if requested
    if args.test_formats:
        test_file_formats(base_dir)
        return

    print("[*] Processing ONLY missing datasets...")
    
    # Load missing datasets
    ultrachat = load_ultrachat_raw(base_dir)
    openorca = load_openorca_raw(base_dir)
    empathetic_dialogues = load_empathetic_dialogues(base_dir)
    financial_phrasebank = load_financial_phrasebank(base_dir)
    reddit_tifu = load_reddit_tifu(base_dir)
    chatgpt_export = load_chatgpt_export(base_dir)
    wikipedia = load_wikipedia_xml(base_dir)
    gutenberg = load_gutenberg_rdf(base_dir)

    # API fallbacks (download+load)
    print("\n[*] Loading via HF datasets API (fallbacks/extras)...")
    try:
        ultra_api = load_ultrachat_api(base_dir)
        if ultra_api:
            ultrachat.extend(ultra_api)
            print(f"    UltraChat API: +{len(ultra_api)}")
    except Exception:
        pass
    try:
        ed_api = load_empathetic_dialogues_api(base_dir)
        if ed_api:
            empathetic_dialogues.extend(ed_api)
            print(f"    Empathetic Dialogues API: +{len(ed_api)}")
    except Exception:
        pass
    try:
        fpb_api = load_financial_phrasebank_api(base_dir)
        if fpb_api:
            financial_phrasebank.extend(fpb_api)
            print(f"    Financial Phrasebank API: +{len(fpb_api)}")
    except Exception:
        pass
    try:
        tifu_api = load_reddit_tifu_api(base_dir)
        if tifu_api:
            reddit_tifu.extend(tifu_api)
            print(f"    Reddit TIFU API: +{len(tifu_api)}")
    except Exception:
        pass
    # Extras
    extras_chat = []
    try:
        extras_chat.extend(load_dolly_api(base_dir))
    except Exception:
        pass
    try:
        extras_chat.extend(load_alpaca_cleaned_api(base_dir))
    except Exception:
        pass
    try:
        extras_chat.extend(load_alpaca_gpt4_api(base_dir))
    except Exception:
        pass
    
    # Also try loading from HF cache
    print("\n[*] Loading from HF cache...")
    hf_datasets = load_hf_cache_datasets(base_dir)
    
    # Combine HF cache datasets with raw datasets
    if "ultrachat_hf" in hf_datasets:
        ultrachat.extend(hf_datasets["ultrachat_hf"])
        print(f"Combined UltraChat: {len(ultrachat)} total samples")
    
    if "openhermes_hf" in hf_datasets:
        openorca.extend(hf_datasets["openhermes_hf"])
        print(f"Combined Open-Orca: {len(openorca)} total samples")

    # Report what we found
    present = {
        DATASET_NAMES["ultrachat"]: len(ultrachat),
        DATASET_NAMES["openorca"]: len(openorca),
        DATASET_NAMES["empathetic_dialogues"]: len(empathetic_dialogues),
        DATASET_NAMES["financial_phrasebank"]: len(financial_phrasebank),
        DATASET_NAMES["reddit_tifu"]: len(reddit_tifu),
        DATASET_NAMES["chatgpt_export"]: len(chatgpt_export),
        DATASET_NAMES["wikipedia"]: len(wikipedia),
        DATASET_NAMES["gutenberg"]: len(gutenberg),
    }
    
    print("\n[*] Missing datasets found:")
    for k, v in present.items():
        print(f"    - {k}: {v} samples")

    # Write missing dataset files
    print("\n[*] Writing missing dataset files...")
    per_files = {}

    def _w(name, rows):
        p = missing_dir / f"{name}.jsonl"
        write_jsonl(p, rows)
        per_files[name] = str(p)
        return p

    if ultrachat: _w(DATASET_NAMES["ultrachat"], ultrachat)
    if openorca: _w(DATASET_NAMES["openorca"], openorca)
    if empathetic_dialogues: _w(DATASET_NAMES["empathetic_dialogues"], empathetic_dialogues)
    if financial_phrasebank: _w(DATASET_NAMES["financial_phrasebank"], financial_phrasebank)
    if reddit_tifu: _w(DATASET_NAMES["reddit_tifu"], reddit_tifu)
    if chatgpt_export: _w(DATASET_NAMES["chatgpt_export"], chatgpt_export)
    if wikipedia: _w(DATASET_NAMES["wikipedia"], wikipedia)
    if gutenberg: _w(DASET_NAMES["gutenberg"], gutenberg)

    # Now merge with existing organized data
    print("\n[*] Merging with existing organized data...")
    
    # Load existing organized data
    existing_chat = []
    existing_docs = []
    
    # Load existing chat datasets
    existing_files = [
        "alterego_uncensored_personal.jsonl",
        "openhermes_uncensored_general.jsonl", 
        "zeroshot_financial_filtered_financial.jsonl",
        "financial_news_articles.jsonl"
    ]
    
    for filename in existing_files:
        filepath = organized_dir / filename
        if filepath.exists():
            try:
                with filepath.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            row = json.loads(line)
                            if row.get("type") == "chat":
                                existing_chat.append(row)
                        except Exception:
                            continue
            except Exception:
                continue
    
    # Load missing chat datasets
    missing_chat = []
    for dataset_name in [ultrachat, openorca, empathetic_dialogues, financial_phrasebank, reddit_tifu, chatgpt_export]:
        missing_chat.extend(dataset_name)
    
    # Combine all chat data (include extras)
    all_chat = existing_chat + missing_chat + extras_chat
    random.shuffle(all_chat)
    
    # Load existing docs
    doc_files = [
        "fineweb_filtered_general.docstore.jsonl",
        "json_misc_datasets.jsonl"
    ]
    
    for filename in doc_files:
        filepath = organized_dir / filename
        if filepath.exists():
            try:
                with filepath.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            row = json.loads(line)
                            if row.get("type") == "doc":
                                existing_docs.append(row)
                        except Exception:
                            continue
            except Exception:
                continue
    
    # Add missing docs
    missing_docs = wikipedia + gutenberg
    all_docs = existing_docs + missing_docs
    
    # Write final complete datasets
    print(f"\n[*] Writing final complete datasets...")
    
    # Write all chat data
    final_chat_path = organized_dir / "complete_chat_datasets.jsonl"
    write_jsonl(final_chat_path, all_chat)
    
    # Write all doc data
    final_docs_path = organized_dir / "complete_document_datasets.jsonl"
    write_jsonl(final_docs_path, all_docs)
    
    # Create final SFT dataset
    print(f"[*] Creating final SFT dataset...")
    tok = TokenEstimator(args.tokenizer, args.context)
    
    final_chunks = []
    for ex in tqdm(all_chat, desc="chunking final"):
        chunks = split_chat_to_windows(ex["messages"], tok, args.context)
        for c in chunks:
            final_chunks.append({
                "source": ex["source"],
                "type": "chat",
                "messages": c
            })
    
    # Train/Val split
    n = len(final_chunks)
    val_n = max(1, int(n * args.val_ratio))
    random.shuffle(final_chunks)
    val = final_chunks[:val_n]
    train = final_chunks[val_n:]
    
    # Write final SFT files
    final_sft_path = organized_dir / "complete_Zia_sft_8k.jsonl"
    write_jsonl(final_sft_path, final_chunks)
    
    final_train_path = organized_dir / "complete_train.jsonl"
    final_val_path = organized_dir / "complete_val.jsonl"
    write_jsonl(final_train_path, train)
    write_jsonl(final_val_path, val)
    
    # Final stats
    final_stats = {
        "existing_chat_samples": len(existing_chat),
        "missing_chat_samples": len(missing_chat),
        "total_chat_samples": len(all_chat),
        "existing_doc_samples": len(existing_docs),
        "missing_doc_samples": len(missing_docs),
        "total_doc_samples": len(all_docs),
        "final_sft_samples": len(final_chunks),
        "final_train": len(train),
        "final_val": len(val),
        "missing_datasets_found": present,
        "tokenizer_used": args.tokenizer if tok.ok else "heuristic_approx_4chars_per_token",
        "context_tokens": args.context
    }
    
    (organized_dir / "complete_stats.json").write_text(json.dumps(final_stats, indent=2), encoding="utf-8")
    
    print("\n✅ Done processing missing datasets!")
    print(f"  Missing dataset files: {missing_dir}")
    print(f"  Final complete chat: {final_chat_path}")
    print(f"  Final complete docs: {final_docs_path}")
    print(f"  Final complete SFT: {final_sft_path}")
    print(f"  Final train/val: {final_train_path} | {final_val_path}")
    print(f"  Final stats: {organized_dir / 'complete_stats.json'}")

if __name__ == "__main__":
    main() 