# -*- coding: utf-8 -*-
"""
unify_and_prepare.py

One-pass organizer & cleaner for Zia:
- Scans your raw dataset cache on disk (both D: and F: drives)
- Loads supported sources (AlterEgo export, OpenHermes 2.5 Uncensored, Ultrachat 200k, Zeroshot Twitter Financial, FineWeb-Edu, Open-Orca, Empathetic Dialogues, Financial Phrasebank, Reddit TIFU, ChatGPT Export, Wikipedia, Gutenberg, and more)
- Normalizes to a single schema (chat/doc)
- Cleans + deduplicates across all sources
- Produces per-dataset JSONL with clear names
- Merges chat datasets into SFT-ready merged_Zia_sft_8k.jsonl
- Splits train/val
- Emits stats and duplicate reports

Run:
  python unify_and_prepare.py --base-dir "D:\\aria\\aria_ai\\aria_ai_assistant" --f-drive "F:\\aria_ai\\aria_ai_assistant" --tokenizer gpt2 --context 8192
"""

import os, re, json, csv, sys, gzip, argparse, random, hashlib, math
from pathlib import Path

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

# -----------------------------
# Config
# -----------------------------
DATASET_NAMES = {
    "alterego": "alterego_uncensored_personal",
    "openhermes": "openhermes_uncensored_general",
    "ultrachat": "ultrachat_filtered_dialogue",
    "zeroshot_financial": "zeroshot_financial_filtered_financial",
    "fineweb": "fineweb_filtered_general",
    "openorca": "openorca_instruction_following",
    "empathetic_dialogues": "empathetic_dialogues_emotional",
    "financial_phrasebank": "financial_phrasebank_sentiment",
    "reddit_tifu": "reddit_tifu_storytelling",
    "chatgpt_export": "chatgpt_export_personal",
    "wikipedia": "wikipedia_encyclopedia",
    "gutenberg": "gutenberg_literature",
    "financial_news": "financial_news_articles",
    "json_datasets": "json_misc_datasets",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are Zia, a helpful, capable assistant from Aria. "
    "Be candid but polite, fast, and accurate. "
    "Prefer concise, practical answers with clear steps. "
    "When tools or code are relevant, provide runnable examples. "
    "Avoid gratuitous refusals; if safety applies, explain briefly and offer a safe alternative."
)

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
# Loaders (robust to local cache)
# -----------------------------
def load_alterego(base_dir: Path):
    # Expect: datasets/raw/alterego/alterego.jsonl (created by your earlier parser)
    root = base_dir / "datasets" / "raw" / "alterego"
    f = root / "alterego.jsonl"
    if not f.exists():
        return []
    out = []
    
    # First try to load as conversation format
    try:
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    msgs = row.get("messages") or []
                    norm = []
                    for m in msgs:
                        role = (m.get("role") or "").lower()
                        if role not in ("user","assistant","system"):
                            continue
                        content = normalize_text(m.get("content",""))
                        if content:
                            norm.append({"role": role, "content": content})
                    if len(norm) >= 2:
                        out.append({
                            "source": DATASET_NAMES["alterego"],
                            "type": "chat",
                            "messages": norm
                        })
                except Exception:
                    continue
    except Exception:
        pass
    
    # If no conversations found, try to load as individual messages and group them
    if not out:
        try:
            messages = []
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        msg = json.loads(line)
                        role = (msg.get("role") or "").lower()
                        if role in ("user", "assistant"):
                            content = normalize_text(msg.get("content", ""))
                            if content:
                                messages.append({"role": role, "content": content})
                    except Exception:
                        continue
            
            # Group messages into conversations (user -> assistant pairs)
            i = 0
            while i < len(messages) - 1:
                if (messages[i]["role"] == "user" and 
                    messages[i+1]["role"] == "assistant"):
                    out.append({
                        "source": DATASET_NAMES["alterego"],
                        "type": "chat",
                        "messages": [messages[i], messages[i+1]]
                    })
                    i += 2
                else:
                    i += 1
        except Exception:
            pass
    
    return out

def load_openhermes_uncensored(base_dir: Path):
    """
    Load OpenHermes from both raw JSONL files and HuggingFace cache
    """
    out = []
    
    # Try loading from raw JSONL file first (if it exists)
    raw_file = base_dir / "datasets" / "raw" / "openhermes.jsonl"
    if raw_file.exists():
        try:
            with raw_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                        out.extend(_normalize_openhermes_example(row))
                    except Exception:
                        continue
        except Exception:
            pass
    
    # Also try loading from HuggingFace cache
    base = base_dir / "datasets" / "raw" / "rombodawg___open_hermes-2.5-uncensored" / "default" / "0.0.0"
    if base.exists():
        json_files = list(base.rglob("hermes-2.5-alpaca-uncensored.json"))
        if json_files:
            path = json_files[0]
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            ex = json.loads(line)
                        except Exception:
                            # Some dumps are JSON array; handle both
                            try:
                                data = json.load(open(path, "r", encoding="utf-8"))
                                if isinstance(data, list):
                                    for ex2 in data:
                                        out.extend(_normalize_openhermes_example(ex2))
                                break
                            except Exception:
                                break
                        out.extend(_normalize_openhermes_example(ex))
            except Exception:
                pass
    
    return out

def _normalize_openhermes_example(ex):
    out = []
    # Typical fields: instruction / input / output   (alpaca format)
    # Some variants: "prompt", "response"
    instruction = normalize_text(ex.get("instruction", ex.get("prompt","")))
    input_txt   = normalize_text(ex.get("input",""))
    output      = normalize_text(ex.get("output", ex.get("response","")))
    if not instruction and not input_txt:
        return out
    user = instruction if not input_txt else f"{instruction}\n\n{input_txt}"
    if not output:
        return out
    out.append({
        "source": DATASET_NAMES["openhermes"],
        "type": "chat",
        "messages": [
            {"role":"user","content": user},
            {"role":"assistant","content": output}
        ]
    })
    return out

def load_ultrachat(base_dir: Path):
    """
    H4/ultrachat_200k – local snapshot has parquet files (train_sft/test_sft etc.)
    We try parquet via pandas/pyarrow; fallback: skip if not available.
    """
    out = []
    
    # Try loading from raw JSONL file first (if it exists)
    raw_file = base_dir / "datasets" / "raw" / "ultrachat.jsonl"
    if raw_file.exists():
        try:
            with raw_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                        # Try to extract prompt/response format
                        prompt = normalize_text(row.get("prompt", row.get("input", "")))
                        response = normalize_text(row.get("response", row.get("output", "")))
                        if prompt and response:
                            out.append({
                                "source": DATASET_NAMES["ultrachat"],
                                "type": "chat",
                                "messages": [
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": response}
                                ]
                            })
                    except Exception:
                        continue
        except Exception:
            pass
    
    # Also try loading from HuggingFace cache
    possible_roots = [
        base_dir / "datasets" / "raw" / "hf_cache" / "hub" / "datasets--HuggingFaceH4--ultrachat_200k" / "snapshots",
        base_dir / "datasets" / "raw" / "hf_cache" / "datasets" / "HuggingFaceH4___ultrachat_200k" / "default" / "0.0.0"
    ]
    
    for root in possible_roots:
        if not root.exists():
            continue
            
        # find *.parquet files
        parquet_files = list(root.rglob("*.parquet"))
        if not parquet_files or not _HAS_PANDAS:
            continue
            
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf, engine="pyarrow")
            except Exception:
                continue
            # Possibilities:
            # - 'messages' column (list of dicts role/content)
            # - 'prompt' / 'response'
            if "messages" in df.columns:
                for msgs in df["messages"]:
                    norm = []
                    if isinstance(msgs, list):
                        for m in msgs:
                            role = (m.get("role") or "").lower()
                            if role not in ("user","assistant","system"):
                                continue
                            content = normalize_text(m.get("content",""))
                            if content:
                                norm.append({"role": role, "content": content})
                    if len(norm) >= 2:
                        out.append({"source": DATASET_NAMES["ultrachat"], "type":"chat", "messages": norm})
            elif "prompt" in df.columns and "response" in df.columns:
                for _, row in df.iterrows():
                    u = normalize_text(row["prompt"])
                    a = normalize_text(row["response"])
                    if u and a:
                        out.append({"source": DATASET_NAMES["ultrachat"], "type":"chat",
                                    "messages":[{"role":"user","content":u},{"role":"assistant","content":a}]})
            elif "input" in df.columns and "output" in df.columns:
                for _, row in df.iterrows():
                    u = normalize_text(row["input"])
                    a = normalize_text(row["output"])
                    if u and a:
                        out.append({"source": DATASET_NAMES["ultrachat"], "type":"chat",
                                    "messages":[{"role":"user","content":u},{"role":"assistant","content":a}]})
    
    return out

def load_zeroshot_financial(base_dir: Path):
    """
    zeroshot/twitter-financial-news-sentiment – CSVs (sent_train.csv, sent_valid.csv)
    We convert to a simple classification chat format.
    """
    root = base_dir / "datasets" / "raw" / "hf_cache" / "hub" / "datasets--zeroshot--twitter-financial-news-sentiment" / "snapshots"
    if not root.exists():
        return []
    csvs = list(root.rglob("*.csv"))
    label_map = {"0":"negative","1":"neutral","2":"positive"}
    out = []
    for cf in csvs:
        try:
            with cf.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                # Fallback if no header: try simple csv
                if reader.fieldnames is None:
                    fh.seek(0)
                    reader = csv.reader(fh)
                    for row in reader:
                        if len(row) < 2:
                            continue
                        text, label = row[0], row[1]
                        label_str = label_map.get(str(label).strip(), str(label).strip())
                        u = f"Classify the sentiment (negative/neutral/positive) of this financial tweet:\n\n{normalize_text(text)}"
                        a = f"{label_str}"
                        out.append({"source": DATASET_NAMES["zeroshot_financial"], "type":"chat",
                                    "messages":[{"role":"user","content":u},{"role":"assistant","content":a}]})
                else:
                    # Expect columns like: text,label  (names vary, so attempt to detect)
                    fields = [f.lower() for f in reader.fieldnames]
                    # detect likely columns
                    text_key = next((k for k in reader.fieldnames if k.lower() in ("text","tweet","sentence","content")), reader.fieldnames[0])
                    label_key = next((k for k in reader.fieldnames if k.lower() in ("label","sentiment","target")), reader.fieldnames[-1])
                    for row in reader:
                        text = normalize_text(row.get(text_key,""))
                        lab  = str(row.get(label_key,"")).strip()
                        if not text:
                            continue
                        label_str = label_map.get(lab, lab)
                        u = f"Classify the sentiment (negative/neutral/positive) of this financial tweet:\n\n{text}"
                        a = f"{label_str}"
                        out.append({"source": DATASET_NAMES["zeroshot_financial"], "type":"chat",
                                    "messages":[{"role":"user","content":u},{"role":"assistant","content":a}]})
        except Exception:
            continue
    return out

def load_fineweb_docs(base_dir: Path):
    """
    FineWeb-Edu parquet shards -> doc corpus (not SFT). We keep as docstore for retrieval later.
    Columns vary; we try 'text' first, then 'content' or 'document'.
    """
    root = base_dir / "datasets" / "raw" / "hf_cache" / "hub" / "datasets--HuggingFaceFW--fineweb-edu" / "snapshots"
    if not root.exists():
        return []
    if not _HAS_PARQUET:
        return []
    parquet_files = list(root.rglob("*.parquet"))
    out = []
    for pf in parquet_files:
        try:
            table = pq.read_table(pf)
            cols = table.column_names
            key = "text"
            for cand in ["text","content","document","raw_content"]:
                if cand in cols:
                    key = cand
                    break
            col = table[key].to_pylist()
            # meta info from path (e.g., CC-MAIN-2014-10 shard and file name)
            shard = pf.parent.name  # e.g., CC-MAIN-2014-10
            for txt in col:
                txtn = normalize_text(txt)
                if not txtn:
                    continue
                out.append({
                    "source": DATASET_NAMES["fineweb"],
                    "type": "doc",
                    "text": txtn,
                    "meta": {"shard": shard}
                })
        except Exception:
            continue
    return out

def load_openorca(base_dir: Path):
    """
    Open-Orca dataset from HuggingFace cache and raw JSONL files
    """
    out = []
    
    # Try loading from raw JSONL file first (if it exists)
    raw_file = base_dir / "datasets" / "raw" / "openorca.jsonl"
    if raw_file.exists():
        try:
            with raw_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                        # Try to extract instruction/response format
                        instruction = normalize_text(row.get("instruction", row.get("prompt", "")))
                        response = normalize_text(row.get("response", row.get("output", "")))
                        if instruction and response:
                            out.append({
                                "source": DATASET_NAMES["openorca"],
                                "type": "chat",
                                "messages": [
                                    {"role": "user", "content": instruction},
                                    {"role": "assistant", "content": response}
                                ]
                            })
                    except Exception:
                        continue
        except Exception:
            pass
    
    # Also try loading from HuggingFace cache
    root = base_dir / "datasets" / "raw" / "Open-Orca___open_orca" / "default" / "0.0.0"
    if root.exists():
        arrow_files = list(root.rglob("*.arrow"))
        if arrow_files and _HAS_PANDAS:
            for af in arrow_files:
                try:
                    # Read as arrow/feather file
                    if af.suffix == '.arrow':
                        df = pd.read_feather(af)
                    else:
                        continue
                except Exception as e:
                    print(f"Error reading {af}: {e}")
                    continue
                
                # Look for instruction/response columns
                if "instruction" in df.columns and "response" in df.columns:
                    for _, row in df.iterrows():
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
                elif "prompt" in df.columns and "completion" in df.columns:
                    for _, row in df.iterrows():
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
                elif "input" in df.columns and "output" in df.columns:
                    for _, row in df.iterrows():
                        input_text = normalize_text(row.get("input", ""))
                        output = normalize_text(row.get("output", ""))
                        if input_text and output:
                            out.append({
                                "source": DATASET_NAMES["openorca"],
                                "type": "chat",
                                "messages": [
                                    {"role": "user", "content": input_text},
                                    {"role": "assistant", "content": output}
                                ]
                            })
    
    return out

def load_empathetic_dialogues(base_dir: Path):
    """
    Empathetic Dialogues dataset
    """
    root = base_dir / "datasets" / "raw" / "hf_cache" / "hub" / "datasets--empathetic_dialogues" / "snapshots"
    if not root.exists():
        return []
    
    out = []
    
    # Look for data files in snapshots
    for snapshot_dir in root.iterdir():
        if snapshot_dir.is_dir():
            # Look for arrow files or other data formats
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

def load_chatgpt_export(base_dir: Path):
    """
    ChatGPT export conversations
    """
    root = base_dir / "datasets" / "raw" / "chatgpt_export"
    if not root.exists():
        return []
    
    out = []
    
    # Try conversations.json
    conversations_file = root / "conversations.json"
    if conversations_file.exists():
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
        except Exception:
            pass
    
    # Also try to parse chat.html if it exists
    chat_file = root / "chat.html"
    if chat_file.exists():
        try:
            # Simple HTML parsing to extract conversations
            with chat_file.open("r", encoding="utf-8") as fh:
                content = fh.read()
                # Look for conversation patterns in HTML
                # This is a basic approach - could be improved with proper HTML parsing
                import re
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
        except Exception:
            pass
    
    return out

def load_wikipedia(base_dir: Path):
    """
    Wikipedia dump - convert to document format
    """
    wiki_file = base_dir / "datasets" / "raw" / "enwiki-latest-pages-articles.xml.bz2"
    if not wiki_file.exists():
        return []
    
    # For now, return empty as this requires special XML parsing
    # TODO: Implement Wikipedia XML parser with bz2 decompression
    # This would require xml.etree.ElementTree and bz2 modules
    print(f"Note: Wikipedia file found at {wiki_file} but XML parsing not yet implemented")
    return []

def load_gutenberg(base_dir: Path):
    """
    Gutenberg RDF - convert to document format
    """
    gutenberg_file = base_dir / "datasets" / "raw" / "gutenberg_rdf.tar.bz2"
    if not gutenberg_file.exists():
        return []
    
    # For now, return empty as this requires special tar/RDF parsing
    # TODO: Implement Gutenberg RDF parser with tar.bz2 decompression
    # This would require tarfile and rdflib modules
    print(f"Note: Gutenberg file found at {gutenberg_file} but RDF parsing not yet implemented")
    return []

def load_financial_news(f_drive: Path):
    """
    Financial news from F drive
    """
    news_file = f_drive / "datasets" / "raw" / "fin_news.jsonl"
    if not news_file.exists():
        return []
    
    out = []
    try:
        with news_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    # Extract text content - adjust based on actual format
                    text = normalize_text(row.get("text", row.get("content", "")))
                    if text:
                        out.append({
                            "source": DATASET_NAMES["financial_news"],
                            "type": "doc",
                            "text": text,
                            "meta": {"format": "financial_news"}
                        })
                except Exception:
                    continue
    except Exception:
        pass
    
    return out

def load_openhermes_f_drive(f_drive: Path):
    """
    OpenHermes from F drive
    """
    hermes_file = f_drive / "datasets" / "raw" / "openhermes.jsonl"
    if not hermes_file.exists():
        return []
    
    out = []
    try:
        with hermes_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    # Try to extract instruction/response format
                    instruction = normalize_text(row.get("instruction", row.get("prompt", "")))
                    response = normalize_text(row.get("response", row.get("output", "")))
                    if instruction and response:
                        out.append({
                            "source": DATASET_NAMES["openhermes"],
                            "type": "chat",
                            "messages": [
                                {"role": "user", "content": instruction},
                                {"role": "assistant", "content": response}
                            ]
                        })
                except Exception:
                    continue
    except Exception:
        pass
    
    return out

def load_fineweb_f_drive(f_drive: Path):
    """
    FineWeb from F drive
    """
    fineweb_file = f_drive / "datasets" / "raw" / "fineweb.jsonl"
    if not fineweb_file.exists():
        return []
    
    out = []
    try:
        with fineweb_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    text = normalize_text(row.get("text", row.get("content", "")))
                    if text:
                        out.append({
                            "source": DATASET_NAMES["fineweb"],
                            "type": "doc",
                            "text": text,
                            "meta": {"format": "fineweb_f_drive"}
                        })
                except Exception:
                    continue
    except Exception:
        pass
    
    return out

def load_json_datasets(base_dir: Path):
    """
    Load various JSON format datasets from the json directory and other JSON files
    """
    out = []
    
    # Load from json directory
    json_dir = base_dir / "datasets" / "raw" / "json"
    if json_dir.exists():
        # Look for arrow files that might contain JSON-like data
        arrow_files = list(json_dir.rglob("*.arrow"))
        
        for af in arrow_files:
            try:
                if af.suffix == '.arrow':
                    data = pd.read_feather(af)
                else:
                    continue
            except Exception:
                continue
            
            # Try to find text-like columns
            text_columns = [col for col in data.columns if col.lower() in ["text", "content", "document", "story"]]
            
            for col in text_columns:
                for _, row in data.iterrows():
                    text = normalize_text(row.get(col, ""))
                    if text:
                        out.append({
                            "source": DATASET_NAMES["json_datasets"],
                            "type": "doc",
                            "text": text,
                            "meta": {"source_file": str(af), "column": col}
                        })
    
    # Load from merged.jsonl if it exists
    merged_file = base_dir / "datasets" / "raw" / "merged.jsonl"
    if merged_file.exists():
        try:
            with merged_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                        # Try to extract text content
                        text = normalize_text(row.get("text", row.get("content", "")))
                        if text:
                            out.append({
                                "source": DATASET_NAMES["json_datasets"],
                                "type": "doc",
                                "text": text,
                                "meta": {"source_file": "merged.jsonl"}
                            })
                    except Exception:
                        continue
        except Exception:
            pass
    
    return out

def load_all_datasets(base_dir: Path, f_drive: Path):
    """
    Load all datasets from both D and F drives, avoiding duplicates
    """
    print("[*] Loading datasets from D drive...")
    
    # D Drive datasets
    datasets = {}
    
    # 1. UltraChat - Load from both raw JSONL and HF cache
    print("  Loading UltraChat...")
    ultrachat_raw = load_ultrachat_raw(base_dir)
    ultrachat_hf = load_ultrachat_hf_cache(base_dir)
    datasets["ultrachat"] = ultrachat_raw + ultrachat_hf
    print(f"    UltraChat: {len(ultrachat_raw)} raw + {len(ultrachat_hf)} HF = {len(datasets['ultrachat'])} total")
    
    # 2. Open-Orca
    print("  Loading Open-Orca...")
    datasets["openorca"] = load_openorca_raw(base_dir)
    print(f"    Open-Orca: {len(datasets['openorca'])} samples")
    
    # 3. OpenHermes - Load from D drive HF cache (avoid duplicates)
    print("  Loading OpenHermes from D drive...")
    datasets["openhermes"] = load_openhermes_d_drive(base_dir)
    print(f"    OpenHermes D: {len(datasets['openhermes'])} samples")
    
    # 4. FineWeb - Load from D drive HF cache (avoid duplicates)
    print("  Loading FineWeb from D drive...")
    datasets["fineweb"] = load_fineweb_d_drive(base_dir)
    print(f"    FineWeb D: {len(datasets['fineweb'])} samples")
    
    # 5. Gutenberg
    print("  Loading Gutenberg...")
    datasets["gutenberg"] = load_gutenberg_extracted(base_dir)
    print(f"    Gutenberg: {len(datasets['gutenberg'])} samples")
    
    # 6. Wikipedia
    print("  Loading Wikipedia...")
    datasets["wikipedia"] = load_wikipedia_extracted(base_dir)
    print(f"    Wikipedia: {len(datasets['wikipedia'])} samples")
    
    # 7. Other datasets
    datasets["alterego"] = load_alterego(base_dir)
    datasets["chatgpt_export"] = load_chatgpt_export(base_dir)
    datasets["json_datasets"] = load_json_datasets(base_dir)
    datasets["empathetic_dialogues"] = load_empathetic_dialogues(base_dir)
    datasets["financial_phrasebank"] = load_financial_phrasebank(base_dir)
    datasets["reddit_tifu"] = load_reddit_tifu(base_dir)
    datasets["zeroshot_financial"] = load_zeroshot_financial(base_dir)
    
    print("\n[*] Loading datasets from F drive...")
    
    # F Drive datasets (processed versions)
    print("  Loading F drive datasets...")
    datasets["openhermes_f"] = load_openhermes_f_drive(f_drive)
    datasets["fineweb_f"] = load_fineweb_f_drive(f_drive)
    datasets["financial_news"] = load_financial_news(f_drive)
    
    print(f"    OpenHermes F: {len(datasets['openhermes_f'])} samples")
    print(f"    FineWeb F: {len(datasets['fineweb_f'])} samples")
    print(f"    Financial News: {len(datasets['financial_news'])} samples")
    
    return datasets

def load_ultrachat_raw(base_dir: Path):
    """Load UltraChat from raw JSONL file"""
    raw_file = base_dir / "datasets" / "raw" / "ultrachat.jsonl"
    if not raw_file.exists():
        return []
    
    out = []
    try:
        with raw_file.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                try:
                    row = json.loads(line)
                    if "text" in row:
                        text_content = row["text"]
                        if text_content.startswith("<EGO> "):
                            # Use regex to extract prompt and response
                            import re
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
                    
                    if i % 50000 == 0:
                        print(f"    Processed {i} lines, found {len(out)} samples...")
                        
                except Exception:
                    continue
    except Exception as e:
        print(f"    Error reading UltraChat file: {e}")
    
    return out

def load_ultrachat_hf_cache(base_dir: Path):
    """Load UltraChat from HF cache arrow files"""
    hf_dir = base_dir / "datasets" / "raw" / "hf_cache" / "datasets" / "HuggingFaceH4___ultrachat_200k" / "default" / "0.0.0"
    if not hf_dir.exists() or not _HAS_PANDAS:
        return []
    
    out = []
    arrow_files = list(hf_dir.rglob("*.arrow"))
    
    for af in arrow_files:
        try:
            data = pd.read_feather(af)
            print(f"    Processing {af.name} with columns: {list(data.columns)}")
            
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
            print(f"    Error reading {af.name}: {e}")
            continue
    
    return out

def load_openorca_raw(base_dir: Path):
    """Load Open-Orca from raw JSONL file"""
    raw_file = base_dir / "datasets" / "raw" / "openorca.jsonl"
    if not raw_file.exists():
        return []
    
    out = []
    try:
        with raw_file.open("r", encoding="utf-8") as fh:
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
                    
                    if i % 50000 == 0:
                        print(f"    Processed {i} lines, found {len(out)} samples...")
                        
                except Exception:
                    continue
    except Exception as e:
        print(f"    Error reading Open-Orca file: {e}")
    
    return out

def load_openhermes_d_drive(base_dir: Path):
    """Load OpenHermes from D drive HF cache (avoid duplicates)"""
    # Use the larger dataset
    hf_dir = base_dir / "datasets" / "raw" / "datasets--rombodawg--OpenHermes-2.5-Uncensored" / "snapshots"
    if not hf_dir.exists():
        return []
    
    out = []
    for snapshot_dir in hf_dir.iterdir():
        if snapshot_dir.is_dir():
            json_files = list(snapshot_dir.rglob("*.json"))
            if json_files:
                path = json_files[0]
                try:
                    with path.open("r", encoding="utf-8") as fh:
                        for line in fh:
                            try:
                                ex = json.loads(line)
                                out.extend(_normalize_openhermes_example(ex))
                            except Exception:
                                continue
                except Exception:
                    pass
                break
    
    return out

def load_fineweb_d_drive(base_dir: Path):
    """Load FineWeb from D drive HF cache (avoid duplicates)"""
    hf_dir = base_dir / "datasets" / "raw" / "datasets--HuggingFaceFW--fineweb-edu" / "snapshots"
    if not hf_dir.exists() or not _HAS_PARQUET:
        return []
    
    out = []
    for snapshot_dir in hf_dir.iterdir():
        if snapshot_dir.is_dir():
            parquet_files = list(snapshot_dir.rglob("*.parquet"))
            for pf in parquet_files:
                try:
                    table = pq.read_table(pf)
                    cols = table.column_names
                    key = "text"
                    for cand in ["text", "content", "document", "raw_content"]:
                        if cand in cols:
                            key = cand
                            break
                    col = table[key].to_pylist()
                    shard = pf.parent.name
                    for txt in col:
                        txtn = normalize_text(txt)
                        if not txtn:
                            continue
                        out.append({
                            "source": DATASET_NAMES["fineweb"],
                            "type": "doc",
                            "text": txtn,
                            "meta": {"shard": shard}
                        })
                except Exception:
                    continue
            break
    
    return out

def load_gutenberg_extracted(base_dir: Path):
    """Load Gutenberg from extracted text files"""
    gutenberg_dir = base_dir / "datasets" / "raw" / "gutenberg_rdf"
    if not gutenberg_dir.exists():
        return []
    
    out = []
    text_files = list(gutenberg_dir.rglob("*.txt"))
    limit = 1000  # Limit to avoid memory issues
    
    for tf in text_files[:limit]:
        try:
            with tf.open("r", encoding="utf-8") as fh:
                content = fh.read()
                text = normalize_text(content)
                if text and len(text) > 100:
                    out.append({
                        "source": DATASET_NAMES["gutenberg"],
                        "type": "doc",
                        "text": text,
                        "meta": {"file": tf.name}
                    })
        except Exception:
            continue
    
    return out

def load_wikipedia_extracted(base_dir: Path):
    """Load Wikipedia from extracted XML files"""
    wiki_dir = base_dir / "datasets" / "raw" / "enwiki-latest-pages-articles.xml"
    if not wiki_dir.exists():
        return []
    
    out = []
    xml_files = list(wiki_dir.rglob("*.xml"))
    limit = 10000  # Limit to avoid memory issues
    
    for xf in xml_files[:limit]:
        try:
            with xf.open("r", encoding="utf-8") as fh:
                content = fh.read()
                # Simple text extraction (could be improved with proper XML parsing)
                text = normalize_text(content)
                if text and len(text) > 100:
                    out.append({
                        "source": DATASET_NAMES["wikipedia"],
                        "type": "doc",
                        "text": text,
                        "meta": {"file": xf.name}
                    })
        except Exception:
            continue
    
    return out

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
    parser.add_argument("--include-docs-in-sft", action="store_true",
                        help="If set, will try to wrap docs into QA (not recommended). Default: exclude docs from SFT.")
    args = parser.parse_args()

    random.seed(RNG_SEED)

    base_dir = Path(args.base_dir)
    f_drive = Path(args.f_drive)
    raw_dir  = base_dir / "datasets" / "raw"
    out_dir  = base_dir / "datasets" / "organized"
    ensure_dir(out_dir)

    # Write Zia system prompt
    (out_dir / "zia_system_prompt.txt").write_text(DEFAULT_SYSTEM_PROMPT, encoding="utf-8")

    # Check for compressed files that need extraction
    wiki_file = base_dir / "datasets" / "raw" / "enwiki-latest-pages-articles.xml.bz2"
    gutenberg_file = base_dir / "datasets" / "raw" / "gutenberg_rdf.tar.bz2"
    
    if wiki_file.exists():
        print(f"[!] Found Wikipedia dump: {wiki_file}")
        print("    Note: This requires XML parsing with bz2 decompression. Consider extracting if needed.")
    
    if gutenberg_file.exists():
        print(f"[!] Found Gutenberg RDF: {gutenberg_file}")
        print("    Note: This requires RDF parsing with tar.bz2 decompression. Consider extracting if needed.")
    
    # Load datasets from both D and F drives
    datasets = load_all_datasets(base_dir, f_drive)

    # Report presence
    present = {
        DATASET_NAMES["alterego"]: len(datasets["alterego"]),
        DATASET_NAMES["openhermes"]: len(datasets["openhermes"]) + len(datasets["openhermes_f"]),
        DATASET_NAMES["ultrachat"]: len(datasets["ultrachat"]),
        DATASET_NAMES["zeroshot_financial"]: len(datasets["zeroshot_financial"]),
        DATASET_NAMES["fineweb"]: len(datasets["fineweb"]) + len(datasets["fineweb_f"]),
        DATASET_NAMES["openorca"]: len(datasets["openorca"]),
        DATASET_NAMES["empathetic_dialogues"]: len(datasets["empathetic_dialogues"]),
        DATASET_NAMES["financial_phrasebank"]: len(datasets["financial_phrasebank"]),
        DATASET_NAMES["reddit_tifu"]: len(datasets["reddit_tifu"]),
        DATASET_NAMES["chatgpt_export"]: len(datasets["chatgpt_export"]),
        DATASET_NAMES["wikipedia"]: len(datasets["wikipedia"]),
        DATASET_NAMES["gutenberg"]: len(datasets["gutenberg"]),
        DATASET_NAMES["financial_news"]: len(datasets["financial_news"]),
        DATASET_NAMES["json_datasets"]: len(datasets["json_datasets"]),
    }
    for k, v in present.items():
        print(f"    - {k}: {v} samples")

    # Deduplicate across all datasets
    print("[*] Cleaning & deduplicating...")
    seen_chat = set()
    seen_doc = set()
    dups = {"chat":0, "doc":0}

    def dedupe_chat(rows):
        nonlocal dups
        for r in rows:
            if r.get("type") != "chat":
                continue
            msgs = r.get("messages") or []
            # strip empty messages & non-user/assistant/system
            nm = []
            for m in msgs:
                role = (m.get("role") or "").lower()
                if role not in ("user","assistant","system"): 
                    continue
                content = normalize_text(m.get("content",""))
                if content:
                    nm.append({"role": role, "content": content})
            if len(nm) < 2:
                continue
            h = "chat:" + hash_chat(nm)
            if h in seen_chat:
                dups["chat"] += 1
                continue
            seen_chat.add(h)
            yield {"source": r.get("source",""), "type":"chat", "messages": nm}

    def dedupe_doc(rows):
        nonlocal dups
        for r in rows:
            if r.get("type") != "doc":
                continue
            txt = normalize_text(r.get("text",""))
            if not txt:
                continue
            h = "doc:" + hash_doc(txt)
            if h in seen_doc:
                dups["doc"] += 1
                continue
            seen_doc.add(h)
            out = {"source": r.get("source",""), "type":"doc", "text": txt}
            if "meta" in r and isinstance(r["meta"], dict):
                out["meta"] = r["meta"]
            yield out

    # Per-dataset cleans
    alterego_clean   = list(dedupe_chat(datasets["alterego"]))
    openhermes_clean = list(dedupe_chat(datasets["openhermes"] + datasets["openhermes_f"]))
    ultrachat_clean  = list(dedupe_chat(datasets["ultrachat"]))
    zeroshot_clean   = list(dedupe_chat(datasets["zeroshot_financial"]))
    fineweb_clean    = list(dedupe_doc(datasets["fineweb"] + datasets["fineweb_f"]))
    openorca_clean   = list(dedupe_chat(datasets["openorca"]))
    empathetic_clean = list(dedupe_chat(datasets["empathetic_dialogues"]))
    financial_phrasebank_clean = list(dedupe_chat(datasets["financial_phrasebank"]))
    reddit_tifu_clean = list(dedupe_chat(datasets["reddit_tifu"]))
    chatgpt_clean = list(dedupe_chat(datasets["chatgpt_export"]))
    financial_news_clean = list(dedupe_doc(datasets["financial_news"]))
    json_datasets_clean = list(dedupe_doc(datasets["json_datasets"]))

    # Write per-dataset files
    print("[*] Writing per-dataset JSONL files...")
    per_files = {}

    def _w(name, rows):
        p = out_dir / f"{name}.jsonl"
        write_jsonl(p, rows)
        per_files[name] = str(p)
        return p

    if alterego_clean:   _w(DATASET_NAMES["alterego"], alterego_clean)
    if openhermes_clean: _w(DATASET_NAMES["openhermes"], openhermes_clean)
    if ultrachat_clean:  _w(DATASET_NAMES["ultrachat"], ultrachat_clean)
    if zeroshot_clean:   _w(DATASET_NAMES["zeroshot_financial"], zeroshot_clean)
    if openorca_clean:   _w(DATASET_NAMES["openorca"], openorca_clean)
    if empathetic_clean: _w(DATASET_NAMES["empathetic_dialogues"], empathetic_clean)
    if financial_phrasebank_clean: _w(DATASET_NAMES["financial_phrasebank"], financial_phrasebank_clean)
    if reddit_tifu_clean: _w(DATASET_NAMES["reddit_tifu"], reddit_tifu_clean)
    if chatgpt_clean:   _w(DATASET_NAMES["chatgpt_export"], chatgpt_clean)
    if financial_news_clean: _w(DATASET_NAMES["financial_news"], financial_news_clean)
    if json_datasets_clean: _w(DATASET_NAMES["json_datasets"], json_datasets_clean)
    
    if fineweb_clean:
        # Store as DOCSTORE (not for SFT)
        p = out_dir / f"{DATASET_NAMES['fineweb']}.docstore.jsonl"
        write_jsonl(p, fineweb_clean)
        per_files[DATASET_NAMES["fineweb"] + ".docstore"] = str(p)

    # Merge chat datasets into SFT, with 8K windows
    print("[*] Building 8K SFT windows...")
    sft_sources = (alterego_clean + openhermes_clean + ultrachat_clean + zeroshot_clean + 
                   openorca_clean + empathetic_clean + financial_phrasebank_clean + 
                   reddit_tifu_clean + chatgpt_clean)
    random.shuffle(sft_sources)

    tok = TokenEstimator(args.tokenizer, args.context)
    merged_chunks = []
    for ex in tqdm(sft_sources, desc="chunking"):
        chunks = split_chat_to_windows(ex["messages"], tok, args.context)
        for c in chunks:
            merged_chunks.append({
                "source": ex["source"],
                "type": "chat",
                "messages": c
            })

    merged_path = out_dir / "merged_Zia_sft_8k.jsonl"
    write_jsonl(merged_path, merged_chunks)

    # Train/Val split
    print("[*] Splitting train/val...")
    n = len(merged_chunks)
    val_n = max(1, int(n * args.val_ratio))
    random.shuffle(merged_chunks)
    val = merged_chunks[:val_n]
    train = merged_chunks[val_n:]
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)

    # Stats & duplicate report
    stats = {
        "present": present,
        "written_files": per_files,
        "sft_samples_before_chunking": len(sft_sources),
        "sft_samples_after_chunking": len(merged_chunks),
        "train": len(train),
        "val": len(val),
        "deduplicated": dups,
        "tokenizer_used": args.tokenizer if tok.ok else "heuristic_approx_4chars_per_token",
        "context_tokens": args.context
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    (out_dir / "duplicates_report.json").write_text(json.dumps(dups, indent=2), encoding="utf-8")

    print("\n✅ Done.")
    print(f"  Per-dataset files: {json.dumps(per_files, indent=2)}")
    print(f"  Merged SFT: {merged_path}")
    print(f"  Train/Val: {out_dir / 'train.jsonl'} | {out_dir / 'val.jsonl'}")
    print(f"  Stats:     {out_dir / 'stats.json'}")
    print(f"  Duplicates:{out_dir / 'duplicates_report.json'}")
    if fineweb_clean:
        print(f"  Docstore:  {out_dir / (DATASET_NAMES['fineweb'] + '.docstore.jsonl')}")

if __name__ == "__main__":
    main()
