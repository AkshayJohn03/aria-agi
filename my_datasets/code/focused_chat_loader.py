#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
focused_chat_loader.py

Focused loader that only processes essential chat datasets for SFT training,
avoiding massive document datasets to stay within storage limits.

Essential datasets:
- UltraChat (dialogue)
- Open-Orca (instruction following)
- OpenHermes (general chat)
- AlterEgo (personal)
- ChatGPT Export (conversations)
- Financial datasets (sentiment/classification)

Run:
  python focused_chat_loader.py --base-dir "D:\aria\aria_ai\aria_ai_assistant" --f-drive "F:\aria_ai\aria_ai_assistant"
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
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs): return it

# Configuration - Essential datasets including Wikipedia
ESSENTIAL_DATASETS = {
    "ultrachat": "ultrachat_filtered_dialogue",
    "openorca": "openorca_instruction_following", 
    "openhermes": "openhermes_uncensored_general",
    "alterego": "alterego_uncensored_personal",
    "wikipedia": "wikipedia_encyclopedia",
    "gutenberg": "gutenberg_literature",
    "financial_news": "financial_news_articles",
    "empathetic_dialogues": "empathetic_dialogues_emotional",
    "financial_phrasebank": "financial_phrasebank_sentiment",
    "reddit_tifu": "reddit_tifu_storytelling",
    "zeroshot_financial": "zeroshot_financial_filtered_financial",
}

def normalize_text(s: str) -> str:
    """Normalize text content"""
    if s is None:
        return ""
    # Strip control chars, collapse whitespace, normalize newlines
    s = re.sub(r'[\u0000-\u0008\u000B-\u000C\u000E-\u001F]', "", str(s))
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r'[ \t\u00A0\u200B\u200C\u200D]+', " ", s)
    return s.strip()

def load_ultrachat_focused(base_dir: Path):
    """Load UltraChat from both raw JSONL and HF cache"""
    print("  Loading UltraChat...")
    out = []
    
    # 1. Raw JSONL file (512 MB)
    raw_file = base_dir / "datasets" / "raw" / "ultrachat.jsonl"
    if raw_file.exists():
        print(f"    Processing raw JSONL: {raw_file}")
        raw_samples = load_ultrachat_raw(raw_file)
        out.extend(raw_samples)
        print(f"    Raw JSONL: {len(raw_samples)} samples")
    
    # 2. HF Cache arrow files (2.5+ GB)
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
                                        "source": ESSENTIAL_DATASETS["ultrachat"],
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
            print(f"      Processing {af.name}...")
            
            # Try different approaches to read the file
            try:
                # Method 1: Try pyarrow IPC
                import pyarrow as pa
                table = pa.ipc.open_file(af).read_all()
                print(f"        Successfully read as Arrow IPC with columns: {table.column_names}")
                
                if "messages" in table.column_names:
                    messages_col = table.column("messages")
                    for i in range(len(messages_col)):
                        msgs = messages_col[i].as_py()
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
                                    "source": ESSENTIAL_DATASETS["ultrachat"],
                                    "type": "chat",
                                    "messages": norm
                                })
                
                elif "prompt" in table.column_names and "response" in table.column_names:
                    prompt_col = table.column("prompt")
                    response_col = table.column("response")
                    for i in range(len(prompt_col)):
                        prompt = normalize_text(prompt_col[i].as_py())
                        response = normalize_text(response_col[i].as_py())
                        if prompt and response:
                            out.append({
                                "source": ESSENTIAL_DATASETS["ultrachat"],
                                "type": "chat",
                                "messages": [
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": response}
                                ]
                            })
                
            except Exception as e1:
                print(f"        Arrow IPC failed: {e1}")
                
                try:
                    # Method 2: Try pandas read_feather
                    if _HAS_PANDAS:
                        data = pd.read_feather(af)
                        print(f"        Successfully read as Feather with columns: {list(data.columns)}")
                        
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
                                            "source": ESSENTIAL_DATASETS["ultrachat"],
                                            "type": "chat",
                                            "messages": norm
                                        })
                        
                        elif "prompt" in data.columns and "response" in data.columns:
                            for _, row in data.iterrows():
                                prompt = normalize_text(row.get("prompt", ""))
                                response = normalize_text(row.get("response", ""))
                                if prompt and response:
                                    out.append({
                                        "source": ESSENTIAL_DATASETS["ultrachat"],
                                        "type": "chat",
                                        "messages": [
                                            {"role": "user", "content": prompt},
                                            {"role": "assistant", "content": response}
                                        ]
                                    })
                
                except Exception as e2:
                    print(f"        Pandas Feather failed: {e2}")
                    
                    try:
                        # Method 3: Try reading as binary to check file header
                        with open(af, 'rb') as f:
                            header = f.read(16)
                            print(f"        File header: {header[:8].hex()}")
                            
                            # Check if it's a parquet file
                            if header.startswith(b'PAR1'):
                                print(f"        File appears to be Parquet, not Arrow")
                                if _HAS_PANDAS:
                                    data = pd.read_parquet(af)
                                    print(f"        Successfully read as Parquet with columns: {list(data.columns)}")
                                    # Process parquet data similar to above
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
                                                        "source": ESSENTIAL_DATASETS["ultrachat"],
                                                        "type": "chat",
                                                        "messages": norm
                                                    })
                    
                    except Exception as e3:
                        print(f"        All reading methods failed: {e3}")
                        continue
        
        except Exception as e:
            print(f"      Error reading {af.name}: {e}")
            continue
    
    return out

def load_openorca_focused(base_dir: Path):
    """Load Open-Orca from raw JSONL file"""
    print("  Loading Open-Orca...")
    out = []
    
    raw_file = base_dir / "datasets" / "raw" / "openorca.jsonl"
    if raw_file.exists():
        print(f"    Processing raw JSONL: {raw_file}")
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
                                    "source": ESSENTIAL_DATASETS["openorca"],
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
    
    print(f"    Open-Orca: {len(out)} samples")
    return out

def load_openhermes_focused(f_drive: Path):
    """Load OpenHermes from F drive (larger version)"""
    print("  Loading OpenHermes from F drive...")
    out = []
    
    hermes_file = f_drive / "datasets" / "raw" / "openhermes.jsonl"
    if hermes_file.exists():
        print(f"    Processing: {hermes_file}")
        try:
            with hermes_file.open("r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    try:
                        row = json.loads(line)
                        # Try to extract instruction/response format
                        instruction = normalize_text(row.get("instruction", row.get("prompt", "")))
                        response = normalize_text(row.get("response", row.get("output", "")))
                        if instruction and response:
                            out.append({
                                "source": ESSENTIAL_DATASETS["openhermes"],
                                "type": "chat",
                                "messages": [
                                    {"role": "user", "content": instruction},
                                    {"role": "assistant", "content": response}
                                ]
                            })
                        
                        if i % 100000 == 0:
                            print(f"      Processed {i} lines, found {len(out)} samples...")
                            
                    except Exception:
                        continue
        except Exception as e:
            print(f"    Error reading OpenHermes file: {e}")
    
    print(f"    OpenHermes: {len(out)} samples")
    return out

def load_alterego_focused(base_dir: Path):
    """Load AlterEgo dataset"""
    print("  Loading AlterEgo...")
    out = []
    
    root = base_dir / "datasets" / "raw" / "alterego"
    f = root / "alterego.jsonl"
    if not f.exists():
        print(f"    AlterEgo file not found at: {f}")
        return out
    
    try:
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    
                    # First try to load as conversation format
                    msgs = row.get("messages") or []
                    if msgs:
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
                                "source": ESSENTIAL_DATASETS["alterego"],
                                "type": "chat",
                                "messages": norm
                            })
                    else:
                        # Try to load as individual message format
                        role = (row.get("role") or "").lower()
                        if role in ("user", "assistant"):
                            content = normalize_text(row.get("content", ""))
                            if content:
                                # Store as single message for now
                                out.append({
                                    "source": ESSENTIAL_DATASETS["alterego"],
                                    "type": "chat",
                                    "messages": [{"role": role, "content": content}]
                                })
                        
                except Exception:
                    continue
    except Exception as e:
        print(f"    Error reading AlterEgo file: {e}")
    
    print(f"    AlterEgo: {len(out)} samples")
    return out

def load_chatgpt_export_focused(base_dir: Path):
    """Load ChatGPT export conversations"""
    print("  Loading ChatGPT export...")
    out = []
    
    root = base_dir / "datasets" / "raw" / "chatgpt_export"
    if not root.exists():
        print(f"    ChatGPT export directory not found at: {root}")
        return out
    
    # Try conversations.json
    conversations_file = root / "conversations.json"
    if conversations_file.exists():
        print(f"    Processing conversations.json")
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
                                "source": ESSENTIAL_DATASETS["chatgpt_export"],
                                "type": "chat",
                                "messages": norm
                            })
        except Exception as e:
            print(f"    Error reading conversations.json: {e}")
    
    # Also try chat.html if conversations.json failed
    if not out:
        chat_file = root / "chat.html"
        if chat_file.exists():
            print(f"    Processing chat.html")
            try:
                with chat_file.open("r", encoding="utf-8") as fh:
                    content = fh.read()
                    # Simple HTML parsing to extract conversations
                    # Look for conversation patterns in HTML
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
                                "source": ESSENTIAL_DATASETS["chatgpt_export"],
                                "type": "chat",
                                "messages": [
                                    {"role": "user", "content": user_text},
                                    {"role": "assistant", "content": assistant_text}
                                ]
                            })
            except Exception as e:
                print(f"    Error reading chat.html: {e}")
    
    print(f"    ChatGPT Export: {len(out)} samples")
    return out

def load_wikipedia_focused(base_dir: Path, max_articles: int = 50000):
    """Load Wikipedia articles with limits to stay within storage"""
    print("  Loading Wikipedia...")
    out = []
    
    wiki_dir = base_dir / "datasets" / "raw" / "enwiki-latest-pages-articles.xml"
    if not wiki_dir.exists():
        print(f"    Wikipedia directory not found at: {wiki_dir}")
        return out
    
    # Look for the main XML file
    xml_files = list(wiki_dir.rglob("*.xml"))
    if not xml_files:
        print(f"    No XML files found in Wikipedia directory")
        return out
    
    print(f"    Found {len(xml_files)} XML files, processing up to {max_articles} articles...")
    
    article_count = 0
    for xf in xml_files:
        if article_count >= max_articles:
            break
            
        try:
            print(f"    Processing {xf.name}...")
            
            # Read file line by line to avoid memory issues
            current_page = ""
            in_page = False
            in_title = False
            in_text = False
            current_title = ""
            current_text = ""
            
            with xf.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    
                    if "<page>" in line:
                        in_page = True
                        current_title = ""
                        current_text = ""
                        continue
                    
                    if "</page>" in line:
                        in_page = False
                        # Process the completed page
                        if current_title and current_text and len(current_text) > 100:
                            # Skip non-article pages
                            if not (current_title.startswith("User:") or current_title.startswith("Talk:") or 
                                   current_title.startswith("File:") or current_title.startswith("Template:") or
                                   current_title.startswith("Help:") or current_title.startswith("Category:")):
                                
                                clean_text = normalize_text(current_text)
                                if clean_text and len(clean_text) > 100:
                                    out.append({
                                        "source": ESSENTIAL_DATASETS["wikipedia"],
                                        "type": "chat",
                                        "messages": [
                                            {"role": "user", "content": f"Tell me about: {current_title}"},
                                            {"role": "assistant", "content": clean_text[:2000]}  # Limit response length
                                        ]
                                    })
                                    article_count += 1
                                    
                                    if article_count % 1000 == 0:
                                        print(f"      Processed {article_count} articles...")
                                    
                                    if article_count >= max_articles:
                                        break
                        continue
                    
                    if in_page:
                        if "<title>" in line:
                            in_title = True
                            current_title = line.replace("<title>", "").replace("</title>", "")
                            continue
                        
                        if "<text>" in line:
                            in_text = True
                            current_text = line.replace("<text>", "").replace("</text>", "")
                            continue
                        
                        if in_text and "</text>" not in line:
                            current_text += " " + line
                        
                        if "</text>" in line:
                            in_text = False
                            current_text += " " + line.replace("</text>", "")
        
        except Exception as e:
            print(f"    Error reading {xf.name}: {e}")
            continue
    
    print(f"    Wikipedia: {len(out)} articles processed")
    return out

def load_gutenberg_focused(base_dir: Path, max_books: int = 1000):
    """Load Gutenberg books with limits"""
    print("  Loading Gutenberg...")
    out = []
    
    gutenberg_dir = base_dir / "datasets" / "raw" / "gutenberg_rdf"
    if not gutenberg_dir.exists():
        print(f"    Gutenberg directory not found at: {gutenberg_dir}")
        return out
    
    # Look for RDF files in the cache/epub subdirectories
    rdf_files = list(gutenberg_dir.rglob("*.rdf"))
    if not rdf_files:
        print(f"    No RDF files found in Gutenberg directory")
        return out
    
    print(f"    Found {len(rdf_files)} RDF files, processing up to {max_books} books...")
    
    book_count = 0
    for rf in rdf_files[:max_books]:
        try:
            with rf.open("r", encoding="utf-8") as fh:
                content = fh.read()
                
                # Simple RDF parsing to extract book information
                # Look for title and text content
                import re
                title_match = re.search(r'<dcterms:title>(.*?)</dcterms:title>', content)
                creator_match = re.search(r'<dcterms:creator>(.*?)</dcterms:creator>', content)
                
                if title_match:
                    title = normalize_text(title_match.group(1))
                    creator = normalize_text(creator_match.group(1)) if creator_match else "Unknown Author"
                    
                    # Extract some text content if available
                    # Look for any text content in the RDF
                    text_content = normalize_text(content)
                    if text_content and len(text_content) > 200:
                        # Convert to Q&A format
                        out.append({
                            "source": ESSENTIAL_DATASETS["gutenberg"],
                            "type": "chat",
                            "messages": [
                                {"role": "user", "content": f"Tell me about the book '{title}' by {creator}"},
                                {"role": "assistant", "content": f"This is '{title}' by {creator}. The book is available in the Gutenberg Project collection. Would you like me to provide more specific information about this work?"}
                            ]
                        })
                        book_count += 1
                        
                        if book_count % 100 == 0:
                            print(f"      Processed {book_count} books...")
        
        except Exception as e:
            print(f"    Error reading {rf.name}: {e}")
            continue
    
    print(f"    Gutenberg: {len(out)} books processed")
    return out

def load_empathetic_dialogues_focused(base_dir: Path):
    """Load Empathetic Dialogues dataset"""
    print("  Loading Empathetic Dialogues...")
    out = []
    
    # Try multiple possible paths
    possible_paths = [
        base_dir / "datasets" / "raw" / "hf_cache" / "hub" / "datasets--empathetic_dialogues" / "snapshots",
        base_dir / "datasets" / "raw" / "hf_cache" / "datasets" / "empathetic_dialogues" / "default" / "0.0.0"
    ]
    
    for root in possible_paths:
        if not root.exists():
            continue
            
        print(f"    Found path: {root}")
        # Look for data files in snapshots
        for snapshot_dir in root.iterdir():
            if snapshot_dir.is_dir():
                data_files = list(snapshot_dir.rglob("*.arrow"))
                if not data_files:
                    continue
                    
                for df in data_files:
                    try:
                        if df.suffix == '.arrow':
                            # Use pyarrow directly
                            import pyarrow as pa
                            table = pa.ipc.open_file(df).read_all()
                            print(f"      Processing {df.name} with columns: {table.column_names}")
                            
                            # Look for conversation columns
                            if "context" in table.column_names and "response" in table.column_names:
                                context_col = table.column("context")
                                response_col = table.column("response")
                                for i in range(len(context_col)):
                                    context = normalize_text(context_col[i].as_py())
                                    response = normalize_text(response_col[i].as_py())
                                    if context and response:
                                        out.append({
                                            "source": ESSENTIAL_DATASETS["empathetic_dialogues"],
                                            "type": "chat",
                                            "messages": [
                                                {"role": "user", "content": context},
                                                {"role": "assistant", "content": response}
                                            ]
                                        })
                            
                            elif "input" in table.column_names and "output" in table.column_names:
                                input_col = table.column("input")
                                output_col = table.column("output")
                                for i in range(len(input_col)):
                                    input_text = normalize_text(input_col[i].as_py())
                                    output = normalize_text(output_col[i].as_py())
                                    if input_text and output:
                                        out.append({
                                            "source": ESSENTIAL_DATASETS["empathetic_dialogues"],
                                            "type": "chat",
                                            "messages": [
                                                {"role": "user", "content": input_text},
                                                {"role": "assistant", "content": output}
                                            ]
                                        })
                    
                    except Exception as e:
                        print(f"      Error reading {df.name}: {e}")
                        continue
                break  # Only process first snapshot
    
    print(f"    Empathetic Dialogues: {len(out)} samples")
    return out

def load_financial_phrasebank_focused(base_dir: Path):
    """Load Financial Phrasebank dataset"""
    print("  Loading Financial Phrasebank...")
    out = []
    
    # Try multiple possible paths
    possible_paths = [
        base_dir / "datasets" / "raw" / "hf_cache" / "hub" / "datasets--financial_phrasebank" / "snapshots",
        base_dir / "datasets" / "raw" / "hf_cache" / "datasets" / "financial_phrasebank" / "default" / "0.0.0"
    ]
    
    for root in possible_paths:
        if not root.exists():
            continue
            
        print(f"    Found path: {root}")
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
                                        "source": ESSENTIAL_DATASETS["financial_phrasebank"],
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
                if arrow_files:
                    for af in arrow_files:
                        try:
                            if af.suffix == '.arrow':
                                import pyarrow as pa
                                table = pa.ipc.open_file(af).read_all()
                                
                                # Look for text and sentiment columns
                                if "text" in table.column_names and "sentiment" in table.column_names:
                                    text_col = table.column("text")
                                    sentiment_col = table.column("sentiment")
                                    for i in range(len(text_col)):
                                        text = normalize_text(text_col[i].as_py())
                                        sentiment = normalize_text(sentiment_col[i].as_py())
                                        if text and sentiment:
                                            out.append({
                                                "source": ESSENTIAL_DATASETS["financial_phrasebank"],
                                                "type": "chat",
                                                "messages": [
                                                    {"role": "user", "content": f"Classify the sentiment of this financial text:\n\n{text}"},
                                                    {"role": "assistant", "content": sentiment}
                                                ]
                                            })
                        
                        except Exception as e:
                            print(f"      Error reading {af.name}: {e}")
                            continue
                break  # Only process first snapshot
    
    print(f"    Financial Phrasebank: {len(out)} samples")
    return out

def load_reddit_tifu_focused(base_dir: Path):
    """Load Reddit TIFU dataset"""
    print("  Loading Reddit TIFU...")
    out = []
    
    # Try multiple possible paths
    possible_paths = [
        base_dir / "datasets" / "raw" / "hf_cache" / "hub" / "datasets--reddit_tifu" / "snapshots",
        base_dir / "datasets" / "raw" / "hf_cache" / "datasets" / "reddit_tifu" / "default" / "0.0.0"
    ]
    
    for root in possible_paths:
        if not root.exists():
            continue
            
        print(f"    Found path: {root}")
        # Look for data files in snapshots
        for snapshot_dir in root.iterdir():
            if snapshot_dir.is_dir():
                data_files = list(snapshot_dir.rglob("*.arrow"))
                if not data_files:
                    continue
                    
                for df in data_files:
                    try:
                        if df.suffix == '.arrow':
                            import pyarrow as pa
                            table = pa.ipc.open_file(df).read_all()
                            print(f"      Processing {df.name} with columns: {table.column_names}")
                            
                            # Look for story and title columns
                            if "story" in table.column_names and "title" in table.column_names:
                                story_col = table.column("story")
                                title_col = table.column("title")
                                for i in range(len(story_col)):
                                    title = normalize_text(title_col[i].as_py())
                                    story = normalize_text(story_col[i].as_py())
                                    if title and story:
                                        out.append({
                                            "source": ESSENTIAL_DATASETS["reddit_tifu"],
                                            "type": "chat",
                                            "messages": [
                                                {"role": "user", "content": f"Tell me a story about: {title}"},
                                                {"role": "assistant", "content": story}
                                            ]
                                        })
                            
                            elif "text" in table.column_names and "title" in table.column_names:
                                text_col = table.column("text")
                                title_col = table.column("title")
                                for i in range(len(text_col)):
                                    title = normalize_text(title_col[i].as_py())
                                    text = normalize_text(text_col[i].as_py())
                                    if title and text:
                                        out.append({
                                            "source": ESSENTIAL_DATASETS["reddit_tifu"],
                                            "type": "chat",
                                            "messages": [
                                                {"role": "user", "content": f"Tell me a story about: {title}"},
                                                {"role": "assistant", "content": text}
                                            ]
                                        })
                
                    except Exception as e:
                        print(f"      Error reading {df.name}: {e}")
                        continue
                break  # Only process first snapshot
    
    print(f"    Reddit TIFU: {len(out)} samples")
    return out

def load_financial_datasets_focused(base_dir: Path, f_drive: Path):
    """Load financial-related datasets"""
    print("  Loading financial datasets...")
    out = []
    
    # Financial news from F drive
    news_file = f_drive / "datasets" / "raw" / "fin_news.jsonl"
    if news_file.exists():
        print(f"    Processing financial news: {news_file}")
        try:
            with news_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                        text = normalize_text(row.get("text", row.get("content", "")))
                        if text:
                            out.append({
                                "source": ESSENTIAL_DATASETS["financial_news"],
                                "type": "chat",
                                "messages": [
                                    {"role": "user", "content": f"Tell me about this financial news:\n\n{text}"},
                                    {"role": "assistant", "content": "This appears to be a financial news article. I can help analyze the content if you have specific questions about it."}
                                ]
                            })
                    except Exception:
                        continue
        except Exception as e:
            print(f"    Error reading financial news: {e}")
    
    # Zeroshot financial from D drive
    zeroshot_dir = base_dir / "datasets" / "raw" / "hf_cache" / "hub" / "datasets--zeroshot--twitter-financial-news-sentiment" / "snapshots"
    if zeroshot_dir.exists():
        print(f"    Processing zeroshot financial")
        csvs = list(zeroshot_dir.rglob("*.csv"))
        label_map = {"0":"negative","1":"neutral","2":"positive"}
        
        for cf in csvs:
            try:
                with cf.open("r", encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    if reader.fieldnames:
                        fields = [f.lower() for f in reader.fieldnames]
                        text_key = next((k for k in reader.fieldnames if k.lower() in ("text","tweet","sentence","content")), reader.fieldnames[0])
                        label_key = next((k for k in reader.fieldnames if k.lower() in ("label","sentiment","target")), reader.fieldnames[-1])
                        
                        for row in reader:
                            text = normalize_text(row.get(text_key,""))
                            lab = str(row.get(label_key,"")).strip()
                            if not text:
                                continue
                            label_str = label_map.get(lab, lab)
                            out.append({
                                "source": ESSENTIAL_DATASETS["zeroshot_financial"],
                                "type": "chat",
                                "messages": [
                                    {"role": "user", "content": f"Classify the sentiment (negative/neutral/positive) of this financial tweet:\n\n{text}"},
                                    {"role": "assistant", "content": label_str}
                                ]
                            })
            except Exception:
                continue
    
    print(f"    Financial datasets: {len(out)} samples")
    return out

def main():
    parser = argparse.ArgumentParser(description="Focused chat dataset loader for SFT training")
    parser.add_argument("--base-dir", type=str, default=str(Path(__file__).resolve().parents[2]),
                        help="Project root (aria_ai_assistant)")
    parser.add_argument("--f-drive", type=str, default="F:\\aria_ai\\aria_ai_assistant",
                        help="F drive project path")
    parser.add_argument("--output-dir", type=str, default="datasets/focused_chat",
                        help="Output directory for focused chat datasets")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    f_drive = Path(args.f_drive)
    output_dir = Path(args.output_dir)
    
    print(f"[*] Focused Chat Dataset Loader")
    print(f"    D Drive: {base_dir}")
    print(f"    F Drive: {f_drive}")
    print(f"    Output: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load essential chat datasets only
    print(f"\n[*] Loading essential chat datasets...")
    
    datasets = {}
    datasets["ultrachat"] = load_ultrachat_focused(base_dir)
    datasets["openorca"] = load_openorca_focused(base_dir)
    datasets["openhermes"] = load_openhermes_focused(f_drive)
    datasets["alterego"] = load_alterego_focused(base_dir)
    datasets["wikipedia"] = load_wikipedia_focused(base_dir)
    datasets["gutenberg"] = load_gutenberg_focused(base_dir)
    datasets["empathetic_dialogues"] = load_empathetic_dialogues_focused(base_dir)
    datasets["financial_phrasebank"] = load_financial_phrasebank_focused(base_dir)
    datasets["reddit_tifu"] = load_reddit_tifu_focused(base_dir)
    datasets["financial"] = load_financial_datasets_focused(base_dir, f_drive)
    
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
    
    # Check if within 40GB limit
    if estimated_storage_mb > 40000:
        print(f"[!] WARNING: Estimated storage ({estimated_storage_mb:.1f} MB) exceeds 40GB limit!")
        print(f"    Consider reducing dataset size or processing in chunks.")
    else:
        print(f"[✓] Estimated storage is within 40GB limit.")
    
    # Save datasets
    print(f"\n[*] Saving focused chat datasets...")
    for name, samples in datasets.items():
        if samples:
            output_file = output_dir / f"{name}.jsonl"
            with output_file.open("w", encoding="utf-8") as fh:
                for sample in samples:
                    fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"    Saved {name}: {len(samples)} samples -> {output_file}")
    
    print(f"\n[✓] Focused chat datasets saved to: {output_dir}")

if __name__ == "__main__":
    main() 