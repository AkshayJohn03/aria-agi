# -*- coding: utf-8 -*-
"""
merge_datasets.py

Simple script to merge existing organized datasets with newly processed missing datasets.
This avoids reprocessing everything from scratch.

Run:
  python merge_datasets.py --base-dir "D:\\aria\\aria_ai\\aria_ai_assistant"
"""

import json
from pathlib import Path
import argparse

def merge_datasets(base_dir: Path):
    """
    Merge existing organized datasets with missing datasets
    """
    organized_dir = base_dir / "datasets" / "organized"
    missing_dir = base_dir / "datasets" / "missing"
    
    if not organized_dir.exists():
        print("Error: Organized directory not found!")
        return
    
    if not missing_dir.exists():
        print("Error: Missing datasets directory not found!")
        return
    
    print("[*] Loading existing organized datasets...")
    
    # Load existing chat datasets
    existing_chat = []
    existing_files = [
        "alterego_uncensored_personal.jsonl",
        "openhermes_uncensored_general.jsonl", 
        "zeroshot_financial_filtered_financial.jsonl",
        "financial_news_articles.jsonl"
    ]
    
    for filename in existing_files:
        filepath = organized_dir / filename
        if filepath.exists():
            print(f"  Loading {filename}...")
            try:
                with filepath.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            row = json.loads(line)
                            if row.get("type") == "chat":
                                existing_chat.append(row)
                        except Exception:
                            continue
            except Exception as e:
                print(f"    Error loading {filename}: {e}")
    
    print(f"  Loaded {len(existing_chat)} existing chat samples")
    
    # Load existing docs
    existing_docs = []
    doc_files = [
        "fineweb_filtered_general.docstore.jsonl",
        "json_misc_datasets.jsonl"
    ]
    
    for filename in doc_files:
        filepath = organized_dir / filename
        if filepath.exists():
            print(f"  Loading {filename}...")
            try:
                with filepath.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            row = json.loads(line)
                            if row.get("type") == "doc":
                                existing_docs.append(row)
                        except Exception:
                            continue
            except Exception as e:
                print(f"    Error loading {filename}: {e}")
    
    print(f"  Loaded {len(existing_docs)} existing doc samples")
    
    # Load missing datasets
    print("\n[*] Loading missing datasets...")
    missing_chat = []
    missing_docs = []
    
    missing_chat_files = [
        "ultrachat_filtered_dialogue.jsonl",
        "openorca_instruction_following.jsonl",
        "empathetic_dialogues_emotional.jsonl",
        "financial_phrasebank_sentiment.jsonl",
        "reddit_tifu_storytelling.jsonl",
        "chatgpt_export_personal.jsonl"
    ]
    
    for filename in missing_chat_files:
        filepath = missing_dir / filename
        if filepath.exists():
            print(f"  Loading {filename}...")
            try:
                with filepath.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            row = json.loads(line)
                            if row.get("type") == "chat":
                                missing_chat.append(row)
                        except Exception:
                            continue
            except Exception as e:
                print(f"    Error loading {filename}: {e}")
    
    missing_doc_files = [
        "wikipedia_encyclopedia.jsonl",
        "gutenberg_literature.jsonl"
    ]
    
    for filename in missing_doc_files:
        filepath = missing_dir / filename
        if filepath.exists():
            print(f"  Loading {filename}...")
            try:
                with filepath.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            row = json.loads(line)
                            if row.get("type") == "doc":
                                missing_docs.append(row)
                        except Exception:
                            continue
            except Exception as e:
                print(f"    Error loading {filename}: {e}")
    
    print(f"  Loaded {len(missing_chat)} missing chat samples")
    print(f"  Loaded {len(missing_docs)} missing doc samples")
    
    # Combine all data
    print("\n[*] Combining datasets...")
    all_chat = existing_chat + missing_chat
    all_docs = existing_docs + missing_docs
    
    print(f"  Total chat samples: {len(all_chat)}")
    print(f"  Total doc samples: {len(all_docs)}")
    
    # Write complete datasets
    print("\n[*] Writing complete datasets...")
    
    # Write all chat data
    final_chat_path = organized_dir / "complete_chat_datasets.jsonl"
    with final_chat_path.open("w", encoding="utf-8") as fh:
        for row in all_chat:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    # Write all doc data
    final_docs_path = organized_dir / "complete_document_datasets.jsonl"
    with final_docs_path.open("w", encoding="utf-8") as fh:
        for row in all_docs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    # Create final stats
    final_stats = {
        "existing_chat_samples": len(existing_chat),
        "missing_chat_samples": len(missing_chat),
        "total_chat_samples": len(all_chat),
        "existing_doc_samples": len(existing_docs),
        "missing_doc_samples": len(missing_docs),
        "total_doc_samples": len(all_docs),
        "files_processed": {
            "existing": existing_files + doc_files,
            "missing": missing_chat_files + missing_doc_files
        }
    }
    
    (organized_dir / "merge_stats.json").write_text(json.dumps(final_stats, indent=2), encoding="utf-8")
    
    print("\n✅ Merge complete!")
    print(f"  Complete chat: {final_chat_path}")
    print(f"  Complete docs: {final_docs_path}")
    print(f"  Merge stats: {organized_dir / 'merge_stats.json'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=str(Path(__file__).resolve().parents[2]),
                        help="Project root (aria_ai_assistant)")
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    merge_datasets(base_dir)

if __name__ == "__main__":
    main() 