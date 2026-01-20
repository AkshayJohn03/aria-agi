#!/usr/bin/env python3
"""
verify_recovered_wiki.py
Show random cleaned Wikipedia samples for manual inspection.
"""

import random
import os

FILE = "artifacts/wiki_recovered_clean/wiki_clean_corpus.txt"

def main():
    print("=== VERIFY CLEAN WIKI SAMPLES ===")

    with open(FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"[i] Total lines: {len(lines)}")

    for _ in range(10):
        idx = random.randint(0, len(lines) - 1)
        print("\n------------------------")
        print(f"[Sample #{idx}]")
        print(lines[idx])
        print("------------------------")

if __name__ == "__main__":
    main()
