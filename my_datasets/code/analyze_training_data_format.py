#!/usr/bin/env python3
"""
analyze_training_data_format.py
Analyzes the training data format to understand how the model expects input/output.
"""

import os
import json
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
import random

def analyze_dataset_format(dataset_path, num_samples=10):
    """Analyze the format of training data"""
    
    print(f"🔍 Analyzing dataset: {dataset_path}")
    print("="*80)
    
    try:
        # Load dataset
        if os.path.exists(os.path.join(dataset_path, "train")):
            dataset = load_from_disk(os.path.join(dataset_path, "train"))
        else:
            dataset = load_from_disk(dataset_path)
        
        print(f"📊 Dataset Info:")
        print(f"   Total samples: {len(dataset):,}")
        print(f"   Features: {dataset.features}")
        print(f"   Column names: {dataset.column_names}")
        
        # Sample random examples
        sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        
        print(f"\n📝 Sample Training Examples:")
        print("="*80)
        
        for i, idx in enumerate(sample_indices):
            example = dataset[idx]
            print(f"\n--- Example {i+1}/{num_samples} (Index: {idx}) ---")
            
            if isinstance(example, dict):
                for key, value in example.items():
                    print(f"{key}: {str(value)[:500]}{'...' if len(str(value)) > 500 else ''}")
            else:
                print(f"Raw text: {str(example)[:500]}{'...' if len(str(example)) > 500 else ''}")
            
            print("-" * 40)
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def analyze_instruction_patterns(dataset, num_samples=20):
    """Look for instruction-following patterns in the data"""
    
    print(f"\n📋 Instruction Pattern Analysis:")
    print("="*80)
    
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    instruction_patterns = {
        "### Instruction:": 0,
        "Human:": 0,
        "Assistant:": 0,
        "Question:": 0,
        "Answer:": 0,
        "Instruction:": 0,
        "Response:": 0,
        "User:": 0,
        "Bot:": 0,
        "Q:": 0,
        "A:": 0,
    }
    
    conversation_patterns = {
        "Human:": 0,
        "Assistant:": 0,
        "User:": 0,
        "Bot:": 0,
        "System:": 0,
    }
    
    for idx in sample_indices:
        example = dataset[idx]
        text = example["text"] if isinstance(example, dict) else str(example)
        
        # Check for instruction patterns
        for pattern in instruction_patterns:
            if pattern in text:
                instruction_patterns[pattern] += 1
        
        # Check for conversation patterns
        for pattern in conversation_patterns:
            if pattern in text:
                conversation_patterns[pattern] += 1
    
    print("Instruction patterns found:")
    for pattern, count in instruction_patterns.items():
        if count > 0:
            print(f"   {pattern}: {count}/{num_samples} samples")
    
    print("\nConversation patterns found:")
    for pattern, count in conversation_patterns.items():
        if count > 0:
            print(f"   {pattern}: {count}/{num_samples} samples")

def main():
    """Main analysis function"""
    
    print("🔍 ZIA Model Training Data Format Analysis")
    print("="*80)
    
    # Dataset paths
    dataset_paths = [
        "my_datasets/processed/arrow_cleaned_v1",
        "artifacts/tokenized_datasets/zia_ift_v3",
    ]
    
    # Analyze each dataset
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            dataset = analyze_dataset_format(dataset_path, num_samples=5)
            if dataset is not None:
                analyze_instruction_patterns(dataset, num_samples=20)
        else:
            print(f"⚠️ Dataset not found: {dataset_path}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()