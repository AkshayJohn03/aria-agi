#!/usr/bin/env python3
"""
download_instruction_datasets.py
Download high-quality instruction-following datasets for fine-tuning.
"""

import os
import json
import requests
from datasets import load_dataset
from transformers import AutoTokenizer

def download_alpaca_dataset():
    """Download and process Alpaca dataset"""
    print("📥 Downloading Alpaca dataset...")
    
    try:
        # Load Alpaca dataset
        dataset = load_dataset("tatsu-lab/alpaca")
        
        # Process and save
        processed_data = []
        for item in dataset['train']:
            if item['instruction'] and item['output']:
                processed_data.append({
                    "instruction": item['instruction'],
                    "response": item['output']
                })
        
        # Save to file
        output_path = "my_datasets/processed/alpaca_instructions.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Alpaca dataset saved: {len(processed_data)} examples")
        return processed_data
        
    except Exception as e:
        print(f"❌ Error downloading Alpaca: {e}")
        return []

def download_dolly_dataset():
    """Download and process Dolly dataset"""
    print("📥 Downloading Dolly dataset...")
    
    try:
        # Load Dolly dataset
        dataset = load_dataset("databricks/databricks-dolly-15k")
        
        # Process and save
        processed_data = []
        for item in dataset['train']:
            if item['instruction'] and item['response']:
                processed_data.append({
                    "instruction": item['instruction'],
                    "response": item['response']
                })
        
        # Save to file
        output_path = "my_datasets/processed/dolly_instructions.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Dolly dataset saved: {len(processed_data)} examples")
        return processed_data
        
    except Exception as e:
        print(f"❌ Error downloading Dolly: {e}")
        return []

def download_openassistant_dataset():
    """Download and process OpenAssistant dataset"""
    print("📥 Downloading OpenAssistant dataset...")
    
    try:
        # Load OpenAssistant dataset
        dataset = load_dataset("OpenAssistant/oasst1")
        
        # Process and save
        processed_data = []
        for item in dataset['train']:
            if item['text'] and len(item['text']) > 10:
                # Simple processing - take first 1000 examples
                if len(processed_data) >= 1000:
                    break
                    
                processed_data.append({
                    "instruction": f"Please respond to: {item['text'][:100]}...",
                    "response": item['text']
                })
        
        # Save to file
        output_path = "my_datasets/processed/openassistant_instructions.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ OpenAssistant dataset saved: {len(processed_data)} examples")
        return processed_data
        
    except Exception as e:
        print(f"❌ Error downloading OpenAssistant: {e}")
        return []

def download_self_instruct_dataset():
    """Download Self-Instruct dataset"""
    print("📥 Downloading Self-Instruct dataset...")
    
    try:
        # Load Self-Instruct dataset
        dataset = load_dataset("allenai/self_instruct")
        
        # Process and save
        processed_data = []
        for item in dataset['train']:
            if item['instruction'] and item['output']:
                processed_data.append({
                    "instruction": item['instruction'],
                    "response": item['output']
                })
        
        # Save to file
        output_path = "my_datasets/processed/self_instruct_instructions.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Self-Instruct dataset saved: {len(processed_data)} examples")
        return processed_data
        
    except Exception as e:
        print(f"❌ Error downloading Self-Instruct: {e}")
        return []

def merge_all_datasets():
    """Merge all downloaded datasets"""
    print("🔄 Merging all datasets...")
    
    all_data = []
    
    # Load all datasets
    dataset_files = [
        "my_datasets/processed/alpaca_instructions.json",
        "my_datasets/processed/dolly_instructions.json", 
        "my_datasets/processed/openassistant_instructions.json",
        "my_datasets/processed/self_instruct_instructions.json"
    ]
    
    for file_path in dataset_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
                    print(f"✅ Loaded {len(data)} examples from {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
    
    # Remove duplicates based on instruction
    seen_instructions = set()
    unique_data = []
    for item in all_data:
        instruction = item['instruction'].strip().lower()
        if instruction not in seen_instructions:
            seen_instructions.add(instruction)
            unique_data.append(item)
    
    print(f"📊 Total unique examples: {len(unique_data)}")
    
    # Save merged dataset
    output_path = "my_datasets/processed/merged_instruction_datasets.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Merged dataset saved: {output_path}")
    return unique_data

def create_high_quality_subset():
    """Create a high-quality subset for fine-tuning"""
    print("🎯 Creating high-quality subset...")
    
    # Load merged dataset
    merged_path = "my_datasets/processed/merged_instruction_datasets.json"
    if not os.path.exists(merged_path):
        print("❌ Merged dataset not found. Run merge_all_datasets() first.")
        return []
    
    with open(merged_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # Filter for high-quality examples
    high_quality_data = []
    
    for item in all_data:
        instruction = item['instruction']
        response = item['response']
        
        # Quality filters
        if (len(instruction) > 10 and len(instruction) < 200 and
            len(response) > 10 and len(response) < 500 and
            not any(word in instruction.lower() for word in ['nsfw', 'explicit', 'adult']) and
            not any(word in response.lower() for word in ['nsfw', 'explicit', 'adult'])):
            
            high_quality_data.append(item)
            
            # Limit to 2000 examples for fine-tuning
            if len(high_quality_data) >= 2000:
                break
    
    # Save high-quality subset
    output_path = "my_datasets/processed/high_quality_instructions.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(high_quality_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ High-quality subset saved: {len(high_quality_data)} examples")
    return high_quality_data

def main():
    """Main function to download and process all datasets"""
    print("🚀 Starting dataset download and processing...")
    print("="*60)
    
    # Download datasets
    alpaca_data = download_alpaca_dataset()
    dolly_data = download_dolly_dataset()
    openassistant_data = download_openassistant_dataset()
    self_instruct_data = download_self_instruct_dataset()
    
    # Merge all datasets
    merged_data = merge_all_datasets()
    
    # Create high-quality subset
    high_quality_data = create_high_quality_subset()
    
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY:")
    print(f"   Alpaca: {len(alpaca_data)} examples")
    print(f"   Dolly: {len(dolly_data)} examples")
    print(f"   OpenAssistant: {len(openassistant_data)} examples")
    print(f"   Self-Instruct: {len(self_instruct_data)} examples")
    print(f"   Total merged: {len(merged_data)} examples")
    print(f"   High-quality subset: {len(high_quality_data)} examples")
    print("="*60)
    
    print("\n✅ All datasets downloaded and processed!")
    print("📁 Files saved in: my_datasets/processed/")
    print("🎯 Use 'high_quality_instructions.json' for fine-tuning")

if __name__ == "__main__":
    main()
