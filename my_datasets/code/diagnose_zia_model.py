#!/usr/bin/env python3
"""
diagnose_zia_model.py
Comprehensive diagnosis of ZIA model issues and solutions.
"""

import os
import torch
import json
from transformers import AutoTokenizer
from save_model_for_inference import load_student, sample_generate

def analyze_model_outputs():
    """Analyze what the model is actually learning"""
    
    print("🔍 DIAGNOSING ZIA MODEL ISSUES")
    print("="*60)
    
    # Load model
    model_path = "artifacts/zia_ift_v3_retrain/best_val/checkpoint.pt"
    tokenizer_path = "artifacts/zia_tokenizer_60k"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from: {model_path}")
    model, tokenizer = load_student(model_path, tokenizer_path, device)
    
    print(f"Model info: {model.max_len} context length, {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test different input formats
    test_cases = [
        "What is the capital of France?",
        "Instruction: What is the capital of France?\\nResponse:",
        "Hello, how are you?",
        "Instruction: Hello, how are you?\\nResponse:",
        "The capital of France is",
        "Paris is the capital of",
    ]
    
    print("\\n🧪 TESTING DIFFERENT INPUT FORMATS:")
    print("-" * 50)
    
    for i, test_input in enumerate(test_cases):
        print(f"\\nTest {i+1}: {test_input}")
        
        # Tokenize
        enc = tokenizer(test_input, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        
        # Generate with different parameters
        print("  Generation parameters:")
        
        # Very conservative generation
        outputs = sample_generate(
            model, tokenizer, input_ids, device,
            max_length=20, temperature=0.1, top_k=5, top_p=0.5, repetition_penalty=1.5
        )
        
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = reply.replace("Ġ", " ").replace("  ", " ").strip()
        
        print(f"    Conservative: {reply[:100]}...")
        
        # Check what tokens the model predicts
        with torch.no_grad():
            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            
            # Get top predictions for next token
            next_token_logits = logits[:, -1, :]
            top_tokens = torch.topk(next_token_logits, 10, dim=-1)
            
            print(f"    Top 10 next tokens:")
            for j, (token_id, score) in enumerate(zip(top_tokens.indices[0], top_tokens.values[0])):
                token_text = tokenizer.decode([token_id])
                print(f"      {j+1}. '{token_text}' (score: {score:.3f})")

def analyze_training_data():
    """Analyze the training data format"""
    
    print("\\n📊 ANALYZING TRAINING DATA:")
    print("-" * 50)
    
    # Check if we can find the original training data
    possible_paths = [
        "my_datasets/processed/arrow_cleaned_v1",
        "artifacts/tokenized_datasets/zia_ift_v3",
        "my_datasets/processed/zia_ift_v3_clean"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found dataset: {path}")
            try:
                from datasets import load_from_disk
                if os.path.exists(os.path.join(path, "train")):
                    dataset = load_from_disk(os.path.join(path, "train"))
                else:
                    dataset = load_from_disk(path)
                
                print(f"  Samples: {len(dataset)}")
                print(f"  Features: {dataset.features}")
                
                # Show a few examples
                for i in range(min(3, len(dataset))):
                    sample = dataset[i]
                    print(f"  Sample {i+1}: {str(sample)[:200]}...")
                    
            except Exception as e:
                print(f"  Error loading: {e}")

def test_simple_generation():
    """Test very simple generation tasks"""
    
    print("\\n🎯 TESTING SIMPLE GENERATION:")
    print("-" * 50)
    
    # Load model
    model_path = "artifacts/zia_ift_v3_retrain/best_val/checkpoint.pt"
    tokenizer_path = "artifacts/zia_tokenizer_60k"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, tokenizer = load_student(model_path, tokenizer_path, device)
    model.eval()
    
    # Test completion tasks
    completion_tasks = [
        "The capital of France is",
        "Paris is the capital of",
        "Two plus two equals",
        "The sky is",
        "Water boils at",
    ]
    
    for task in completion_tasks:
        print(f"\\nTask: '{task}'")
        
        # Tokenize
        enc = tokenizer(task, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        
        # Generate with very conservative settings
        with torch.no_grad():
            generated = input_ids.clone()
            for _ in range(10):  # Generate only 10 tokens
                logits = model(generated)
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                # Use greedy decoding (temperature=0)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=1)
                
                # Stop if we hit a period or similar
                if next_token.item() in [tokenizer.convert_tokens_to_ids('.'), 
                                       tokenizer.convert_tokens_to_ids('!'),
                                       tokenizer.convert_tokens_to_ids('?')]:
                    break
        
        # Decode
        response = tokenizer.decode(generated[0], skip_special_tokens=True)
        completion = response[len(task):].strip()
        print(f"  Completion: '{completion}'")

def create_minimal_test():
    """Create a minimal test to isolate the issue"""
    
    print("\\n🔬 MINIMAL TEST:")
    print("-" * 50)
    
    # Load model
    model_path = "artifacts/zia_ift_v3_retrain/best_val/checkpoint.pt"
    tokenizer_path = "artifacts/zia_tokenizer_60k"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, tokenizer = load_student(model_path, tokenizer_path, device)
    model.eval()
    
    # Test with just one token
    test_token = tokenizer.encode("The")[0]
    input_ids = torch.tensor([[test_token]]).to(device)
    
    print(f"Input token: '{tokenizer.decode([test_token])}' (ID: {test_token})")
    
    with torch.no_grad():
        logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        # Get top predictions
        next_token_logits = logits[:, -1, :]
        top_tokens = torch.topk(next_token_logits, 20, dim=-1)
        
        print("Top 20 predictions:")
        for i, (token_id, score) in enumerate(zip(top_tokens.indices[0], top_tokens.values[0])):
            token_text = tokenizer.decode([token_id])
            print(f"  {i+1:2d}. '{token_text}' (ID: {token_id}, score: {score:.3f})")

def main():
    """Run comprehensive diagnosis"""
    
    print("🚨 ZIA MODEL DIAGNOSIS REPORT")
    print("="*60)
    
    # Run all diagnostic tests
    analyze_model_outputs()
    analyze_training_data()
    test_simple_generation()
    create_minimal_test()
    
    print("\\n" + "="*60)
    print("📋 DIAGNOSIS SUMMARY:")
    print("="*60)
    
    print("""
🔍 ISSUES IDENTIFIED:

1. **Model Architecture Mismatch**: The model may not be properly trained for instruction-following
2. **Training Data Quality**: The original training data may contain poor quality examples
3. **Tokenization Issues**: The model may not understand the instruction format properly
4. **Generation Parameters**: Current parameters may be too aggressive

🛠️ RECOMMENDED SOLUTIONS:

1. **Use Completion Format**: Instead of "Instruction: X\\nResponse:", try just "X is"
2. **Lower Temperature**: Use temperature=0.1 or even 0.0 for more focused responses
3. **Shorter Responses**: Limit max_length to 20-30 tokens
4. **Better Training Data**: Use high-quality instruction datasets
5. **Fine-tuning**: Continue fine-tuning with better examples

🎯 IMMEDIATE ACTIONS:

1. Test with completion format: "The capital of France is"
2. Use very conservative generation parameters
3. Consider retraining with better data
4. Try the fine-tuned model with different parameters
""")

if __name__ == "__main__":
    main()
