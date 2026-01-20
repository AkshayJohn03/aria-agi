#!/usr/bin/env python3
"""
evaluate_zia_models.py
Comprehensive evaluation script for ZIA models.
"""

import os
import json
import torch
from transformers import AutoTokenizer
from my_datasets.code.save_model_for_inference import load_student, sample_generate

def evaluate_model(ckpt_path, tokenizer_path, device="cuda"):
    """Evaluate a model with standardized prompts"""
    
    print(f"\n{'='*60}")
    print(f"Evaluating model: {ckpt_path}")
    print(f"{'='*60}")
    
    try:
        # Load model
        model, tokenizer = load_student(ckpt_path, tokenizer_path, device)
        
        # Test prompts
        test_prompts = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Explain photosynthesis in simple terms.",
            "Write a short poem about the ocean.",
            "What are the benefits of exercise?",
            "How do computers work?",
            "Tell me a joke.",
            "What is artificial intelligence?",
            "Describe the process of making bread.",
            "What is the meaning of life?"
        ]
        
        print(f"\nModel loaded successfully!")
        print(f"Architecture: {model.max_len} context length, {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test each prompt
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}/10 ---")
            print(f"Prompt: {prompt}")
            
            # Encode input
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = enc["input_ids"]
            
            # Generate response
            with torch.no_grad():
                outputs = sample_generate(
                    model, tokenizer, input_ids, device,
                    max_length=150, temperature=0.7, top_k=40, top_p=0.85, repetition_penalty=1.1
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace("Ġ", " ").replace("  ", " ").strip()
            
            # Remove the input prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            print(f"Response: {response}")
            
            # Simple quality assessment
            quality_score = assess_response_quality(prompt, response)
            print(f"Quality Score: {quality_score}/10")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def assess_response_quality(prompt, response):
    """Simple quality assessment based on response characteristics"""
    score = 0
    
    # Length check (not too short, not too long)
    if 10 <= len(response) <= 200:
        score += 2
    elif 5 <= len(response) <= 300:
        score += 1
    
    # Coherence checks
    if not any(word in response.lower() for word in ["Ġ", "townspeople", "hotel", "wine", "collagen"]):
        score += 2  # Avoid gibberish patterns
    
    if any(word in response.lower() for word in ["the", "and", "is", "are", "of", "in", "to"]):
        score += 1  # Contains common words
    
    # Relevance check
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    if len(prompt_words.intersection(response_words)) > 0:
        score += 2  # Some word overlap
    
    # Completeness check
    if response.endswith(('.', '!', '?')):
        score += 1  # Proper ending
    
    # Avoid repetition
    words = response.split()
    if len(set(words)) / max(len(words), 1) > 0.7:
        score += 2  # Good vocabulary diversity
    
    return min(score, 10)

def main():
    """Main evaluation function"""
    
    # Model paths to test
    models_to_test = [
        {
            "name": "ZIA Dense Base (256 context)",
            "ckpt": "artifacts/zia_dense_base/recovered_best.pt",
            "tokenizer": "artifacts/zia_tokenizer_60k"
        },
        {
            "name": "ZIA IFT V3 Retrain (512 context)",
            "ckpt": "artifacts/zia_ift_v3_retrain/best_val/checkpoint.pt",
            "tokenizer": "artifacts/zia_tokenizer_60k"
        },
        {
            "name": "ZIA Dense Runs Best",
            "ckpt": "artifacts/zia_dense_runs/best_val/checkpoint.pt",
            "tokenizer": "artifacts/zia_tokenizer_60k"
        }
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    results = {}
    
    for model_info in models_to_test:
        print(f"\n{'='*80}")
        print(f"Testing: {model_info['name']}")
        print(f"{'='*80}")
        
        if os.path.exists(model_info['ckpt']):
            success = evaluate_model(
                model_info['ckpt'], 
                model_info['tokenizer'], 
                device
            )
            results[model_info['name']] = {
                "status": "success" if success else "failed",
                "path": model_info['ckpt']
            }
        else:
            print(f"❌ Checkpoint not found: {model_info['ckpt']}")
            results[model_info['name']] = {
                "status": "not_found",
                "path": model_info['ckpt']
            }
    
    # Save results
    with open("model_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    for name, result in results.items():
        status = result['status']
        if status == "success":
            print(f"✅ {name}: Successfully evaluated")
        elif status == "failed":
            print(f"❌ {name}: Evaluation failed")
        else:
            print(f"⚠️  {name}: Checkpoint not found")
    
    print(f"\nDetailed results saved to: model_evaluation_results.json")

if __name__ == "__main__":
    main()
