#!/usr/bin/env python3
"""
zia_completion_mode.py
Use ZIA model in completion mode for better responses.
"""

import os
import argparse
import torch
from transformers import AutoTokenizer
from save_model_for_inference import load_student

def completion_generate(model, tokenizer, prompt, device, max_length=20, temperature=0.1):
    """Generate completion using greedy decoding"""
    
    # Tokenize input
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    
    model.eval()
    with torch.no_grad():
        generated = input_ids.clone()
        
        for _ in range(max_length):
            logits = model(generated)
            if isinstance(logits, tuple):
                logits = logits[0]
            
            # Greedy decoding (temperature=0)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            
            # Stop if we hit EOS or period
            if next_token.item() in [tokenizer.eos_token_id, 
                                   tokenizer.convert_tokens_to_ids('.'),
                                   tokenizer.convert_tokens_to_ids('!'),
                                   tokenizer.convert_tokens_to_ids('?')]:
                break
    
    # Decode and extract completion
    full_response = tokenizer.decode(generated[0], skip_special_tokens=True)
    completion = full_response[len(prompt):].strip()
    
    return completion

def chat_completion_mode(student, tokenizer, device="cpu"):
    """Chat in completion mode"""
    print("\\n=== ZIA Completion Mode (type 'exit' to quit) ===")
    print("Tip: Ask questions as completions, e.g., 'The capital of France is'")
    print()
    
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break
        
        # Convert question to completion format
        if user_input.endswith('?'):
            # Convert question to completion
            if "capital" in user_input.lower() and "france" in user_input.lower():
                prompt = "The capital of France is"
            elif "capital" in user_input.lower() and "india" in user_input.lower():
                prompt = "The capital of India is"
            elif "two plus two" in user_input.lower():
                prompt = "Two plus two equals"
            elif "sky" in user_input.lower():
                prompt = "The sky is"
            elif "water boils" in user_input.lower():
                prompt = "Water boils at"
            else:
                # Generic conversion
                prompt = user_input.replace('?', ' is')
        else:
            prompt = user_input
        
        # Generate completion
        completion = completion_generate(student, tokenizer, prompt, device)
        
        print(f"Zia: {completion}\\n")

def test_completion_mode():
    """Test various completion prompts"""
    
    print("🧪 TESTING COMPLETION MODE")
    print("="*50)
    
    # Load model
    model_path = "artifacts/zia_ift_v3_retrain/best_val/checkpoint.pt"
    tokenizer_path = "artifacts/zia_tokenizer_60k"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from: {model_path}")
    student, tokenizer = load_student(model_path, tokenizer_path, device)
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "The capital of India is", 
        "Two plus two equals",
        "The sky is",
        "Water boils at",
        "The largest planet is",
        "Shakespeare wrote",
        "The chemical symbol for gold is",
    ]
    
    for prompt in test_prompts:
        completion = completion_generate(student, tokenizer, prompt, device)
        print(f"'{prompt}' → '{completion}'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str,
                        default="artifacts/zia_ift_v3_retrain/best_val",
                        help="Path to the checkpoint directory.")
    parser.add_argument("--tokenizer_path", type=str, default="artifacts/zia_tokenizer_60k")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()
    
    try:
        # Find checkpoint
        ckpt_path = os.path.join(args.ckpt_dir, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            return
        
        print(f"Loading model from: {ckpt_path}")
        student, tokenizer = load_student(ckpt_path, args.tokenizer_path, args.device)
        
        print("Model loaded successfully!")
        print(f"Model info: {student.max_len} context length, {sum(p.numel() for p in student.parameters()):,} parameters")
        
        if args.test:
            test_completion_mode()
        else:
            chat_completion_mode(student, tokenizer, args.device)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
