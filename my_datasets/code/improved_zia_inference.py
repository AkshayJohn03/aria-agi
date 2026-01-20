#!/usr/bin/env python3
"""
improved_zia_inference.py
Improved inference script with correct input formatting for ZIA model.
"""

import os
import argparse
import torch
from transformers import AutoTokenizer
from save_model_for_inference import load_student, sample_generate

def format_input_for_model(user_input):
    """Format user input in the way the model was trained on"""
    return f"Instruction: {user_input}\\nResponse:"

def chat_loop(student, tokenizer, device="cpu"):
    print("\\n=== Chat with ZIA (type 'exit' to quit) ===")
    print("Tip: Ask questions naturally - the model will respond appropriately!")
    print()
    
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break

        # Format input correctly for the model
        formatted_input = format_input_for_model(user_input)
        
        enc = tokenizer(formatted_input, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]

        outputs = sample_generate(
            student, tokenizer, input_ids, device,
            max_length=100, temperature=0.3, top_k=20, top_p=0.7, repetition_penalty=1.3
        )

        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = reply.replace("Ġ", " ").replace("  ", " ").strip()
        
        # Extract just the response part
        if "Response:" in reply:
            reply = reply.split("Response:")[-1].strip()
        
        print(f"Zia: {reply}\\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str,
                        default="artifacts/zia_ift_v4_longctx/checkpoints",
                        help="Path to the checkpoint directory.")
    parser.add_argument("--tokenizer_path", type=str, default="artifacts/zia_tokenizer_60k")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    try:
        # Find checkpoint
        ckpt_path = os.path.join(args.ckpt_dir, "best_val.pt")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            return
        
        print(f"Loading model from: {ckpt_path}")
        student, tokenizer = load_student(ckpt_path, args.tokenizer_path, args.device)
        
        print("Model loaded successfully!")
        print(f"Model info: {student.max_len} context length, {sum(p.numel() for p in student.parameters()):,} parameters")
        
        chat_loop(student, tokenizer, args.device)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
