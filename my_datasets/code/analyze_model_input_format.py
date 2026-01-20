#!/usr/bin/env python3
"""
analyze_model_input_format.py
Comprehensive analysis of ZIA model's expected input format based on training data.
"""

import os
import json
import torch
from transformers import AutoTokenizer

def analyze_ift_dataset_format():
    """Analyze the IFT dataset format that your model was trained on"""
    
    print("🔍 Analyzing IFT Dataset Format")
    print("="*80)
    
    # The key insight from retrain_zia_IFT.py
    print("📋 Training Data Format Analysis:")
    print("Based on retrain_zia_IFT.py and train_zia_ift_v3_rope_alibi.py:")
    print()
    print("🎯 EXPECTED INPUT FORMAT:")
    print("   Template: 'Instruction: {instruction}\\nResponse: {response}'")
    print()
    print("📝 Training Process:")
    print("   1. Raw data formatted as: 'Instruction: {question}\\nResponse: {answer}'")
    print("   2. Tokenized with 512 max length")
    print("   3. Labels masked for instruction part (only response part trained)")
    print("   4. Model learns to continue after 'Instruction: {question}\\nResponse:'")
    print()
    
    return "Instruction: {instruction}\\nResponse: {response}"

def test_model_with_correct_format(model_path, tokenizer_path, device="cuda"):
    """Test the model with the correct input format"""
    
    print("🤖 Testing Model with Correct Input Format")
    print("="*80)
    
    try:
        # Import model loading
        import sys
        sys.path.append("my_datasets/code")
        from save_model_for_inference import load_student, sample_generate
        
        # Load model
        model, tokenizer = load_student(model_path, tokenizer_path, device)
        
        # Test questions with correct format
        test_questions = [
            "What is the capital of France?",
            "Explain photosynthesis in simple terms.",
            "Write a short poem about the ocean.",
            "What are the benefits of exercise?",
            "How do computers work?"
        ]
        
        print("Testing with CORRECT format: 'Instruction: {question}\\nResponse:'")
        print()
        
        for i, question in enumerate(test_questions, 1):
            print(f"--- Test {i} ---")
            
            # CORRECT format - what the model was trained on
            correct_prompt = f"Instruction: {question}\\nResponse:"
            
            print(f"Question: {question}")
            print(f"Input format: {correct_prompt}")
            
            # Encode and generate
            enc = tokenizer(correct_prompt, return_tensors="pt").to(device)
            input_ids = enc["input_ids"]
            
            with torch.no_grad():
                outputs = sample_generate(
                    model, tokenizer, input_ids, device,
                    max_length=150, temperature=0.7, top_k=40, top_p=0.85
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace("Ġ", " ").replace("  ", " ").strip()
            
            # Extract just the response part
            if "Response:" in response:
                response = response.split("Response:")[-1].strip()
            
            print(f"Response: {response}")
            print("-" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def create_improved_inference_script():
    """Create an improved inference script with correct formatting"""
    
    print("\\n🛠️ Creating Improved Inference Script")
    print("="*80)
    
    script_content = '''#!/usr/bin/env python3
"""
improved_zia_inference.py
Improved inference script with correct input formatting for ZIA model.
"""

import os
import argparse
import torch
from transformers import AutoTokenizer
from my_datasets.code.save_model_for_inference import load_student, sample_generate

def format_input_for_model(user_input):
    """Format user input in the way the model was trained on"""
    return f"Instruction: {user_input}\\nResponse:"

def chat_loop(student, tokenizer, device="cpu"):
    print("\\n=== Chat with ZIA (type 'exit' to quit) ===")
    print("💡 Tip: Ask questions naturally - the model will respond appropriately!")
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
            max_length=200, temperature=0.7, top_k=40, top_p=0.85, repetition_penalty=1.1
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
                        default="artifacts/zia_ift_v3_retrain/best_val",
                        help="Path to the checkpoint directory.")
    parser.add_argument("--tokenizer_path", type=str, default="artifacts/zia_tokenizer_60k")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    try:
        # Find checkpoint
        ckpt_path = os.path.join(args.ckpt_dir, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            print(f"❌ Checkpoint not found: {ckpt_path}")
            return
        
        print(f"🔄 Loading model from: {ckpt_path}")
        student, tokenizer = load_student(ckpt_path, args.tokenizer_path, args.device)
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model info: {student.max_len} context length, {sum(p.numel() for p in student.parameters()):,} parameters")
        
        chat_loop(student, tokenizer, args.device)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open("my_datasets/code/improved_zia_inference.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created improved_zia_inference.py")
    print("   This script formats inputs correctly for your model!")

def main():
    """Main analysis function"""
    
    print("🔍 ZIA Model Input Format Analysis")
    print("="*80)
    
    # Analyze the training format
    template = analyze_ift_dataset_format()
    
    # Test with correct format
    model_paths = [
        "artifacts/zia_ift_v3_retrain/best_val/checkpoint.pt",
        "artifacts/zia_dense_base/recovered_best.pt",
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"\\nTesting model: {model_path}")
            test_model_with_correct_format(model_path, "artifacts/zia_tokenizer_60k", device)
        else:
            print(f"⚠️ Model not found: {model_path}")
    
    # Create improved inference script
    create_improved_inference_script()
    
    print(f"\\n{'='*80}")
    print("🎯 KEY FINDINGS & RECOMMENDATIONS")
    print(f"{'='*80}")
    print()
    print("✅ CORRECT INPUT FORMAT:")
    print("   Use: 'Instruction: {your_question}\\nResponse:'")
    print()
    print("❌ WRONG FORMATS (causing gibberish):")
    print("   - Just the question: 'What is the capital of France?'")
    print("   - Human/Assistant format: 'Human: What is...\\nAssistant:'")
    print("   - Other formats not matching training data")
    print()
    print("🔧 SOLUTION:")
    print("   1. Use the improved_zia_inference.py script")
    print("   2. Or format your inputs as: 'Instruction: {question}\\nResponse:'")
    print("   3. The model was trained to continue after this exact pattern")
    print()
    print("📈 EXPECTED IMPROVEMENT:")
    print("   - Much more coherent responses")
    print("   - Better instruction following")
    print("   - Proper conversational flow")

if __name__ == "__main__":
    main()
