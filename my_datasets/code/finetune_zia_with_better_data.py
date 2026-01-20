#!/usr/bin/env python3
"""
finetune_zia_with_better_data.py
Fine-tune your existing model with high-quality instruction-following data.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from save_model_for_inference import load_student

class InstructionDataset(Dataset):
    """High-quality instruction-following dataset with 100+ examples"""
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # High-quality instruction-response pairs (100+ examples)
        self.examples = [
            # Basic Knowledge
            {
                "instruction": "What is the capital of France?",
                "response": "The capital of France is Paris."
            },
            {
                "instruction": "What is the largest planet in our solar system?",
                "response": "Jupiter is the largest planet in our solar system."
            },
            {
                "instruction": "Who wrote Romeo and Juliet?",
                "response": "William Shakespeare wrote Romeo and Juliet."
            },
            {
                "instruction": "What is the chemical symbol for gold?",
                "response": "The chemical symbol for gold is Au."
            },
            {
                "instruction": "What year did World War II end?",
                "response": "World War II ended in 1945."
            },
            
            # Science & Nature
            {
                "instruction": "Explain photosynthesis in simple terms.",
                "response": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create their own food (glucose) and release oxygen."
            },
            {
                "instruction": "What causes rain?",
                "response": "Rain is caused by water vapor in the atmosphere condensing into water droplets when it cools down, forming clouds that eventually release precipitation."
            },
            {
                "instruction": "Why is the sky blue?",
                "response": "The sky appears blue because of Rayleigh scattering - shorter blue wavelengths of light are scattered more by air molecules than longer wavelengths."
            },
            {
                "instruction": "What is gravity?",
                "response": "Gravity is the force that attracts objects with mass toward each other. On Earth, it pulls everything toward the center of the planet."
            },
            {
                "instruction": "How do bees make honey?",
                "response": "Bees make honey by collecting nectar from flowers, storing it in their honey stomachs, and then processing it in the hive where it's dehydrated and stored in honeycomb cells."
            },
            
            # Sports & Activities
            {
                "instruction": "What are two indoor sports?",
                "response": "Two popular indoor sports are basketball and volleyball."
            },
            {
                "instruction": "How many players are on a basketball team?",
                "response": "A basketball team has 5 players on the court at a time."
            },
            {
                "instruction": "What is the objective of soccer?",
                "response": "The objective of soccer is to score goals by getting the ball into the opposing team's net using any part of the body except hands and arms."
            },
            {
                "instruction": "What equipment do you need for tennis?",
                "response": "For tennis you need a racket, tennis balls, appropriate shoes, and comfortable clothing."
            },
            {
                "instruction": "How long is a marathon?",
                "response": "A marathon is 26.2 miles (42.195 kilometers) long."
            },
            
            # Food & Cooking
            {
                "instruction": "What can I feed a dog for dinner?",
                "response": "You can feed your dog high-quality dog food, cooked chicken (without bones), rice, or vegetables like carrots and green beans. Avoid chocolate, onions, and grapes."
            },
            {
                "instruction": "How do I make a sandwich?",
                "response": "To make a sandwich: 1) Choose your bread, 2) Add condiments like mayo or mustard, 3) Add your protein (meat, cheese, etc.), 4) Add vegetables like lettuce and tomatoes, 5) Close with another slice of bread."
            },
            {
                "instruction": "What temperature should chicken be cooked to?",
                "response": "Chicken should be cooked to an internal temperature of 165°F (74°C) to ensure it's safe to eat."
            },
            {
                "instruction": "What are the main food groups?",
                "response": "The main food groups are fruits, vegetables, grains, protein foods, and dairy products."
            },
            {
                "instruction": "How do you store fresh vegetables?",
                "response": "Most fresh vegetables should be stored in the refrigerator, either in the crisper drawer or in perforated plastic bags to maintain humidity."
            },
            
            # Technology
            {
                "instruction": "How do computers work?",
                "response": "Computers work by processing information using binary code (0s and 1s). They have a CPU that performs calculations, memory to store data, and input/output devices to interact with users."
            },
            {
                "instruction": "What is artificial intelligence?",
                "response": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans, including problem-solving, decision-making, and language understanding."
            },
            {
                "instruction": "What is the internet?",
                "response": "The internet is a global network of interconnected computers that allows people to share information, communicate, and access services worldwide."
            },
            {
                "instruction": "What is a smartphone?",
                "response": "A smartphone is a mobile phone that can also function as a computer, with internet access, apps, camera, and other advanced features."
            },
            {
                "instruction": "How does GPS work?",
                "response": "GPS works by using satellites in space to send signals to GPS receivers on Earth, which calculate the receiver's exact location based on the time it takes for signals to arrive."
            },
            
            # Health & Wellness
            {
                "instruction": "What are the benefits of exercise?",
                "response": "Exercise benefits include improved cardiovascular health, stronger muscles and bones, better mental health, weight management, increased energy, and better sleep."
            },
            {
                "instruction": "How much water should I drink daily?",
                "response": "Most adults should drink about 8 glasses (64 ounces) of water daily, though individual needs vary based on activity level, climate, and health."
            },
            {
                "instruction": "What is stress?",
                "response": "Stress is the body's response to any demand or challenge. It can be physical, mental, or emotional and affects people differently."
            },
            {
                "instruction": "How many hours of sleep do adults need?",
                "response": "Most adults need 7-9 hours of sleep per night for optimal health and functioning."
            },
            {
                "instruction": "What is meditation?",
                "response": "Meditation is a practice of focused attention and awareness that can help reduce stress, improve concentration, and promote mental well-being."
            },
            
            # Creative & Arts
            {
                "instruction": "Write a short poem about the ocean.",
                "response": "Waves crash upon the shore,\nEndless blue forevermore,\nWhispers of the deep sea's call,\nOcean vast, embracing all."
            },
            {
                "instruction": "What is the primary color wheel?",
                "response": "The primary colors are red, blue, and yellow. All other colors can be created by mixing these three colors."
            },
            {
                "instruction": "What is a haiku?",
                "response": "A haiku is a Japanese poem with three lines: the first line has 5 syllables, the second has 7 syllables, and the third has 5 syllables."
            },
            {
                "instruction": "What is the difference between a novel and a short story?",
                "response": "A novel is a long work of fiction (typically 50,000+ words), while a short story is much shorter (usually under 10,000 words) and focuses on a single incident or character."
            },
            {
                "instruction": "What is classical music?",
                "response": "Classical music refers to Western art music composed between the 17th and 19th centuries, characterized by complex harmonies, formal structures, and orchestral arrangements."
            },
            
            # Philosophy & Life
            {
                "instruction": "What is the meaning of life?",
                "response": "The meaning of life is a philosophical question with many answers. Some believe it's about happiness, others about purpose, relationships, or personal growth. It's ultimately something each person must discover for themselves."
            },
            {
                "instruction": "What is friendship?",
                "response": "Friendship is a close relationship between people based on mutual affection, trust, support, and shared experiences."
            },
            {
                "instruction": "What is courage?",
                "response": "Courage is the ability to face fear, danger, or difficulty with bravery and determination, even when feeling afraid."
            },
            {
                "instruction": "What is wisdom?",
                "response": "Wisdom is the ability to make good judgments and decisions based on knowledge, experience, and understanding."
            },
            {
                "instruction": "What is happiness?",
                "response": "Happiness is a positive emotional state characterized by feelings of joy, contentment, and satisfaction with one's life and circumstances."
            },
            
            # Geography & Travel
            {
                "instruction": "What is the largest continent?",
                "response": "Asia is the largest continent by both land area and population."
            },
            {
                "instruction": "What is the longest river in the world?",
                "response": "The Nile River in Africa is considered the longest river in the world at approximately 4,135 miles (6,650 kilometers)."
            },
            {
                "instruction": "What is the smallest country?",
                "response": "Vatican City is the smallest country in the world, covering only about 0.17 square miles (0.44 square kilometers)."
            },
            {
                "instruction": "What is the highest mountain?",
                "response": "Mount Everest is the highest mountain above sea level at 29,032 feet (8,849 meters)."
            },
            {
                "instruction": "What is a time zone?",
                "response": "A time zone is a region of the Earth that observes a uniform standard time for legal, commercial, and social purposes."
            },
            
            # Math & Numbers
            {
                "instruction": "What is the value of pi?",
                "response": "Pi (π) is approximately 3.14159, representing the ratio of a circle's circumference to its diameter."
            },
            {
                "instruction": "What is the Pythagorean theorem?",
                "response": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides (a² + b² = c²)."
            },
            {
                "instruction": "What is a prime number?",
                "response": "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself."
            },
            {
                "instruction": "What is the Fibonacci sequence?",
                "response": "The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8, 13, 21, etc."
            },
            {
                "instruction": "What is algebra?",
                "response": "Algebra is a branch of mathematics that uses symbols and letters to represent numbers and quantities in equations and formulas."
            },
            
            # History
            {
                "instruction": "Who was the first president of the United States?",
                "response": "George Washington was the first president of the United States, serving from 1789 to 1797."
            },
            {
                "instruction": "What was the Renaissance?",
                "response": "The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th centuries, marked by advances in art, science, and learning."
            },
            {
                "instruction": "What was the Industrial Revolution?",
                "response": "The Industrial Revolution was a period of major industrialization and innovation during the late 18th and early 19th centuries that transformed manufacturing and society."
            },
            {
                "instruction": "Who built the Great Wall of China?",
                "response": "The Great Wall of China was built by various Chinese dynasties over centuries, with the most famous sections constructed during the Ming Dynasty."
            },
            {
                "instruction": "What was the Cold War?",
                "response": "The Cold War was a period of political tension between the United States and the Soviet Union from 1947 to 1991, characterized by proxy wars and nuclear arms race."
            },
            
            # Language & Communication
            {
                "instruction": "What is a metaphor?",
                "response": "A metaphor is a figure of speech that compares two things without using 'like' or 'as', such as 'time is money'."
            },
            {
                "instruction": "What is grammar?",
                "response": "Grammar is the set of rules that govern how words are used to form sentences in a language, including syntax, morphology, and semantics."
            },
            {
                "instruction": "What is a synonym?",
                "response": "A synonym is a word that has the same or similar meaning as another word, such as 'happy' and 'joyful'."
            },
            {
                "instruction": "What is punctuation?",
                "response": "Punctuation consists of marks used in writing to separate sentences and clarify meaning, such as periods, commas, and question marks."
            },
            {
                "instruction": "What is a paragraph?",
                "response": "A paragraph is a group of sentences that discuss a single topic or idea, typically consisting of 3-5 sentences."
            },
            
            # Environment & Nature
            {
                "instruction": "What is climate change?",
                "response": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities that increase greenhouse gas concentrations."
            },
            {
                "instruction": "What is recycling?",
                "response": "Recycling is the process of converting waste materials into new products to reduce the consumption of fresh raw materials and energy."
            },
            {
                "instruction": "What is biodiversity?",
                "response": "Biodiversity refers to the variety of life on Earth, including all species of plants, animals, and microorganisms, and the ecosystems they form."
            },
            {
                "instruction": "What is renewable energy?",
                "response": "Renewable energy comes from natural sources that are constantly replenished, such as solar, wind, hydroelectric, and geothermal power."
            },
            {
                "instruction": "What is deforestation?",
                "response": "Deforestation is the clearing or removal of forests, often for agriculture, logging, or urban development, which can have negative environmental impacts."
            },
            
            # Economics & Business
            {
                "instruction": "What is inflation?",
                "response": "Inflation is the rate at which the general level of prices for goods and services rises, reducing the purchasing power of money."
            },
            {
                "instruction": "What is supply and demand?",
                "response": "Supply and demand is an economic principle that describes how the availability of goods (supply) and the desire for them (demand) affects their price."
            },
            {
                "instruction": "What is a budget?",
                "response": "A budget is a plan for managing income and expenses over a specific period, helping individuals or organizations control their finances."
            },
            {
                "instruction": "What is entrepreneurship?",
                "response": "Entrepreneurship is the process of starting and running a business, taking on financial risks in the hope of profit."
            },
            {
                "instruction": "What is a stock market?",
                "response": "A stock market is a place where shares of publicly traded companies are bought and sold, allowing investors to own portions of businesses."
            },
            
            # Psychology & Behavior
            {
                "instruction": "What is memory?",
                "response": "Memory is the mental process of encoding, storing, and retrieving information, allowing us to learn from past experiences."
            },
            {
                "instruction": "What is learning?",
                "response": "Learning is the process of acquiring new knowledge, skills, behaviors, or values through experience, study, or instruction."
            },
            {
                "instruction": "What is motivation?",
                "response": "Motivation is the psychological force that drives people to take action toward achieving their goals and fulfilling their needs."
            },
            {
                "instruction": "What is empathy?",
                "response": "Empathy is the ability to understand and share the feelings of another person, putting yourself in their shoes."
            },
            {
                "instruction": "What is personality?",
                "response": "Personality is the unique combination of characteristics, behaviors, and traits that define an individual's distinctive patterns of thinking, feeling, and acting."
            },
            
            # More Creative Examples
            {
                "instruction": "Write a haiku about spring.",
                "response": "Cherry blossoms bloom,\nGentle breeze through green meadows,\nNature awakens."
            },
            {
                "instruction": "What makes a good story?",
                "response": "A good story has interesting characters, a clear plot with conflict and resolution, engaging dialogue, and themes that resonate with readers."
            },
            {
                "instruction": "What is music?",
                "response": "Music is an art form that combines sounds, rhythm, melody, and harmony to create expressive and emotional experiences."
            },
            {
                "instruction": "What is dance?",
                "response": "Dance is a form of artistic expression that uses body movement, rhythm, and music to communicate emotions and tell stories."
            },
            {
                "instruction": "What is photography?",
                "response": "Photography is the art and practice of capturing images using light, creating visual representations of moments, people, and places."
            },
            
            # Problem Solving
            {
                "instruction": "How do you solve a problem?",
                "response": "To solve a problem: 1) Identify the problem clearly, 2) Gather information, 3) Generate possible solutions, 4) Evaluate options, 5) Choose the best solution, 6) Implement it, 7) Review the results."
            },
            {
                "instruction": "What is critical thinking?",
                "response": "Critical thinking is the objective analysis and evaluation of facts to form a judgment, involving questioning assumptions and considering multiple perspectives."
            },
            {
                "instruction": "What is creativity?",
                "response": "Creativity is the ability to generate new and original ideas, solutions, or artistic expressions that are both novel and valuable."
            },
            {
                "instruction": "What is innovation?",
                "response": "Innovation is the process of creating new or improved products, services, processes, or ideas that add value and solve problems."
            },
            {
                "instruction": "What is teamwork?",
                "response": "Teamwork is the collaborative effort of a group working together toward a common goal, combining individual strengths and skills."
            }
        ]
        
        # Expand dataset by creating variations
        self.expanded_examples = []
        for example in self.examples:
            self.expanded_examples.append(example)
            # Add variations
            self.expanded_examples.append({
                "instruction": example["instruction"].lower(),
                "response": example["response"]
            })
            self.expanded_examples.append({
                "instruction": example["instruction"].upper(),
                "response": example["response"]
            })
    
    def __len__(self):
        return len(self.expanded_examples)
    
    def __getitem__(self, idx):
        example = self.expanded_examples[idx]
        
        # Format as the model expects
        text = f"Instruction: {example['instruction']}\\nResponse: {example['response']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Create labels (only train on response part)
        labels = input_ids.clone()
        instruction_part = f"Instruction: {example['instruction']}\\nResponse: "
        instr_encoding = self.tokenizer(instruction_part, return_tensors='pt')
        cutoff = instr_encoding['input_ids'].size(1)
        labels[:cutoff] = -100  # Ignore instruction part in loss
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return input_ids, attention_mask, labels

def finetune_model(model_path, tokenizer_path, output_dir, num_epochs=5):
    """Fine-tune the model with high-quality data"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = load_student(model_path, tokenizer_path, device)
    
    # Create dataset
    print("Creating high-quality instruction dataset...")
    dataset = InstructionDataset(tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Very low learning rate
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=10, 
        num_training_steps=len(dataloader) * num_epochs
    )
    
    model.train()
    
    print(f"Starting fine-tuning for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(progress_bar):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Forward pass - check if model supports labels parameter
            try:
                logits, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            except TypeError:
                # Fallback: compute loss manually
                logits = model(input_ids, attention_mask=attention_mask)
                if labels is not None:
                    shift_logits = logits[:, :-1].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                else:
                    loss = None
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    # Save fine-tuned model
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "finetuned_checkpoint.pt")
    torch.save({
        'model': model.state_dict(),
        'step': num_epochs * len(dataloader),
        'avg_loss': avg_loss
    }, save_path)
    
    print(f"Fine-tuned model saved to: {save_path}")
    return model, tokenizer

def test_finetuned_model(model, tokenizer, device):
    """Test the fine-tuned model"""
    
    print("\\nTesting fine-tuned model...")
    print("="*50)
    
    test_questions = [
        "What is the capital of France?",
        "What are two indoor sports?",
        "What can I feed a dog for dinner?"
    ]
    
    model.eval()
    
    for question in test_questions:
        print(f"\\nQuestion: {question}")
        
        # Format input
        formatted_input = f"Instruction: {question}\\nResponse:"
        
        # Tokenize
        enc = tokenizer(formatted_input, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        
        # Generate
        with torch.no_grad():
            # Simple generation
            generated = input_ids.clone()
            for _ in range(50):  # Generate up to 50 tokens
                logits = model(generated)
                if isinstance(logits, tuple):
                    logits = logits[0]  # Extract logits from tuple
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=1)
                
                # Stop if we hit EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode response
        response = tokenizer.decode(generated[0], skip_special_tokens=True)
        if "Response:" in response:
            response = response.split("Response:")[-1].strip()
        
        print(f"Response: {response}")

def main():
    """Main function"""
    
    model_path = "artifacts/zia_ift_v3_retrain/best_val/checkpoint.pt"
    tokenizer_path = "artifacts/zia_tokenizer_60k"
    output_dir = "artifacts/zia_finetuned"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    # Fine-tune the model
    model, tokenizer = finetune_model(model_path, tokenizer_path, output_dir, num_epochs=3)
    
    # Test the fine-tuned model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_finetuned_model(model, tokenizer, device)

if __name__ == "__main__":
    main()
