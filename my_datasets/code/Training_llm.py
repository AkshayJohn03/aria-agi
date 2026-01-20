import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig

# === Config ===
CLEAN_DIR = "./datasets/cleaned"
MODEL_NAME = "gpt2"  # change later to bigger base
OUTPUT_DIR = "./checkpoints"
CTX = 2048
BATCH_SIZE = 1
LR = 3e-4
EPOCHS = 1

# === Dataset Loader ===
class JsonlDataset(Dataset):
    def __init__(self, file_list, tokenizer, ctx_len=CTX):
        self.data = []
        self.tokenizer = tokenizer
        for path in file_list:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        txt = json.loads(line)["text"]
                        tokenized = tokenizer(txt, truncation=True, max_length=ctx_len)
                        self.data.append(tokenized)
                    except:
                        continue
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {"input_ids": torch.tensor(item["input_ids"]),
                "attention_mask": torch.tensor(item["attention_mask"]),
                "labels": torch.tensor(item["input_ids"])}

# === BitsAndBytes (8-bit) ===
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# === Tokenizer & Model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# === Load Dataset ===
files = [os.path.join(CLEAN_DIR, f) for f in os.listdir(CLEAN_DIR) if f.endswith(".jsonl")]
dataset = JsonlDataset(files, tokenizer)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === HF Trainer ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_steps=50,
    save_steps=200,
    save_total_limit=3,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="no",
    bf16=torch.cuda.is_available(),
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# === Train ===
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
