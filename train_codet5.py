#!/usr/bin/env python3
"""
Fine-tune CodeT5-small for Text-to-CAD generation.
Run this on Google Colab (free T4 GPU) or any GPU machine.
"""
import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)

@dataclass
class CADSample:
    text_input: str
    cad_output: Dict[str, Any]
    metadata: Dict[str, Any]

class TextToCADDataset(Dataset):
    def __init__(self, samples: List[CADSample], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample.text_input
        cad_json = json.dumps(sample.cad_output, separators=(',', ':'))
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            cad_json,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

def load_samples(filepath: str) -> List[CADSample]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [CADSample(**item) for item in data]

def load_all_samples(data_dir: Path) -> List[CADSample]:
    all_samples = []
    for f in data_dir.glob("*.json"):
        try:
            samples = load_samples(str(f))
            all_samples.extend(samples)
            print(f"Loaded {len(samples)} samples from {f.name}")
        except Exception as e:
            print(f"Failed to load {f}: {e}")
    return all_samples

def main():
    # Configuration
    DATA_DIR = Path("./data")
    MODEL_NAME = "google/flan-t5-small"
    OUTPUT_DIR = "./cad_flan_t5_finetuned"
    NUM_EPOCHS = 10
    BATCH_SIZE = 4  # Adjust based on GPU memory
    LEARNING_RATE = 3e-4
    MAX_LENGTH = 512
    
    print(f"Loading data from {DATA_DIR}...")
    samples = load_all_samples(DATA_DIR)
    print(f"Total samples: {len(samples)}")
    
    if len(samples) == 0:
        print("No training data found!")
        return
    
    # Show sample
    print(f"\nSample: {samples[0].text_input}")
    print(f"CAD: {json.dumps(samples[0].cad_output)}")
    
    # Load tokenizer and model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
    
    # Create dataset
    dataset = TextToCADDataset(samples, tokenizer, max_length=MAX_LENGTH)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments optimized for T4
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        learning_rate=LEARNING_RATE,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard
        save_total_limit=3,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")
    
    # Quick test
    print("\nTesting generation...")
    test_text = "Create a 50mm x 30mm x 20mm block with a 10mm hole"
    inputs = tokenizer(test_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model.cuda()
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=MAX_LENGTH)
    cad_json = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {test_text}")
    print(f"Output: {cad_json}")

if __name__ == "__main__":
    main()