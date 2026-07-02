#!/usr/bin/env python3
"""
Fine-tune CodeT5-base for Text-to-CAD generation.
Optimized for Google Colab (A100/T4) or local GPU training.
"""
import os
import json
import torch
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split

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

def compute_metrics(eval_preds):
    """Compute exact match and valid JSON rate for CAD generation"""
    from transformers import AutoTokenizer
    import re
    
    preds, labels = eval_preds
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    exact_match = 0
    valid_json = 0
    valid_ops = 0
    
    for pred, label in zip(decoded_preds, decoded_labels):
        if pred.strip() == label.strip():
            exact_match += 1
        
        try:
            spec = json.loads(pred)
            valid_json += 1
            if 'operations' in spec and isinstance(spec['operations'], list):
                valid_ops += 1
        except:
            pass
    
    n = len(decoded_preds)
    return {
        'exact_match': exact_match / n if n > 0 else 0,
        'valid_json_rate': valid_json / n if n > 0 else 0,
        'valid_ops_rate': valid_ops / n if n > 0 else 0,
    }

def main():
    # Configuration
    DATA_DIR = Path("./data")
    MODEL_NAME = "Salesforce/codet5-base"  # 220M params - better for code generation
    OUTPUT_DIR = "./cad_codet5_base_finetuned"
    NUM_EPOCHS = 40
    BATCH_SIZE = 8  # Adjust based on GPU memory (A100: 16-32, T4: 4-8)
    LEARNING_RATE = 2e-4
    MAX_LENGTH = 512
    VAL_SPLIT = 0.1
    SEED = 42
    
    torch.manual_seed(SEED)
    random.seed(SEED)
    
    print(f"Loading data from {DATA_DIR}...")
    samples = load_all_samples(DATA_DIR)
    print(f"Total samples: {len(samples)}")
    
    if len(samples) == 0:
        print("No training data found!")
        return
    
    # Show sample
    print(f"\nSample: {samples[0].text_input[:100]}...")
    print(f"CAD: {json.dumps(samples[0].cad_output)}")
    
    # Train/val split
    train_samples, val_samples = train_test_split(
        samples, test_size=VAL_SPLIT, random_state=SEED
    )
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Load tokenizer and model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
    
    # Create datasets
    train_dataset = TextToCADDataset(train_samples, tokenizer, max_length=MAX_LENGTH)
    val_dataset = TextToCADDataset(val_samples, tokenizer, max_length=MAX_LENGTH)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments - optimized for CodeT5
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,  # A100+
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_valid_ops_rate",
        greater_is_better=True,
        lr_scheduler_type="cosine",
        label_smoothing_factor=0.1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    results = trainer.evaluate()
    print(f"Results: {results}")
    
    # Test generation
    print("\nTesting generation...")
    model.eval()
    test_cases = [
        "Create a 50mm x 30mm x 20mm block with a 10mm hole",
        "Make a 40mm cube with 3mm fillets",
        "Design a cylinder 100mm long, 25mm diameter with 5mm keyway",
        "Create a hollow cylinder 30mm OD, 20mm ID, 50mm height",
    ]
    
    for tc in test_cases:
        inputs = tokenizer(tc, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model.cuda()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=MAX_LENGTH, 
                num_beams=4,
                early_stopping=True,
                temperature=0.7
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input:  {tc}")
        print(f"Output: {result}")
        try:
            json.loads(result)
            print("  ✓ Valid JSON")
        except:
            print("  ✗ Invalid JSON")
        print()

if __name__ == "__main__":
    main()