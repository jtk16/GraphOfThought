

import os
import sys
import json
import argparse
import logging
from dataclasses import dataclass

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from src.kvtg.graph import ThoughtGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GraphReasoningDataset(Dataset):
    """
    A PyTorch Dataset to transform graph-structured reasoning data into a format
    suitable for next-step prediction fine-tuning.
    """
    def __init__(self, tokenizer, file_path: str, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []

        logging.info(f"Loading and processing data from {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                graph = ThoughtGraph.from_dict(data)
                
                # For each edge in the graph, create a training example.
                # This teaches the model to predict the next step given the previous ones.
                path = []
                for node in graph.nodes:
                    # The input is the story so far.
                    prompt = self._format_prompt(graph.question, path)
                    # The label is the next step.
                    completion = node.text

                    # Structural Decision: Format for Next-Step Prediction
                    # We combine the prompt and completion, and the model learns to predict
                    # the completion part. The loss is only calculated on the completion tokens.
                    text = f"{prompt}{completion}{self.tokenizer.eos_token}"
                    tokenized_example = self.tokenizer(text, truncation=True, max_length=self.block_size, padding="max_length")
                    
                    self.examples.append(tokenized_example)
                    
                    # Add the current node to the path for the next iteration
                    path.append(node.text)

    def _format_prompt(self, question: str, path: list[str]) -> str:
        """Formats the context into a clear prompt for the model."""
        path_str = "\n".join(path)
        return f"Question: {question}\n\nReasoning Path:\n{path_str}\n\nNext Step:"

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

def main():
    parser = argparse.ArgumentParser(description="Base SFT training for Graph of Thought model.")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", help="Name of the base model to fine-tune.")
    parser.add_argument("--dataset_path", type=str, default="c:/Users/jackt/Downloads/Coding/GraphOfThought/data/processed/gsm8k_graphs.jsonl", help="Path to the processed graph dataset.")
    parser.add_argument("--output_dir", type=str, default="c:/Users/jackt/Downloads/Coding/GraphOfThought/models/base_sft_model", help="Directory to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--block_size", type=int, default=1024, help="Maximum sequence length.")
    args = parser.parse_args()

    logging.info(f"Starting training with arguments: {args}")

    # 1. Load Tokenizer and Model
    logging.info(f"Loading model and tokenizer for '{args.model_name}'")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Set a padding token if one is not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

    # 2. Load Dataset
    train_dataset = GraphReasoningDataset(tokenizer=tokenizer, file_path=args.dataset_path, block_size=args.block_size)

    # 3. Configure Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=500,
        fp16=True, # Use mixed precision for speed and memory savings on compatible GPUs
        report_to="none", # Disable wandb/tensorboard reporting for this example
    )

    # Data collator handles padding and creating labels for language modeling.
    # The model will predict the next token, and loss will be calculated only on the completion part.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 5. Start Training
    logging.info("Starting model training...")
    trainer.train()
    logging.info("Training finished.")

    # 6. Save the final model
    final_model_path = os.path.join(args.output_dir, "final_checkpoint")
    trainer.save_model(final_model_path)
    logging.info(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()

