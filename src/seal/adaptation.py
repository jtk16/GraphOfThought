import os
import logging
from typing import Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from src.kvtg.graph import ThoughtGraph

class SingleGraphDataset(Dataset):
    """A simple dataset that holds a single tokenized training example."""
    def __init__(self, tokenized_example: Dict[str, Any]):
        self.example = tokenized_example

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return self.example

class SEALAdapter:
    """
    Manages the Self-Adapting Language model (SEAL) fine-tuning process.
    This class takes a successful reasoning path (a ThoughtGraph) and uses it
    to perform a single SFT update on the model.
    """
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, output_dir: str, learning_rate: float = 5e-5, block_size: int = 1024):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.block_size = block_size
        logging.info(f"SEALAdapter initialized. Model updates will be saved to '{self.output_dir}'.")

    def _format_prompt(self, question: str, path: list[str]) -> str:
        """Formats the context into a clear prompt for the model."""
        path_str = "\n".join(path)
        return f"Question: {question}\n\nReasoning Path:\n{path_str}\n\nNext Step:"

    def _create_training_example_from_graph(self, graph: ThoughtGraph) -> Dict[str, Any]:
        """
        Converts a ThoughtGraph into a single, coherent training example.
        This mirrors the logic in `train_base.py`'s dataset but for a single graph.
        """
        full_text = f"Question: {graph.question}\n\nReasoning Path:\n"
        path_nodes = sorted(graph.nodes, key=lambda n: int(n.id)) # Assuming numeric IDs for ordering
        
        for node in path_nodes:
            full_text += f"{node.text}\n"
        
        full_text += "\nNext Step:" # This structure might need refinement
        
        # For simplicity, we'll treat the whole successful path as the completion.
        # A more advanced version could break it down step-by-step.
        tokenized_example = self.tokenizer(
            full_text + self.tokenizer.eos_token,
            truncation=True,
            max_length=self.block_size,
            padding="max_length"
        )
        return tokenized_example

    def finetune_on_path(self, successful_path: ThoughtGraph):
        """
        Performs a single fine-tuning step on the model using the provided successful path.
        """
        logging.info(f"Starting SEAL fine-tuning for graph ID: {successful_path.id}")

        training_example = self._create_training_example_from_graph(successful_path)
        train_dataset = SingleGraphDataset(training_example)

        # Configure minimal training arguments for a single update step
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, "checkpoints"),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            learning_rate=self.learning_rate,
            logging_steps=1,
            fp16=True,
            report_to="none",
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        trainer.train()
        logging.info("SEAL fine-tuning step complete.")

    def save_model(self, final_save_path: str = None):
        """Saves the final state of the adapted model."""
        if final_save_path is None:
            final_save_path = os.path.join(self.output_dir, "final_model")
        self.model.save_pretrained(final_save_path)
        self.tokenizer.save_pretrained(final_save_path)
        logging.info(f"Final SEAL model saved to '{final_save_path}'.")