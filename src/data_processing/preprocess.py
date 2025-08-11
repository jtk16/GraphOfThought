

import os
import json
import logging
from datasets import load_from_disk
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_gsm8k_example(example, index):
    """Converts a single gsm8k example to our graph format."""
    question = example['question']
    answer_text = example['answer']
    
    # Split the answer into reasoning and the final result
    parts = answer_text.split('####')
    reasoning = parts[0].strip()
    final_answer = parts[1].strip() if len(parts) > 1 else ''
    
    # Create nodes for each step in the reasoning
    steps = [s.strip() for s in reasoning.split('\n') if s.strip()]
    nodes = [{'id': f'step_{i+1}', 'text': text} for i, text in enumerate(steps)]
    
    # Create sequential edges
    edges = []
    if len(nodes) > 1:
        edges = [{'source': f'step_{i}', 'target': f'step_{i+1}', 'type': 'sequential'} for i in range(1, len(nodes))]
        
    return {
        'id': f'gsm8k_{index}',
        'question': question,
        'final_answer': final_answer,
        'nodes': nodes,
        'edges': edges
    }

def process_openorca_example(example, index):
    """Converts a single OpenOrca example to our graph format."""
    question = example['user']
    reasoning_text = example['assistant']
    
    # For OpenOrca, we treat each line of the assistant's response as a node.
    # There isn't always a clear final answer to extract, so we leave it blank.
    steps = [s.strip() for s in reasoning_text.split('\n') if s.strip()]
    nodes = [{'id': f'step_{i+1}', 'text': text} for i, text in enumerate(steps)]
    
    # Create sequential edges
    edges = []
    if len(nodes) > 1:
        edges = [{'source': f'step_{i}', 'target': f'step_{i+1}', 'type': 'sequential'} for i in range(1, len(nodes))]

    return {
        'id': f'openorca_{index}',
        'question': question,
        'final_answer': '', # No distinct final answer in this dataset format
        'nodes': nodes,
        'edges': edges
    }

def main():
    """Main function to load, process, and save the datasets."""
    base_dir = 'c:/Users/jackt/Downloads/Coding/GraphOfThought'
    raw_data_dir = os.path.join(base_dir, 'data', 'raw')
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(processed_data_dir, exist_ok=True)

    datasets_to_process = {
        'openai_gsm8k': {
            'path': os.path.join(raw_data_dir, 'openai_gsm8k'),
            'processor': process_gsm8k_example,
            'output_file': 'gsm8k_graphs.jsonl'
        },
        'Josephgflowers_OpenOrca-Step-by-step-reasoning': {
            'path': os.path.join(raw_data_dir, 'Josephgflowers_OpenOrca-Step-by-step-reasoning'),
            'processor': process_openorca_example,
            'output_file': 'openorca_graphs.jsonl'
        }
    }

    for name, details in datasets_to_process.items():
        logging.info(f"Processing dataset: {name}")
        output_path = os.path.join(processed_data_dir, details['output_file'])
        
        try:
            # Load the dataset from disk
            dataset = load_from_disk(details['path'])
            
            with open(output_path, 'w', encoding='utf-8') as f_out:
                # Process both train and test splits if they exist
                for split in dataset.keys():
                    logging.info(f"Processing split: {split}")
                    for i, example in enumerate(dataset[split]):
                        processed_graph = details['processor'](example, f'{split}_{i}')
                        f_out.write(json.dumps(processed_graph) + '\n')
            
            logging.info(f"Successfully processed and saved to {output_path}")

        except FileNotFoundError:
            logging.error(f"Dataset path not found: {details['path']}. Please ensure it's downloaded.")
        except Exception as e:
            logging.error(f"An error occurred while processing {name}: {e}")

if __name__ == '__main__':
    main()

