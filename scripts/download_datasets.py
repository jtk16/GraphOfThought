
import os
from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define datasets and their output directories
datasets_to_download = {
    "openai/gsm8k": "main",
    "Josephgflowers/OpenOrca-Step-by-step-reasoning": None
}
output_dir = "c:/Users/jackt/Downloads/Coding/GraphOfThought/data/raw"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
logging.info(f"Output directory '{output_dir}' is ready.")

# Loop through the datasets and download them
for dataset_name, config in datasets_to_download.items():
    sanitized_name = dataset_name.replace("/", "_")
    save_path = os.path.join(output_dir, sanitized_name)
    
    if os.path.exists(save_path):
        logging.info(f"Dataset '{dataset_name}' already exists at '{save_path}'. Skipping download.")
        continue

    logging.info(f"Downloading '{dataset_name}'...")
    try:
        # Load the dataset from the Hugging Face Hub
        dataset = load_dataset(dataset_name, config if config else "default")
        
        logging.info(f"Saving '{dataset_name}' to '{save_path}'...")
        # Save the dataset to the specified path
        dataset.save_to_disk(save_path)
        
        logging.info(f"Dataset '{dataset_name}' downloaded and saved successfully.")
    except Exception as e:
        logging.error(f"Failed to download or save '{dataset_name}'. Error: {e}")

logging.info("All dataset operations complete.")
