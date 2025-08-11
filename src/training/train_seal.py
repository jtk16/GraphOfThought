import os
import sys
import json
import argparse
import logging
import re
from typing import Iterable, Dict, Any, Optional

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.kvtg.controller import KVTGController
from src.kvtg.storage import KVTGStorage
from src.seal.adaptation import SEALAdapter
from src.kvtg.graph import ThoughtGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_problem_dataset(file_path: str) -> Iterable[Dict[str, Any]]:
    """
    Loads problems from a JSONL file. Assumes each line is a JSON object
    representing a ThoughtGraph, and extracts the question and final answer.
    """
    logging.info(f"Loading problems from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # We only need the question and the ground truth answer for the SEAL loop
                if 'question' in data and 'final_answer' in data:
                    yield {'question': data['question'], 'answer': data['final_answer']}
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Skipping malformed line: {line.strip()} | Error: {e}")

def extract_answer(text: str) -> Optional[str]:
    """
    Extracts the numerical answer from a string like 'Final Answer: 123'.
    """
    match = re.search(r'(\d+(\.\d+)?)', str(text))
    return match.group(1) if match else None

def evaluate_solution(generated_graph: ThoughtGraph, golden_answer: str) -> bool:
    """
    Evaluates if the generated graph's solution matches the golden answer.
    For GSM8K, this involves parsing the final numerical answer.
    """
    if not generated_graph:
        return False
        
    terminal_nodes = generated_graph.get_terminal_nodes()
    if not terminal_nodes:
        logging.warning("Evaluation failed: No terminal node found in the generated graph.")
        return False

    # Use the first terminal node found as the proposed solution
    generated_text = terminal_nodes[0].text
    generated_answer = extract_answer(generated_text)
    
    # Clean up golden answer for comparison
    golden_answer_clean = extract_answer(golden_answer)

    if generated_answer is None:
        logging.warning(f"Evaluation failed: Could not extract an answer from '{generated_text}'.")
        return False

    is_correct = generated_answer == golden_answer_clean
    if is_correct:
        logging.info(f"Correct solution found! Generated: {generated_answer}, Golden: {golden_answer_clean}")
    else:
        logging.warning(f"Incorrect solution. Generated: {generated_answer}, Golden: {golden_answer_clean}")
        
    return is_correct

def main():
    parser = argparse.ArgumentParser(description="KVTG+SEAL self-improvement training loop.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base SFT model checkpoint.")
    parser.add_argument("--dataset_path", type=str, default="c:/Users/jackt/Downloads/Coding/GraphOfThought/data/processed/gsm8k_graphs.jsonl", help="Path to the problem dataset (in graph JSONL format).")
    parser.add_argument("--output_dir", type=str, default="c:/Users/jackt/Downloads/Coding/GraphOfThought/models/seal_model", help="Directory to save the self-improved model.")
    parser.add_argument("--max_iterations", type=int, default=100, help="Maximum number of self-improvement iterations.")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Learning rate for SEAL fine-tuning steps.")
    parser.add_argument("--exploration_budget", type=int, default=30, help="Max nodes for KVTG controller to generate per problem.")
    parser.add_argument("--beam_width", type=int, default=3, help="Beam width for generation in KVTG controller.")
    parser.add_argument("--gpu_cache_capacity", type=int, default=50, help="Number of KV-cache snapshots to hold in GPU VRAM.")
    args = parser.parse_args()

    logging.info(f"Starting SEAL training with arguments: {args}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Model and Tokenizer
    logging.info(f"Loading model from '{args.model_path}'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)

    # 2. Initialize KVTG Controller and SEAL Adapter
    kv_storage = KVTGStorage(device=device, gpu_capacity=args.gpu_cache_capacity)
    kvtg_controller = KVTGController(model, tokenizer, storage=kv_storage, exploration_budget=args.exploration_budget, beam_width=args.beam_width)
    seal_adapter = SEALAdapter(model, tokenizer, output_dir=args.output_dir, learning_rate=args.learning_rate)

    # 3. Load Problem Dataset
    problem_dataset = load_problem_dataset(args.dataset_path)

    # 4. The Self-Improvement Loop
    logging.info("Starting the KVTG+SEAL self-improvement loop...")
    successful_updates = 0
    for i, problem in enumerate(problem_dataset):
        if i >= args.max_iterations:
            logging.info(f"Maximum iterations ({args.max_iterations}) reached.")
            break

        logging.info(f"\n--- Iteration {i+1}/{args.max_iterations} ---")
        logging.info(f"Problem: {problem['question']}")

        # a. Explore with KVTG
        generated_graph = kvtg_controller.explore(problem['question'])

        # b. Evaluate the solution
        is_correct = evaluate_solution(generated_graph, problem['answer'])

        # c. Fine-tune with SEAL on success
            # The adapter takes the successful path, creates a training example,
            # and performs a fine-tuning step on the model.
            # seal_adapter.finetune_on_path(successful_path)
            logging.info("SEAL update complete. Model has been improved.")
        else:
            logging.info("Solution was incorrect or not found. Moving to next problem.")

    logging.info("SEAL training loop finished.")
    # seal_adapter.save_model() # Final save

if __name__ == "__main__":
    # main() # Commented out until dependencies are implemented
        model=model, 
        tokenizer=tokenizer, 
        storage=storage,
        exploration_budget=args.exploration_budget,
        beam_width=args.beam_width
    )
    
    seal_adapter = SEALAdapter(
        model=model, 
        tokenizer=tokenizer, 
        output_dir=args.output_dir,
        learning_rate=args.learning_rate
    )
    
    evaluator = MathEvaluator()

    # 3. Load Problem Dataset
    problem_dataset = load_problem_dataset(args.dataset_path)
    logging.info(f"Loaded {len(problem_dataset)} problems for training")

    # 4. Training Statistics
    stats = {
        'total_problems': len(problem_dataset),
        'successful_solutions': 0,
        'failed_solutions': 0,
        'seal_updates': 0,
        'iteration_results': []
    }

    # 5. The Self-Improvement Loop
    logging.info("Starting the KVTG+SEAL self-improvement loop...")
    successful_count = 0
    
    for i, problem in enumerate(problem_dataset):
        if i >= args.max_iterations:
            logging.info("Maximum iterations reached.")
            break

        logging.info(f"\n=== Iteration {i+1}/{min(len(problem_dataset), args.max_iterations)} ===")
        logging.info(f"Problem ID: {problem.get('id', 'unknown')}")
        logging.info(f"Question: {problem['question']}")
        logging.info(f"Expected Answer: {problem['answer']}")

        iteration_start = logging.getLogger().getEffectiveLevel()
        
        try:
            # a. Explore with KVTG
            logging.info("🔍 Starting KVTG exploration...")
            successful_path = kvtg_controller.explore(problem['question'])
            
            if successful_path is None:
                logging.warning("KVTG exploration failed to generate any path")
                stats['failed_solutions'] += 1
                stats['iteration_results'].append({
                    'iteration': i+1,
                    'problem_id': problem.get('id', 'unknown'),
                    'success': False,
                    'error': 'no_path_generated'
                })
                continue

            # b. Evaluate the solution
            logging.info("📊 Evaluating generated solution...")
            is_correct = evaluate_solution(successful_path, problem['answer'], evaluator)

            # c. Record results
            if is_correct:
                stats['successful_solutions'] += 1
                successful_count += 1
                
                # Apply SEAL fine-tuning on successful solutions
                logging.info("🎯 Solution is correct! Applying SEAL fine-tuning...")
                try:
                    seal_adapter.finetune_on_path(successful_path)
                    stats['seal_updates'] += 1
                    logging.info("✅ SEAL update complete. Model has been improved.")
                    
                    # Save checkpoint periodically
                    if successful_count % args.success_threshold == 0:
                        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{i+1}")
                        logging.info(f"💾 Saving checkpoint after {successful_count} successes...")
                        seal_adapter.save_model(checkpoint_path)
                        
                except Exception as e:
                    logging.error(f"SEAL fine-tuning failed: {e}")
                
                stats['iteration_results'].append({
                    'iteration': i+1,
                    'problem_id': problem.get('id', 'unknown'),
                    'success': True,
                    'nodes_explored': len(successful_path.nodes),
                    'final_answer': successful_path.final_answer
                })
            else:
                stats['failed_solutions'] += 1
                logging.info("❌ Solution was incorrect. Moving to next problem.")
                stats['iteration_results'].append({
                    'iteration': i+1,
                    'problem_id': problem.get('id', 'unknown'),
                    'success': False,
                    'nodes_explored': len(successful_path.nodes) if successful_path else 0,
                    'error': 'incorrect_answer'
                })
                
        except Exception as e:
            logging.error(f"Error during iteration {i+1}: {e}")
            stats['failed_solutions'] += 1
            stats['iteration_results'].append({
                'iteration': i+1,
                'problem_id': problem.get('id', 'unknown'),
                'success': False,
                'error': str(e)
            })
        
        # Log progress periodically
        if (i+1) % 10 == 0:
            accuracy = stats['successful_solutions'] / (i+1) * 100
            logging.info(f"📈 Progress: {i+1} problems processed, {accuracy:.1f}% accuracy, {stats['seal_updates']} SEAL updates")

    # 6. Final Statistics and Model Saving
    final_accuracy = stats['successful_solutions'] / len(problem_dataset) * 100 if problem_dataset else 0
    
    logging.info(f"\n🎉 SEAL training loop finished!")
    logging.info(f"📊 Final Statistics:")
    logging.info(f"  • Total problems: {stats['total_problems']}")
    logging.info(f"  • Successful solutions: {stats['successful_solutions']}")
    logging.info(f"  • Failed solutions: {stats['failed_solutions']}")
    logging.info(f"  • Accuracy: {final_accuracy:.2f}%")
    logging.info(f"  • SEAL updates applied: {stats['seal_updates']}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    logging.info(f"💾 Saving final model to {final_model_path}...")
    seal_adapter.save_model(final_model_path)
    
    # Save training statistics
    stats_path = os.path.join(args.output_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logging.info(f"📈 Training statistics saved to {stats_path}")
    
    # Clean up KV-cache storage
    storage.clear()
    logging.info("🧹 KV-cache storage cleared")

if __name__ == "__main__":
    main()