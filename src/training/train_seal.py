import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.kvtg.controller import KVTGController
from src.kvtg.storage import KVTGStorage
from src.seal.adaptation import SEALAdapter
from src.evaluation.math_evaluator import MathEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_problem_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load problems from JSONL file."""
    problems = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                problems.append({
                    'question': data['question'],
                    'answer': data.get('final_answer', ''),
                    'id': data.get('id', f'problem_{len(problems)}')
                })
        logging.info(f"Loaded {len(problems)} problems from {dataset_path}")
    except FileNotFoundError:
        logging.error(f"Dataset file not found: {dataset_path}")
        # Create some sample problems for demonstration
        problems = [
            {'question': 'What is 15 + 27?', 'answer': '42', 'id': 'sample_1'},
            {'question': 'If a book costs $12 and I buy 3 books, how much do I pay?', 'answer': '36', 'id': 'sample_2'},
            {'question': 'What is 8 × 7?', 'answer': '56', 'id': 'sample_3'}
        ]
        logging.info("Using sample problems for demonstration")
    
    return problems

def evaluate_solution(generated_path, golden_answer, evaluator: MathEvaluator) -> bool:
    """
    Evaluate if the generated reasoning path produces the correct answer.
    """
    if not generated_path or not generated_path.nodes:
        logging.warning("Empty or invalid generated path")
        return False
    
    # Get the final reasoning from all nodes in the path
    full_solution = "\n".join(node.text for node in generated_path.nodes)
    
    # Use the robust math evaluator
    result = evaluator.evaluate_gsm8k_answer(full_solution, str(golden_answer))
    
    if result.is_correct:
        logging.info(f"✓ Correct solution found (confidence: {result.confidence}): {result.extracted_answer}")
        return True
    else:
        logging.warning(f"✗ Incorrect solution. Got: {result.extracted_answer}, Expected: {golden_answer}")
        logging.debug(f"Error type: {result.error_type}")
        return False

def main():
    parser = argparse.ArgumentParser(description="KVTG+SEAL self-improvement training loop.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base SFT model checkpoint.")
    parser.add_argument("--dataset_path", type=str, default="data/processed/gsm8k_graphs.jsonl", help="Path to the problem dataset.")
    parser.add_argument("--output_dir", type=str, default="models/seal_model", help="Directory to save the self-improved model.")
    parser.add_argument("--max_iterations", type=int, default=100, help="Maximum number of self-improvement iterations.")
    parser.add_argument("--exploration_budget", type=int, default=20, help="KVTG exploration budget per problem.")
    parser.add_argument("--beam_width", type=int, default=3, help="Beam width for KVTG exploration.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for SEAL adaptation.")
    parser.add_argument("--success_threshold", type=int, default=5, help="Number of successes before saving checkpoint.")
    
    args = parser.parse_args()
    logging.info(f"Starting SEAL training with arguments: {args}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Model and Tokenizer
    logging.info(f"Loading model from '{args.model_path}'")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Model loaded on device: {device}")

    # 2. Initialize KVTG Controller, SEAL Adapter, and Evaluator
    storage = KVTGStorage(max_memory_items=50, persist_to_disk=True, 
                         storage_dir=os.path.join(args.output_dir, "kv_cache"))
    
    kvtg_controller = KVTGController(
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