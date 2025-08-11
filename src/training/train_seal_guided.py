#!/usr/bin/env python3
"""
Enhanced SEAL training with guided exploration for improved initial RL phase.
Implements curriculum learning, value-guided search, and demonstration learning.
"""

import os
import sys
import json
import argparse
import logging
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.kvtg.controller import KVTGController
from src.kvtg.storage import KVTGStorage
from src.kvtg.guided_exploration import (
    create_guided_exploration_system, 
    ValueGuidedController, 
    CurriculumManager, 
    DemonstrationLearning,
    NodeValue
)
from src.seal.adaptation import SEALAdapter
from src.evaluation.math_evaluator import MathEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class GuidedTrainingConfig:
    """Configuration for guided SEAL training."""
    # Basic training parameters
    model_path: str
    dataset_path: str
    output_dir: str
    max_iterations: int = 100
    
    # KVTG parameters
    exploration_budget: int = 20
    beam_width: int = 3
    
    # SEAL parameters
    learning_rate: float = 5e-5
    success_threshold: int = 5
    
    # Guided exploration parameters
    initial_difficulty: float = 0.3
    curriculum_adaptation_rate: float = 0.1
    exploration_temperature: float = 0.8
    value_weight: float = 0.6
    
    # Progressive training phases
    enable_warmup_phase: bool = True
    warmup_iterations: int = 20
    enable_curriculum: bool = True
    enable_demonstrations: bool = True
    
    # Advanced options
    save_demonstrations: bool = True
    demonstration_buffer_size: int = 50
    value_learning_enabled: bool = False  # For future neural value function

class GuidedSEALTrainer:
    """Enhanced SEAL trainer with guided exploration capabilities."""
    
    def __init__(self, config: GuidedTrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize model and components
        self._initialize_model()
        self._initialize_components()
        self._initialize_guidance_systems()
        
        # Training state
        self.current_phase = "warmup" if config.enable_warmup_phase else "main"
        self.phase_iteration = 0
        self.global_iteration = 0
        
        # Statistics tracking
        self.training_stats = {
            'phases': {},
            'difficulty_progression': [],
            'value_score_history': [],
            'demonstration_quality': [],
            'curriculum_adaptations': 0
        }
        
        self.logger.info("🎯 GuidedSEALTrainer initialized with enhanced guidance")
    
    def _initialize_model(self):
        """Initialize the language model and tokenizer."""
        self.logger.info(f"📥 Loading model from '{self.config.model_path}'")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path, 
            torch_dtype=torch.bfloat16
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.logger.info(f"✅ Model loaded on {self.device}")
    
    def _initialize_components(self):
        """Initialize KVTG, SEAL, and evaluation components."""
        # KV-cache storage
        self.storage = KVTGStorage(
            max_memory_items=50,
            persist_to_disk=True,
            storage_dir=os.path.join(self.config.output_dir, "kv_cache"),
            compress=True
        )
        
        # Base KVTG controller
        self.base_controller = KVTGController(
            model=self.model,
            tokenizer=self.tokenizer,
            storage=self.storage,
            exploration_budget=self.config.exploration_budget,
            beam_width=self.config.beam_width
        )
        
        # SEAL adapter
        self.seal_adapter = SEALAdapter(
            model=self.model,
            tokenizer=self.tokenizer,
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate
        )
        
        # Evaluator
        self.evaluator = MathEvaluator()
        
        self.logger.info("✅ Core components initialized")
    
    def _initialize_guidance_systems(self):
        """Initialize guided exploration systems."""
        # Create guided exploration system
        self.guided_controller, self.curriculum_manager, self.demonstration_learning = \
            create_guided_exploration_system(
                self.base_controller, 
                difficulty=self.config.initial_difficulty
            )
        
        # Override some parameters from config
        self.guided_controller.exploration_temperature = self.config.exploration_temperature
        self.guided_controller.value_weight = self.config.value_weight
        self.curriculum_manager.adaptation_rate = self.config.curriculum_adaptation_rate
        
        # Active controller (will switch between base and guided)
        self.active_controller = self.guided_controller if self.config.enable_curriculum else self.base_controller
        
        self.logger.info("🎯 Guidance systems initialized")
    
    def load_and_filter_problems(self) -> List[Dict[str, Any]]:
        """Load problems and apply curriculum filtering."""
        problems = self._load_problem_dataset(self.config.dataset_path)
        
        if self.config.enable_curriculum and len(problems) > 10:
            # Apply curriculum filtering
            filtered_problems = self.curriculum_manager.filter_problems_by_difficulty(problems)
            difficulty_name = self.curriculum_manager.get_difficulty_level_name()
            
            self.logger.info(f"📚 Curriculum: {len(filtered_problems)} problems at '{difficulty_name}' level")
            return filtered_problems
        
        return problems
    
    def _load_problem_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
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
            self.logger.info(f"📖 Loaded {len(problems)} problems from {dataset_path}")
        except FileNotFoundError:
            self.logger.error(f"❌ Dataset file not found: {dataset_path}")
            # Use sample problems for demonstration
            problems = [
                {'question': 'What is 12 + 15?', 'answer': '27', 'id': 'warmup_1'},
                {'question': 'Sarah has 8 stickers. She buys 5 more. How many does she have now?', 'answer': '13', 'id': 'warmup_2'},
                {'question': 'What is 6 × 4?', 'answer': '24', 'id': 'warmup_3'},
                {'question': 'Tom has 20 marbles. He gives 7 to his friend. How many marbles does Tom have left?', 'answer': '13', 'id': 'intermediate_1'},
                {'question': 'A pizza costs $12. If you buy 3 pizzas and pay with a $50 bill, how much change do you get?', 'answer': '14', 'id': 'intermediate_2'}
            ]
            self.logger.info("🎭 Using sample problems for demonstration")
        
        return problems
    
    def run_warmup_phase(self, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run warmup phase with very simple problems and high guidance."""
        self.logger.info("🔥 Starting Warmup Phase")
        self.logger.info("=" * 50)
        
        # Use only the simplest problems for warmup
        warmup_problems = [p for p in problems if 'warmup' in p.get('id', '')][:self.config.warmup_iterations]
        if len(warmup_problems) < 5:
            warmup_problems = problems[:min(self.config.warmup_iterations, len(problems))]
        
        # Increase guidance during warmup
        original_temp = self.guided_controller.exploration_temperature
        original_weight = self.guided_controller.value_weight
        
        self.guided_controller.exploration_temperature = 0.5  # More deterministic
        self.guided_controller.value_weight = 0.8  # Heavily favor quality
        
        warmup_stats = {
            'problems_attempted': 0,
            'problems_solved': 0,
            'total_nodes_explored': 0,
            'avg_value_scores': [],
            'demonstration_examples': []
        }
        
        try:
            for i, problem in enumerate(warmup_problems):
                if i >= self.config.warmup_iterations:
                    break
                
                self.logger.info(f"\n🔥 Warmup {i+1}/{min(len(warmup_problems), self.config.warmup_iterations)}")
                self.logger.info(f"Problem: {problem['question']}")
                
                result = self._solve_single_problem(problem, phase="warmup")
                warmup_stats['problems_attempted'] += 1
                
                if result['success']:
                    warmup_stats['problems_solved'] += 1
                    
                    # Add to demonstrations
                    if self.config.enable_demonstrations:
                        solution_path = [node.text for node in result['graph'].nodes]
                        self.demonstration_learning.add_demonstration(
                            problem['question'], solution_path, True
                        )
                        warmup_stats['demonstration_examples'].append({
                            'problem': problem['question'],
                            'solution_length': len(solution_path)
                        })
                
                warmup_stats['total_nodes_explored'] += result.get('nodes_explored', 0)
                if result.get('avg_node_value'):
                    warmup_stats['avg_value_scores'].append(result['avg_node_value'])
                
                self.phase_iteration += 1
                self.global_iteration += 1
        
        finally:
            # Restore original parameters
            self.guided_controller.exploration_temperature = original_temp
            self.guided_controller.value_weight = original_weight
        
        warmup_accuracy = warmup_stats['problems_solved'] / max(1, warmup_stats['problems_attempted']) * 100
        self.logger.info(f"\n🎉 Warmup Phase Complete!")
        self.logger.info(f"   Accuracy: {warmup_accuracy:.1f}%")
        self.logger.info(f"   Demonstrations collected: {len(warmup_stats['demonstration_examples'])}")
        
        self.training_stats['phases']['warmup'] = warmup_stats
        return warmup_stats
    
    def run_main_training_phase(self, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run main training phase with curriculum learning and full guidance."""
        self.logger.info("🚀 Starting Main Training Phase")
        self.logger.info("=" * 50)
        
        main_stats = {
            'problems_attempted': 0,
            'problems_solved': 0,
            'seal_updates': 0,
            'curriculum_adaptations': 0,
            'difficulty_progression': [],
            'value_score_progression': []
        }
        
        successful_count = 0
        problems_processed = 0
        
        while problems_processed < self.config.max_iterations:
            # Get current difficulty-appropriate problems
            current_problems = self.load_and_filter_problems()
            
            for problem in current_problems:
                if problems_processed >= self.config.max_iterations:
                    break
                
                self.logger.info(f"\n🧠 Main Training {problems_processed + 1}/{self.config.max_iterations}")
                difficulty_name = self.curriculum_manager.get_difficulty_level_name()
                self.logger.info(f"📚 Difficulty Level: {difficulty_name}")
                self.logger.info(f"Problem: {problem['question']}")
                
                result = self._solve_single_problem(problem, phase="main")
                main_stats['problems_attempted'] += 1
                problems_processed += 1
                
                # Record curriculum feedback
                if self.config.enable_curriculum:
                    self.curriculum_manager.record_attempt(result['success'])
                
                if result['success']:
                    main_stats['problems_solved'] += 1
                    successful_count += 1
                    
                    # Apply SEAL fine-tuning
                    self.logger.info("🎯 Applying SEAL fine-tuning...")
                    try:
                        self.seal_adapter.finetune_on_path(result['graph'])
                        main_stats['seal_updates'] += 1
                        
                        # Save checkpoint periodically
                        if successful_count % self.config.success_threshold == 0:
                            checkpoint_path = os.path.join(
                                self.config.output_dir, 
                                f"checkpoint_main_{problems_processed}"
                            )
                            self.seal_adapter.save_model(checkpoint_path)
                            self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
                        
                    except Exception as e:
                        self.logger.error(f"❌ SEAL fine-tuning failed: {e}")
                    
                    # Add to demonstrations
                    if self.config.enable_demonstrations:
                        solution_path = [node.text for node in result['graph'].nodes]
                        self.demonstration_learning.add_demonstration(
                            problem['question'], solution_path, True
                        )
                
                # Track statistics
                if result.get('avg_node_value'):
                    main_stats['value_score_progression'].append(result['avg_node_value'])
                
                # Adapt curriculum
                if self.config.enable_curriculum and problems_processed % 10 == 0:
                    old_difficulty = self.curriculum_manager.current_difficulty
                    self.curriculum_manager.adapt_difficulty()
                    
                    if abs(old_difficulty - self.curriculum_manager.current_difficulty) > 0.05:
                        main_stats['curriculum_adaptations'] += 1
                        main_stats['difficulty_progression'].append({
                            'iteration': problems_processed,
                            'difficulty': self.curriculum_manager.current_difficulty,
                            'level_name': self.curriculum_manager.get_difficulty_level_name()
                        })
                
                self.global_iteration += 1
                
                # Progress logging
                if problems_processed % 10 == 0:
                    accuracy = main_stats['problems_solved'] / problems_processed * 100
                    self.logger.info(f"📈 Progress: {accuracy:.1f}% accuracy, {main_stats['seal_updates']} SEAL updates")
        
        final_accuracy = main_stats['problems_solved'] / max(1, main_stats['problems_attempted']) * 100
        self.logger.info(f"\n🏆 Main Training Phase Complete!")
        self.logger.info(f"   Final Accuracy: {final_accuracy:.1f}%")
        self.logger.info(f"   SEAL Updates: {main_stats['seal_updates']}")
        self.logger.info(f"   Curriculum Adaptations: {main_stats['curriculum_adaptations']}")
        
        self.training_stats['phases']['main'] = main_stats
        return main_stats
    
    def _solve_single_problem(self, problem: Dict[str, Any], phase: str = "main") -> Dict[str, Any]:
        """Solve a single problem with guided exploration."""
        start_time = time.time()
        
        try:
            # Use guided exploration
            thought_graph = self.active_controller.explore(problem['question'])
            
            if thought_graph is None:
                return {
                    'success': False,
                    'error': 'no_path_generated',
                    'nodes_explored': 0,
                    'exploration_time': time.time() - start_time
                }
            
            # Evaluate solution
            result = self.evaluator.evaluate_gsm8k_answer(
                "\n".join(node.text for node in thought_graph.nodes),
                problem['answer']
            )
            
            # Calculate average node value scores
            avg_node_value = None
            if hasattr(self.active_controller, 'node_values'):
                node_values = [
                    self.active_controller.node_values[node.id].quality_score 
                    for node in thought_graph.nodes 
                    if node.id in self.active_controller.node_values
                ]
                avg_node_value = sum(node_values) / len(node_values) if node_values else None
            
            exploration_time = time.time() - start_time
            
            if result.is_correct:
                self.logger.info(f"✅ Correct! (Confidence: {result.confidence:.2f}, Time: {exploration_time:.1f}s)")
            else:
                self.logger.info(f"❌ Incorrect (Error: {result.error_type}, Time: {exploration_time:.1f}s)")
            
            return {
                'success': result.is_correct,
                'graph': thought_graph,
                'evaluation_result': result,
                'nodes_explored': len(thought_graph.nodes),
                'avg_node_value': avg_node_value,
                'exploration_time': exploration_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ Problem solving failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'nodes_explored': 0,
                'exploration_time': time.time() - start_time
            }
    
    def train(self) -> Dict[str, Any]:
        """Run the complete guided training pipeline."""
        self.logger.info("🚀 Starting Guided SEAL Training")
        self.logger.info(f"Configuration: {self.config}")
        
        start_time = time.time()
        
        # Load initial problems
        all_problems = self._load_problem_dataset(self.config.dataset_path)
        
        # Phase 1: Warmup (optional)
        if self.config.enable_warmup_phase:
            warmup_stats = self.run_warmup_phase(all_problems)
            self.current_phase = "main"
            self.phase_iteration = 0
        
        # Phase 2: Main training
        main_stats = self.run_main_training_phase(all_problems)
        
        # Finalize training
        total_time = time.time() - start_time
        final_results = self._finalize_training(total_time)
        
        return final_results
    
    def _finalize_training(self, total_time: float) -> Dict[str, Any]:
        """Finalize training and save results."""
        self.logger.info("🎯 Finalizing guided training...")
        
        # Save final model
        final_model_path = os.path.join(self.config.output_dir, "final_guided_model")
        self.seal_adapter.save_model(final_model_path)
        
        # Compile final statistics
        total_solved = sum(
            phase_stats.get('problems_solved', 0) 
            for phase_stats in self.training_stats['phases'].values()
        )
        total_attempted = sum(
            phase_stats.get('problems_attempted', 0)
            for phase_stats in self.training_stats['phases'].values()
        )
        
        final_results = {
            'training_config': self.config.__dict__,
            'total_time': total_time,
            'total_problems_attempted': total_attempted,
            'total_problems_solved': total_solved,
            'final_accuracy': total_solved / max(1, total_attempted) * 100,
            'phases': self.training_stats['phases'],
            'curriculum_final_difficulty': self.curriculum_manager.current_difficulty,
            'demonstration_count': len(self.demonstration_learning.demonstrations)
        }
        
        # Save detailed statistics
        stats_path = os.path.join(self.config.output_dir, "guided_training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Clean up
        self.storage.clear()
        
        self.logger.info(f"🎉 Guided Training Complete!")
        self.logger.info(f"   Total Time: {total_time/60:.1f} minutes")
        self.logger.info(f"   Final Accuracy: {final_results['final_accuracy']:.1f}%")
        self.logger.info(f"   Model saved to: {final_model_path}")
        self.logger.info(f"   Statistics saved to: {stats_path}")
        
        return final_results

def main():
    """Main entry point for guided SEAL training."""
    parser = argparse.ArgumentParser(description="Guided KVTG+SEAL training with curriculum learning")
    
    # Basic parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--dataset_path", type=str, default="data/processed/gsm8k_graphs.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/guided_seal_model")
    parser.add_argument("--max_iterations", type=int, default=100)
    
    # KVTG parameters
    parser.add_argument("--exploration_budget", type=int, default=20)
    parser.add_argument("--beam_width", type=int, default=3)
    
    # SEAL parameters
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--success_threshold", type=int, default=5)
    
    # Guided exploration parameters
    parser.add_argument("--initial_difficulty", type=float, default=0.3)
    parser.add_argument("--curriculum_adaptation_rate", type=float, default=0.1)
    parser.add_argument("--exploration_temperature", type=float, default=0.8)
    parser.add_argument("--value_weight", type=float, default=0.6)
    
    # Training phases
    parser.add_argument("--enable_warmup", action="store_true", default=True)
    parser.add_argument("--warmup_iterations", type=int, default=20)
    parser.add_argument("--disable_curriculum", action="store_true")
    parser.add_argument("--disable_demonstrations", action="store_true")
    
    args = parser.parse_args()
    
    # Create configuration
    config = GuidedTrainingConfig(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        exploration_budget=args.exploration_budget,
        beam_width=args.beam_width,
        learning_rate=args.learning_rate,
        success_threshold=args.success_threshold,
        initial_difficulty=args.initial_difficulty,
        curriculum_adaptation_rate=args.curriculum_adaptation_rate,
        exploration_temperature=args.exploration_temperature,
        value_weight=args.value_weight,
        enable_warmup_phase=args.enable_warmup,
        warmup_iterations=args.warmup_iterations,
        enable_curriculum=not args.disable_curriculum,
        enable_demonstrations=not args.disable_demonstrations
    )
    
    try:
        # Initialize and run trainer
        trainer = GuidedSEALTrainer(config)
        results = trainer.train()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Final accuracy: {results['final_accuracy']:.1f}%")
        print(f"Model saved to: {config.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚡ Training interrupted by user")
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()