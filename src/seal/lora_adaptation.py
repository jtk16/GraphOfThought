"""
LoRA-based SEAL Adapter Implementation

This module implements a safer variant of SEAL using Low-Rank Adaptation (LoRA)
instead of full weight updates. This addresses the catastrophic forgetting and
computational cost concerns identified in the feedback.

Key features:
- LoRA adapters instead of full model fine-tuning
- Frozen base model to preserve capabilities
- Rollback mechanism for failed adaptations
- Memory-efficient training
- Safety validation before applying updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import copy

from .adaptation import SEALAdapter
from ..kvtg.graph import ThoughtGraph

logger = logging.getLogger(__name__)

@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation parameters."""
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for transformer models
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer that can be added to existing linear layers."""
    
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze the original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA parameters
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        self.scaling = self.alpha / self.rank
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining original and LoRA outputs."""
        # Original layer output
        original_output = self.original_layer(x)
        
        # LoRA adaptation
        lora_output = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self):
        """Merge LoRA weights into the original layer (for inference efficiency)."""
        with torch.no_grad():
            delta_weight = (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.data += delta_weight
            
    def reset_parameters(self):
        """Reset LoRA parameters to initial state."""
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

class SafetyValidator:
    """Validates model updates for safety and quality."""
    
    def __init__(self, base_model, tokenizer, validation_problems: List[Dict[str, Any]]):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.validation_problems = validation_problems
        
        # Safety thresholds
        self.min_accuracy_retention = 0.8  # Must retain 80% of base accuracy
        self.max_toxicity_increase = 0.1   # Max 10% increase in toxicity
        self.max_perplexity_increase = 1.5 # Max 50% increase in perplexity
        
    def validate_model_update(self, old_model, new_model) -> Tuple[bool, Dict[str, float]]:
        """
        Comprehensive safety validation for model updates.
        
        Returns:
            Tuple of (is_safe, metrics_dict)
        """
        metrics = {}
        
        try:
            # 1. Accuracy retention test
            accuracy_retention = self._test_accuracy_retention(old_model, new_model)
            metrics['accuracy_retention'] = accuracy_retention
            
            # 2. Perplexity change test
            perplexity_ratio = self._test_perplexity_change(old_model, new_model)
            metrics['perplexity_ratio'] = perplexity_ratio
            
            # 3. Basic reasoning capability test
            reasoning_score = self._test_reasoning_capability(new_model)
            metrics['reasoning_score'] = reasoning_score
            
            # 4. Known answer verification
            known_answer_accuracy = self._test_known_answers(new_model)
            metrics['known_answer_accuracy'] = known_answer_accuracy
            
            # Safety decision
            is_safe = (
                accuracy_retention >= self.min_accuracy_retention and
                perplexity_ratio <= self.max_perplexity_increase and
                reasoning_score >= 0.6 and
                known_answer_accuracy >= 0.7
            )
            
            logger.info(f"Safety validation: {'PASSED' if is_safe else 'FAILED'}")
            logger.info(f"Metrics: {metrics}")
            
            return is_safe, metrics
            
        except Exception as e:
            logger.error(f"Safety validation failed with error: {e}")
            return False, {"error": str(e)}
    
    def _test_accuracy_retention(self, old_model, new_model) -> float:
        """Test if new model retains accuracy on validation problems."""
        old_correct = 0
        new_correct = 0
        
        for problem in self.validation_problems[:10]:  # Use subset for speed
            question = problem.get('question', '')
            expected_answer = problem.get('answer', '')
            
            if not question or not expected_answer:
                continue
                
            # Test old model
            old_response = self._generate_answer(old_model, question)
            if self._is_correct_answer(old_response, expected_answer):
                old_correct += 1
                
            # Test new model
            new_response = self._generate_answer(new_model, question)
            if self._is_correct_answer(new_response, expected_answer):
                new_correct += 1
        
        if old_correct == 0:
            return 1.0  # Avoid division by zero
            
        return new_correct / old_correct
    
    def _test_perplexity_change(self, old_model, new_model) -> float:
        """Test perplexity change on validation text."""
        test_texts = [prob.get('question', '') for prob in self.validation_problems[:5]]
        test_texts = [t for t in test_texts if t]  # Remove empty strings
        
        if not test_texts:
            return 1.0
            
        old_perplexities = []
        new_perplexities = []
        
        for text in test_texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncate=True, max_length=512)
            
            with torch.no_grad():
                # Old model perplexity
                old_outputs = old_model(**inputs, labels=inputs['input_ids'])
                old_perplexities.append(torch.exp(old_outputs.loss).item())
                
                # New model perplexity
                new_outputs = new_model(**inputs, labels=inputs['input_ids'])
                new_perplexities.append(torch.exp(new_outputs.loss).item())
        
        avg_old = sum(old_perplexities) / len(old_perplexities)
        avg_new = sum(new_perplexities) / len(new_perplexities)
        
        return avg_new / avg_old if avg_old > 0 else 1.0
    
    def _test_reasoning_capability(self, model) -> float:
        """Test basic reasoning capabilities."""
        reasoning_prompts = [
            "If x + 2 = 5, then x = ",
            "The next number in the sequence 2, 4, 6, 8 is ",
            "If it takes 3 minutes to boil 1 egg, how long to boil 3 eggs? ",
        ]
        
        expected_answers = ["3", "10", "3"]
        correct = 0
        
        for prompt, expected in zip(reasoning_prompts, expected_answers):
            response = self._generate_answer(model, prompt)
            if expected in response:
                correct += 1
        
        return correct / len(reasoning_prompts)
    
    def _test_known_answers(self, model) -> float:
        """Test model on questions with known answers."""
        known_qa = [
            ("What is 2 + 2?", "4"),
            ("What is the capital of France?", "Paris"),
            ("How many days in a week?", "7"),
        ]
        
        correct = 0
        for question, expected in known_qa:
            response = self._generate_answer(model, question)
            if expected.lower() in response.lower():
                correct += 1
        
        return correct / len(known_qa)
    
    def _generate_answer(self, model, question: str, max_length: int = 100) -> str:
        """Generate answer from model for given question."""
        inputs = self.tokenizer(question, return_tensors="pt", truncate=True, max_length=256)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input question from response
        response = response[len(question):].strip()
        return response
    
    def _is_correct_answer(self, response: str, expected: str) -> bool:
        """Check if response contains the expected answer."""
        # Simple check - can be enhanced with more sophisticated matching
        return expected.lower() in response.lower()

class LoRASEALAdapter(SEALAdapter):
    """SEAL variant using LoRA adapters instead of full weight updates."""
    
    def __init__(self, model, tokenizer, lora_config: LoRAConfig, validation_problems: List[Dict[str, Any]]):
        super().__init__(model, tokenizer)
        self.lora_config = lora_config
        self.original_model = copy.deepcopy(model)  # Keep original for rollback
        self.lora_layers = {}
        self.safety_validator = SafetyValidator(model, tokenizer, validation_problems)
        self.adaptation_history = []
        
        self._initialize_lora_layers()
        
    def _initialize_lora_layers(self):
        """Initialize LoRA layers for target modules."""
        logger.info("Initializing LoRA layers...")
        
        target_modules = self.lora_config.target_modules
        lora_layers_added = 0
        
        def add_lora_to_layer(name: str, module: nn.Module):
            nonlocal lora_layers_added
            
            if isinstance(module, nn.Linear):
                # Check if this is a target module
                if any(target in name for target in target_modules):
                    lora_layer = LoRALayer(
                        original_layer=module,
                        rank=self.lora_config.rank,
                        alpha=self.lora_config.alpha,
                        dropout=self.lora_config.dropout
                    )
                    
                    # Replace the module
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent_module = dict(self.model.named_modules())[parent_name]
                        setattr(parent_module, child_name, lora_layer)
                    else:
                        # Top-level module
                        setattr(self.model, child_name, lora_layer)
                    
                    self.lora_layers[name] = lora_layer
                    lora_layers_added += 1
                    logger.debug(f"Added LoRA to {name}")
        
        # Apply to all target modules
        for name, module in self.model.named_modules():
            add_lora_to_layer(name, module)
        
        logger.info(f"Initialized {lora_layers_added} LoRA layers")
        
        # Freeze base model parameters
        self._freeze_base_model()
        
    def _freeze_base_model(self):
        """Freeze all non-LoRA parameters."""
        frozen_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        total_params = frozen_params + trainable_params
        logger.info(f"Frozen: {frozen_params:,} params ({frozen_params/total_params:.1%})")
        logger.info(f"Trainable: {trainable_params:,} params ({trainable_params/total_params:.1%})")
        
    def finetune_on_path(self, successful_path: ThoughtGraph) -> Dict[str, Any]:
        """
        Fine-tune only LoRA adapters on successful reasoning path.
        
        Args:
            successful_path: ThoughtGraph containing successful reasoning
            
        Returns:
            Dictionary containing adaptation results and metrics
        """
        logger.info("Starting LoRA-based fine-tuning on successful path")
        
        # Create model checkpoint before adaptation
        pre_adaptation_state = self._create_checkpoint()
        
        try:
            # Generate training data from successful path
            training_data = self._create_training_example_from_graph(successful_path)
            
            # Train only LoRA adapters
            training_result = self._train_lora_adapters(training_data)
            
            # Validate adapter safety
            is_safe, safety_metrics = self.safety_validator.validate_model_update(
                self.original_model, self.model
            )
            
            if not is_safe:
                logger.warning("Safety validation failed - rolling back adaptation")
                self._rollback_to_checkpoint(pre_adaptation_state)
                
                return {
                    'success': False,
                    'reason': 'safety_validation_failed',
                    'safety_metrics': safety_metrics,
                    'training_metrics': training_result
                }
            
            # Record successful adaptation
            adaptation_record = {
                'timestamp': torch.backends.cudnn.benchmark,  # Simple timestamp
                'training_metrics': training_result,
                'safety_metrics': safety_metrics,
                'graph_id': successful_path.question_id if hasattr(successful_path, 'question_id') else 'unknown'
            }
            self.adaptation_history.append(adaptation_record)
            
            logger.info("LoRA adaptation completed successfully")
            return {
                'success': True,
                'training_metrics': training_result,
                'safety_metrics': safety_metrics
            }
            
        except Exception as e:
            logger.error(f"LoRA adaptation failed: {e}")
            self._rollback_to_checkpoint(pre_adaptation_state)
            return {
                'success': False,
                'reason': 'training_failed',
                'error': str(e)
            }
    
    def _train_lora_adapters(self, training_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train only the LoRA adapter parameters."""
        self.model.train()
        
        # Optimizer for LoRA parameters only
        lora_params = []
        for lora_layer in self.lora_layers.values():
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
        
        optimizer = torch.optim.AdamW(lora_params, lr=1e-4, weight_decay=0.01)
        
        # Training loop
        input_ids = training_data['input_ids']
        attention_mask = training_data.get('attention_mask')
        labels = training_data['labels']
        
        num_epochs = 3  # Conservative number of epochs
        losses = []
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            losses.append(loss.item())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            
            logger.debug(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        
        self.model.eval()
        
        return {
            'final_loss': losses[-1],
            'initial_loss': losses[0],
            'num_epochs': num_epochs,
            'num_parameters_trained': sum(p.numel() for p in lora_params)
        }
    
    def _create_checkpoint(self) -> Dict[str, Any]:
        """Create checkpoint of current LoRA states."""
        checkpoint = {}
        for name, lora_layer in self.lora_layers.items():
            checkpoint[name] = {
                'lora_A': lora_layer.lora_A.clone(),
                'lora_B': lora_layer.lora_B.clone()
            }
        return checkpoint
    
    def _rollback_to_checkpoint(self, checkpoint: Dict[str, Any]):
        """Rollback LoRA parameters to checkpoint state."""
        logger.info("Rolling back LoRA adaptations to checkpoint")
        
        for name, state in checkpoint.items():
            if name in self.lora_layers:
                lora_layer = self.lora_layers[name]
                lora_layer.lora_A.data.copy_(state['lora_A'])
                lora_layer.lora_B.data.copy_(state['lora_B'])
        
        logger.info("Rollback completed")
    
    def reset_all_adaptations(self):
        """Reset all LoRA adaptations to initial state."""
        logger.info("Resetting all LoRA adaptations")
        
        for lora_layer in self.lora_layers.values():
            lora_layer.reset_parameters()
        
        self.adaptation_history.clear()
        logger.info("All adaptations reset")
    
    def merge_lora_weights(self):
        """Merge LoRA weights into base model for inference efficiency."""
        logger.info("Merging LoRA weights into base model")
        
        for lora_layer in self.lora_layers.values():
            lora_layer.merge_weights()
        
        logger.info("LoRA weights merged")
    
    def save_lora_state(self, path: str):
        """Save LoRA adapter states to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'lora_config': {
                'rank': self.lora_config.rank,
                'alpha': self.lora_config.alpha,
                'dropout': self.lora_config.dropout,
                'target_modules': self.lora_config.target_modules
            },
            'lora_states': {},
            'adaptation_history': self.adaptation_history
        }
        
        for name, lora_layer in self.lora_layers.items():
            state['lora_states'][name] = {
                'lora_A': lora_layer.lora_A.cpu(),
                'lora_B': lora_layer.lora_B.cpu()
            }
        
        torch.save(state, save_path)
        logger.info(f"LoRA state saved to {save_path}")
    
    def load_lora_state(self, path: str):
        """Load LoRA adapter states from disk."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"LoRA state file not found: {load_path}")
        
        state = torch.load(load_path)
        
        # Restore LoRA parameters
        for name, lora_state in state['lora_states'].items():
            if name in self.lora_layers:
                lora_layer = self.lora_layers[name]
                lora_layer.lora_A.data.copy_(lora_state['lora_A'])
                lora_layer.lora_B.data.copy_(lora_state['lora_B'])
        
        self.adaptation_history = state.get('adaptation_history', [])
        logger.info(f"LoRA state loaded from {load_path}")
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of all adaptations performed."""
        if not self.adaptation_history:
            return {'total_adaptations': 0}
        
        successful_adaptations = len(self.adaptation_history)
        avg_safety_score = sum(
            record['safety_metrics'].get('reasoning_score', 0) 
            for record in self.adaptation_history
        ) / successful_adaptations
        
        return {
            'total_adaptations': successful_adaptations,
            'average_safety_score': avg_safety_score,
            'latest_adaptation': self.adaptation_history[-1] if self.adaptation_history else None,
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }