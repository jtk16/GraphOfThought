import logging
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.kvtg.graph import ThoughtGraph, ThoughtNode

@dataclass
class NodeValue:
    """Represents the estimated value of a reasoning node."""
    quality_score: float  # 0.0 to 1.0
    progress_score: float  # How much progress towards solution
    confidence: float     # Confidence in the scoring
    features: Dict[str, float]  # Additional features used for scoring

class NodeEvaluator(ABC):
    """Abstract base class for evaluating reasoning node quality."""
    
    @abstractmethod
    def evaluate_node(self, node: ThoughtNode, context: str, question: str) -> NodeValue:
        """Evaluate the quality of a reasoning node."""
        pass

class HeuristicNodeEvaluator(NodeEvaluator):
    """
    Heuristic-based node evaluator for mathematical reasoning.
    Uses pattern matching and mathematical indicators to score nodes.
    """
    
    def __init__(self):
        # Patterns that indicate good reasoning steps
        self.positive_patterns = [
            r'\d+\s*[+\-*/]\s*\d+',  # Mathematical operations
            r'=\s*\d+',               # Equals with result
            r'total|sum|product|difference|quotient',  # Mathematical concepts
            r'first|then|next|finally|step',  # Step indicators
            r'let me|i need to|we have',      # Problem-solving language
            r'therefore|so|thus|hence',       # Conclusion words
        ]
        
        # Patterns that indicate poor reasoning
        self.negative_patterns = [
            r'i don\'t know|not sure|unclear',  # Uncertainty
            r'maybe|perhaps|possibly',           # Hedging
            r'um|uh|well|hmm',                  # Hesitation
            r'^[^a-zA-Z0-9]*$',                 # Empty or just punctuation
        ]
        
        # Mathematical operation patterns
        self.math_operations = [
            r'\d+\s*\+\s*\d+\s*=\s*\d+',  # Addition
            r'\d+\s*-\s*\d+\s*=\s*\d+',   # Subtraction
            r'\d+\s*\*\s*\d+\s*=\s*\d+',  # Multiplication
            r'\d+\s*/\s*\d+\s*=\s*\d+',   # Division
        ]
        
        logging.info("HeuristicNodeEvaluator initialized with mathematical reasoning patterns")
    
    def evaluate_node(self, node: ThoughtNode, context: str, question: str) -> NodeValue:
        """Evaluate node quality using heuristic patterns."""
        text = node.text.lower().strip()
        
        if not text:
            return NodeValue(0.0, 0.0, 1.0, {"empty": 1.0})
        
        features = {}
        quality_score = 0.5  # Neutral starting point
        
        # Check for positive patterns
        positive_matches = 0
        for pattern in self.positive_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 0:
                positive_matches += matches
                features[f"positive_{pattern[:10]}"] = matches
        
        # Check for negative patterns
        negative_matches = 0
        for pattern in self.negative_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 0:
                negative_matches += matches
                features[f"negative_{pattern[:10]}"] = matches
        
        # Mathematical operation bonus
        math_operations = 0
        for pattern in self.math_operations:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            math_operations += matches
        
        features["math_operations"] = math_operations
        features["positive_matches"] = positive_matches
        features["negative_matches"] = negative_matches
        
        # Calculate quality score
        quality_score += positive_matches * 0.1
        quality_score += math_operations * 0.2
        quality_score -= negative_matches * 0.15
        
        # Text length normalization (not too short, not too long)
        text_length = len(text.split())
        if 5 <= text_length <= 50:
            quality_score += 0.1
        elif text_length < 3:
            quality_score -= 0.2
        elif text_length > 100:
            quality_score -= 0.1
        
        features["text_length"] = text_length
        
        # Progress estimation based on content
        progress_score = self._estimate_progress(text, question)
        features["estimated_progress"] = progress_score
        
        # Confidence based on pattern matches
        confidence = min(0.9, 0.3 + (positive_matches + math_operations) * 0.1)
        
        # Clamp quality score
        quality_score = max(0.0, min(1.0, quality_score))
        
        return NodeValue(quality_score, progress_score, confidence, features)
    
    def _estimate_progress(self, text: str, question: str) -> float:
        """Estimate how much progress this step makes toward solving the problem."""
        progress = 0.0
        
        # Look for numerical calculations
        if re.search(r'\d+\s*[+\-*/]\s*\d+\s*=\s*\d+', text):
            progress += 0.3
        
        # Look for final answer patterns
        if re.search(r'final answer|the answer is|therefore.*\d+', text, re.IGNORECASE):
            progress += 0.5
        
        # Look for problem setup
        if any(word in text for word in ['given', 'have', 'start with', 'initially']):
            progress += 0.2
        
        # Look for intermediate steps
        if any(word in text for word in ['next', 'then', 'step', 'now']):
            progress += 0.1
        
        return min(1.0, progress)

class ValueGuidedController:
    """
    Enhanced KVTG controller with value-guided exploration.
    Uses node evaluation to prioritize promising reasoning paths.
    """
    
    def __init__(self, base_controller, node_evaluator: NodeEvaluator, 
                 exploration_temperature: float = 1.0, value_weight: float = 0.5):
        self.base_controller = base_controller
        self.node_evaluator = node_evaluator
        self.exploration_temperature = exploration_temperature
        self.value_weight = value_weight
        self.node_values: Dict[str, NodeValue] = {}
        
        # Delegate properties from base controller
        self.model = base_controller.model
        self.tokenizer = base_controller.tokenizer
        self.device = base_controller.device
        self.storage = base_controller.storage
        self.exploration_budget = base_controller.exploration_budget
        self.beam_width = base_controller.beam_width
        
        logging.info(f"ValueGuidedController initialized with temp={exploration_temperature}, weight={value_weight}")
    
    def evaluate_and_store_node(self, node: ThoughtNode, graph: ThoughtGraph) -> NodeValue:
        """Evaluate a node and store its value."""
        context = self._get_node_context(node, graph)
        node_value = self.node_evaluator.evaluate_node(node, context, graph.question)
        self.node_values[node.id] = node_value
        return node_value
    
    def _get_node_context(self, node: ThoughtNode, graph: ThoughtGraph) -> str:
        """Get the reasoning context leading to this node."""
        # Traverse backward to get full reasoning path
        path = []
        current_id = node.id
        
        # Simple backward traversal (assumes single parent per node)
        while current_id != "0":
            current_node = graph.get_node(current_id)
            if not current_node:
                break
            path.insert(0, current_node.text)
            
            # Find parent
            parent_edge = next((e for e in graph.edges if e.target == current_id), None)
            if not parent_edge:
                break
            current_id = parent_edge.source
        
        return " ".join(path)
    
    def _select_nodes_to_expand(self, graph: ThoughtGraph) -> List[ThoughtNode]:
        """Select nodes using value-guided strategy."""
        leaf_nodes = graph.get_leaf_nodes()
        
        if not leaf_nodes:
            return []
        
        # Evaluate all leaf nodes if not already done
        for node in leaf_nodes:
            if node.id not in self.node_values:
                self.evaluate_and_store_node(node, graph)
        
        # Get values and apply temperature-based selection
        node_scores = []
        for node in leaf_nodes:
            value = self.node_values.get(node.id)
            if value:
                # Combine quality and progress with value weight
                combined_score = (self.value_weight * value.quality_score + 
                                (1 - self.value_weight) * value.progress_score)
                node_scores.append((node, combined_score))
        
        if not node_scores:
            return leaf_nodes[:3]  # Fallback to first few nodes
        
        # Temperature-based selection
        if self.exploration_temperature > 0:
            scores = [score / self.exploration_temperature for _, score in node_scores]
            probs = F.softmax(torch.tensor(scores), dim=0)
            
            # Sample based on probabilities, but ensure we get at least top candidates
            sorted_nodes = sorted(node_scores, key=lambda x: x[1], reverse=True)
            
            # Take top 50% deterministically, sample rest
            deterministic_count = max(1, len(sorted_nodes) // 2)
            selected = [node for node, _ in sorted_nodes[:deterministic_count]]
            
            # Add some stochastic exploration
            remaining = [node for node, _ in sorted_nodes[deterministic_count:]]
            if remaining and len(selected) < 3:
                # Sample one more from remaining
                remaining_probs = probs[deterministic_count:]
                if len(remaining_probs) > 0:
                    idx = torch.multinomial(remaining_probs, 1).item()
                    selected.append(remaining[idx])
            
            return selected[:3]  # Limit to top 3
        
        else:
            # Greedy selection
            sorted_nodes = sorted(node_scores, key=lambda x: x[1], reverse=True)
            return [node for node, _ in sorted_nodes[:3]]
    
    def explore(self, question: str):
        """Main exploration method that delegates to base controller but uses guided selection."""
        # Temporarily override the base controller's selection method
        original_method = self.base_controller._select_nodes_to_expand
        self.base_controller._select_nodes_to_expand = self._select_nodes_to_expand
        
        try:
            return self.base_controller.explore(question)
        finally:
            # Restore original method
            self.base_controller._select_nodes_to_expand = original_method

class CurriculumManager:
    """
    Manages curriculum learning for mathematical reasoning.
    Progressively increases problem difficulty based on success rate.
    """
    
    def __init__(self, initial_difficulty: float = 0.3, adaptation_rate: float = 0.1):
        self.current_difficulty = initial_difficulty
        self.adaptation_rate = adaptation_rate
        self.success_history = []
        self.difficulty_levels = {
            "very_easy": 0.2,    # Single-step arithmetic
            "easy": 0.4,         # Two-step problems
            "medium": 0.6,       # Multi-step with intermediate calculations
            "hard": 0.8,         # Complex word problems
            "very_hard": 1.0     # Advanced reasoning required
        }
        
        logging.info(f"CurriculumManager initialized at difficulty {initial_difficulty}")
    
    def record_attempt(self, success: bool, problem_difficulty: float = None):
        """Record the result of a problem attempt."""
        self.success_history.append(success)
        
        # Keep only recent history
        if len(self.success_history) > 20:
            self.success_history = self.success_history[-20:]
    
    def should_increase_difficulty(self) -> bool:
        """Determine if we should increase problem difficulty."""
        if len(self.success_history) < 5:
            return False
        
        recent_success_rate = sum(self.success_history[-10:]) / min(10, len(self.success_history))
        return recent_success_rate > 0.7  # 70% success rate threshold
    
    def should_decrease_difficulty(self) -> bool:
        """Determine if we should decrease problem difficulty."""
        if len(self.success_history) < 5:
            return False
        
        recent_success_rate = sum(self.success_history[-10:]) / min(10, len(self.success_history))
        return recent_success_rate < 0.3  # 30% success rate threshold
    
    def adapt_difficulty(self):
        """Adapt the current difficulty based on performance."""
        if self.should_increase_difficulty() and self.current_difficulty < 1.0:
            old_difficulty = self.current_difficulty
            self.current_difficulty = min(1.0, self.current_difficulty + self.adaptation_rate)
            logging.info(f"📈 Increased difficulty: {old_difficulty:.2f} → {self.current_difficulty:.2f}")
        
        elif self.should_decrease_difficulty() and self.current_difficulty > 0.1:
            old_difficulty = self.current_difficulty
            self.current_difficulty = max(0.1, self.current_difficulty - self.adaptation_rate)
            logging.info(f"📉 Decreased difficulty: {old_difficulty:.2f} → {self.current_difficulty:.2f}")
    
    def get_difficulty_level_name(self) -> str:
        """Get human-readable difficulty level name."""
        for name, threshold in sorted(self.difficulty_levels.items(), key=lambda x: x[1]):
            if self.current_difficulty <= threshold:
                return name
        return "very_hard"
    
    def filter_problems_by_difficulty(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter problems to match current difficulty level."""
        # Simple heuristic based on problem characteristics
        filtered = []
        target_count = max(5, int(len(problems) * 0.3))  # At least 5 problems
        
        for problem in problems:
            problem_difficulty = self._estimate_problem_difficulty(problem)
            difficulty_diff = abs(problem_difficulty - self.current_difficulty)
            
            if difficulty_diff <= 0.2:  # Within tolerance
                filtered.append(problem)
            
            if len(filtered) >= target_count:
                break
        
        # If we don't have enough problems at current difficulty, expand tolerance
        if len(filtered) < 3:
            tolerance = 0.3
            for problem in problems:
                if len(filtered) >= target_count:
                    break
                problem_difficulty = self._estimate_problem_difficulty(problem)
                difficulty_diff = abs(problem_difficulty - self.current_difficulty)
                
                if difficulty_diff <= tolerance and problem not in filtered:
                    filtered.append(problem)
        
        return filtered[:target_count]
    
    def _estimate_problem_difficulty(self, problem: Dict[str, Any]) -> float:
        """Estimate problem difficulty based on characteristics."""
        question = problem.get('question', '').lower()
        answer = str(problem.get('answer', ''))
        
        difficulty = 0.3  # Base difficulty
        
        # Number complexity
        numbers = re.findall(r'\d+(?:\.\d+)?', question)
        if numbers:
            max_num = max(float(n) for n in numbers)
            if max_num > 100:
                difficulty += 0.2
            elif max_num > 1000:
                difficulty += 0.3
        
        # Operation complexity
        operations = len(re.findall(r'[+\-*/]', question))
        difficulty += operations * 0.1
        
        # Word problem indicators
        word_indicators = ['total', 'remaining', 'difference', 'fraction', 'percentage', 'ratio']
        word_count = sum(1 for word in word_indicators if word in question)
        difficulty += word_count * 0.1
        
        # Multiple steps indicated
        step_indicators = ['first', 'then', 'next', 'finally', 'after']
        step_count = sum(1 for word in step_indicators if word in question)
        difficulty += step_count * 0.15
        
        # Answer complexity
        try:
            answer_num = float(answer)
            if answer_num > 1000:
                difficulty += 0.1
            elif '/' in answer or '.' in answer:  # Fractions or decimals
                difficulty += 0.15
        except ValueError:
            difficulty += 0.2  # Non-numeric answers are typically harder
        
        return min(1.0, difficulty)

class DemonstrationLearning:
    """
    Learns from high-quality reasoning demonstrations to guide initial exploration.
    """
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.demonstrations = []
        self.pattern_templates = []
        
        # Common mathematical reasoning templates
        self.default_templates = [
            "Let me identify what we know: {given_info}",
            "I need to find: {target}",
            "First, I'll calculate: {step_1}",
            "Then, I'll compute: {step_2}",
            "Therefore, the answer is: {result}"
        ]
        
        logging.info("DemonstrationLearning initialized with default templates")
    
    def add_demonstration(self, problem: str, solution_path: List[str], is_correct: bool):
        """Add a demonstration of problem-solving."""
        if is_correct:
            self.demonstrations.append({
                'problem': problem,
                'path': solution_path,
                'quality': 1.0
            })
            
            # Extract patterns from successful demonstrations
            self._extract_patterns(problem, solution_path)
    
    def _extract_patterns(self, problem: str, solution_path: List[str]):
        """Extract reusable patterns from successful demonstrations."""
        # Simple pattern extraction - look for common structures
        for step in solution_path:
            step_lower = step.lower()
            
            # Mathematical operation patterns
            if re.search(r'\d+\s*[+\-*/]\s*\d+\s*=\s*\d+', step):
                pattern = re.sub(r'\d+', '{number}', step)
                if pattern not in self.pattern_templates:
                    self.pattern_templates.append(pattern)
            
            # Problem setup patterns
            if any(indicator in step_lower for indicator in ['we have', 'given', 'initially']):
                pattern = re.sub(r'\d+', '{number}', step)
                if pattern not in self.pattern_templates:
                    self.pattern_templates.append(pattern)
    
    def generate_guided_prompt(self, problem: str, current_path: List[str]) -> str:
        """Generate a guided prompt based on learned demonstrations."""
        base_prompt = f"Question: {problem}\n\nReasoning Path:\n"
        base_prompt += "\n".join(current_path)
        base_prompt += "\n\nNext Step:"
        
        # Add guidance based on current context
        if len(current_path) == 0:
            # Starting step - suggest problem analysis
            guidance = " (Start by identifying what information is given and what we need to find)"
        elif len(current_path) == 1:
            # Second step - suggest calculation approach
            guidance = " (Consider what mathematical operations are needed)"
        else:
            # Later steps - suggest solution consolidation
            guidance = " (Work towards the final answer)"
        
        return base_prompt + guidance
    
    def get_step_suggestions(self, problem: str, current_context: str) -> List[str]:
        """Get suggested next steps based on demonstrations."""
        suggestions = []
        
        # Match current context to demonstration patterns
        for template in self.pattern_templates[:5]:  # Top 5 patterns
            if len(suggestions) < 3:
                suggestions.append(template.format(number="<calculate>"))
        
        # Add default suggestions if we don't have enough patterns
        if len(suggestions) < 3:
            remaining = 3 - len(suggestions)
            suggestions.extend(self.default_templates[:remaining])
        
        return suggestions

def create_guided_exploration_system(base_controller, difficulty: float = 0.3) -> Tuple[ValueGuidedController, CurriculumManager, DemonstrationLearning]:
    """
    Factory function to create a complete guided exploration system.
    
    Returns:
        Tuple of (guided_controller, curriculum_manager, demonstration_learning)
    """
    # Create components
    node_evaluator = HeuristicNodeEvaluator()
    
    guided_controller = ValueGuidedController(
        base_controller=base_controller,
        node_evaluator=node_evaluator,
        exploration_temperature=0.8,  # Slightly stochastic
        value_weight=0.6  # Favor quality over progress initially
    )
    
    curriculum_manager = CurriculumManager(
        initial_difficulty=difficulty,
        adaptation_rate=0.1
    )
    
    demonstration_learning = DemonstrationLearning(
        tokenizer=base_controller.tokenizer
    )
    
    logging.info("🎯 Guided exploration system created successfully")
    
    return guided_controller, curriculum_manager, demonstration_learning