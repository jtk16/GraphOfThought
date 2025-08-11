import re
import logging
from typing import Optional, Tuple, Any, Dict, List
from dataclasses import dataclass
import ast
import operator
import math

@dataclass
class EvaluationResult:
    """Result of evaluating a mathematical solution."""
    is_correct: bool
    extracted_answer: Optional[str]
    expected_answer: str
    confidence: float  # 0.0 to 1.0
    error_type: Optional[str] = None
    intermediate_steps: List[str] = None

class MathEvaluator:
    """
    Robust evaluator for mathematical reasoning problems.
    Handles answer extraction, numerical comparison, and algebraic validation.
    """
    
    def __init__(self):
        # Common answer patterns in mathematical reasoning
        self.answer_patterns = [
            r"Final Answer:\s*([^.]+)",
            r"The answer is\s*([^.]+)",
            r"Therefore,?\s*([^.]+)",
            r"So,?\s*([^.]+)",
            r"Answer:\s*([^.]+)",
            r"=\s*([^.]+?)(?:\s|$)",
            r"(\$?[\d,]+(?:\.\d+)?)(?:\s*dollars?|\$)?$"
        ]
        
        # Safe mathematical operations for evaluation
        self.safe_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub, 
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        
        # Safe mathematical functions
        self.safe_funcs = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'sqrt': math.sqrt,
            'pow': pow,
            'floor': math.floor,
            'ceil': math.ceil,
        }
        
        logging.info("MathEvaluator initialized with robust answer extraction")
    
    def extract_numerical_answer(self, text: str) -> Optional[str]:
        """
        Extract numerical answer from reasoning text using multiple strategies.
        """
        if not text:
            return None
        
        text = text.strip()
        
        # Strategy 1: Look for explicit answer markers
        for pattern in self.answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                candidate = match.group(1).strip()
                # Clean up the extracted answer
                candidate = self._clean_answer(candidate)
                if candidate:
                    return candidate
        
        # Strategy 2: Find the last number in the text (common in step-by-step solutions)
        numbers = re.findall(r'-?\$?[\d,]+(?:\.\d+)?', text)
        if numbers:
            last_number = numbers[-1].replace('$', '').replace(',', '')
            try:
                float(last_number)  # Validate it's a number
                return last_number
            except ValueError:
                pass
        
        # Strategy 3: Look for equals signs and extract what follows
        equals_matches = re.findall(r'=\s*(-?\$?[\d,]+(?:\.\d+)?)', text)
        if equals_matches:
            candidate = equals_matches[-1].replace('$', '').replace(',', '')
            try:
                float(candidate)
                return candidate
            except ValueError:
                pass
        
        return None
    
    def _clean_answer(self, answer: str) -> Optional[str]:
        """Clean and normalize extracted answer."""
        if not answer:
            return None
        
        # Remove common suffixes and prefixes
        answer = re.sub(r'[.\s]*$', '', answer)  # Remove trailing periods/spaces
        answer = re.sub(r'^[^\d-]*', '', answer)  # Remove non-numeric prefixes
        answer = answer.replace('$', '').replace(',', '').replace(' ', '')
        
        # Handle common representations
        if answer.lower() in ['zero', '0', 'none']:
            return '0'
        elif answer.lower() in ['one', '1']:
            return '1'
        elif answer.lower() in ['two', '2']:
            return '2'
        
        # Validate it's a number
        if re.match(r'^-?[\d]+(?:\.\d+)?$', answer):
            return answer
        
        return None
    
    def safe_eval(self, expression: str) -> Optional[float]:
        """
        Safely evaluate mathematical expressions.
        Only allows basic arithmetic operations.
        """
        try:
            # Parse the expression into AST
            node = ast.parse(expression, mode='eval')
            return self._eval_node(node.body)
        except Exception as e:
            logging.debug(f"Failed to evaluate expression '{expression}': {e}")
            return None
    
    def _eval_node(self, node):
        """Recursively evaluate AST nodes."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Legacy Python
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.safe_ops.get(type(node.op))
            if op and left is not None and right is not None:
                return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.safe_ops.get(type(node.op))
            if op and operand is not None:
                return op(operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name in self.safe_funcs:
                args = [self._eval_node(arg) for arg in node.args]
                if all(arg is not None for arg in args):
                    return self.safe_funcs[func_name](*args)
        
        return None
    
    def numerical_compare(self, answer1: str, answer2: str, tolerance: float = 1e-6) -> bool:
        """
        Compare two numerical answers with tolerance for floating point errors.
        """
        try:
            # Try direct numerical comparison first
            num1 = float(answer1)
            num2 = float(answer2)
            return abs(num1 - num2) <= tolerance
        except ValueError:
            # Fall back to string comparison if not parseable as numbers
            return self._normalize_answer(answer1) == self._normalize_answer(answer2)
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not answer:
            return ""
        
        # Remove common variations
        normalized = answer.lower().strip()
        normalized = re.sub(r'[^\w\d.-]', '', normalized)  # Remove punctuation except decimals
        
        # Handle fractions
        if '/' in normalized:
            try:
                parts = normalized.split('/')
                if len(parts) == 2:
                    num, denom = float(parts[0]), float(parts[1])
                    normalized = str(num / denom)
            except ValueError:
                pass
        
        return normalized
    
    def evaluate_gsm8k_answer(self, generated_solution: str, expected_answer: str) -> EvaluationResult:
        """
        Evaluate a GSM8K-style mathematical reasoning solution.
        
        Args:
            generated_solution: The model's complete solution text
            expected_answer: The expected numerical answer
            
        Returns:
            EvaluationResult with correctness and diagnostic information
        """
        # Extract the final answer from generated solution
        extracted = self.extract_numerical_answer(generated_solution)
        
        if extracted is None:
            return EvaluationResult(
                is_correct=False,
                extracted_answer=None,
                expected_answer=expected_answer,
                confidence=0.0,
                error_type="no_answer_found"
            )
        
        # Normalize expected answer
        expected_normalized = self._clean_answer(str(expected_answer))
        if expected_normalized is None:
            expected_normalized = str(expected_answer)
        
        # Compare answers
        is_correct = self.numerical_compare(extracted, expected_normalized)
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(generated_solution, extracted, is_correct)
        
        # Analyze solution quality
        intermediate_steps = self._extract_reasoning_steps(generated_solution)
        error_type = None if is_correct else self._diagnose_error(extracted, expected_normalized, generated_solution)
        
        return EvaluationResult(
            is_correct=is_correct,
            extracted_answer=extracted,
            expected_answer=expected_normalized,
            confidence=confidence,
            error_type=error_type,
            intermediate_steps=intermediate_steps
        )
    
    def _calculate_confidence(self, solution: str, extracted_answer: str, is_correct: bool) -> float:
        """Calculate confidence score based on solution quality indicators."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for clear answer markers
        if any(pattern.lower() in solution.lower() for pattern in ["final answer", "the answer is", "therefore"]):
            confidence += 0.2
        
        # Boost for multiple calculation steps
        calculation_patterns = len(re.findall(r'=\s*[\d,]+', solution))
        confidence += min(calculation_patterns * 0.1, 0.3)
        
        # Boost for clear reasoning structure
        if len(solution.split('\n')) >= 3:  # Multi-step reasoning
            confidence += 0.1
        
        # Penalty for very short or very long solutions
        word_count = len(solution.split())
        if word_count < 10:
            confidence -= 0.2
        elif word_count > 500:
            confidence -= 0.1
        
        # Adjust based on correctness
        if is_correct:
            confidence = min(confidence + 0.2, 1.0)
        else:
            confidence = max(confidence - 0.3, 0.1)
        
        return round(confidence, 2)
    
    def _extract_reasoning_steps(self, solution: str) -> List[str]:
        """Extract intermediate reasoning steps from solution."""
        steps = []
        lines = solution.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and any(indicator in line.lower() for indicator in ['step', 'first', 'then', 'next', 'finally', '=']):
                steps.append(line)
        
        return steps[:10]  # Limit to first 10 steps
    
    def _diagnose_error(self, extracted: str, expected: str, solution: str) -> str:
        """Diagnose the type of error in the solution."""
        try:
            extracted_num = float(extracted) if extracted else None
            expected_num = float(expected)
            
            if extracted_num is None:
                return "extraction_error"
            
            # Check for common error patterns
            ratio = extracted_num / expected_num if expected_num != 0 else float('inf')
            
            if abs(ratio - 2) < 0.1:
                return "double_counting"
            elif abs(ratio - 0.5) < 0.1:
                return "missed_factor"
            elif abs(extracted_num - expected_num) < 1:
                return "rounding_error"
            elif "%" in solution and abs(ratio - 100) < 10:
                return "percentage_confusion"
            else:
                return "calculation_error"
                
        except (ValueError, ZeroDivisionError):
            return "format_error"
    
    def batch_evaluate(self, solutions: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Evaluate multiple solutions and return aggregate statistics.
        
        Args:
            solutions: List of (generated_solution, expected_answer) tuples
            
        Returns:
            Dictionary with accuracy, error analysis, and detailed results
        """
        results = []
        error_types = {}
        total_confidence = 0.0
        
        for solution, expected in solutions:
            result = self.evaluate_gsm8k_answer(solution, expected)
            results.append(result)
            
            total_confidence += result.confidence
            
            if not result.is_correct and result.error_type:
                error_types[result.error_type] = error_types.get(result.error_type, 0) + 1
        
        correct_count = sum(1 for r in results if r.is_correct)
        total_count = len(results)
        
        return {
            "accuracy": correct_count / total_count if total_count > 0 else 0.0,
            "total_evaluated": total_count,
            "correct_count": correct_count,
            "average_confidence": total_confidence / total_count if total_count > 0 else 0.0,
            "error_type_distribution": error_types,
            "detailed_results": results
        }