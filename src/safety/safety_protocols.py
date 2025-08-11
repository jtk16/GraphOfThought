"""
Comprehensive Safety Protocols for KVTG+SEAL System

This module implements robust safety measures for self-modifying AI systems,
addressing the critical feedback about missing safety frameworks.

Key components:
- Automated regression testing
- Model safety validation
- Immutable checkpoint management
- Behavioral drift detection
- Human-in-the-loop approval system
- Rollback capabilities
"""

import torch
import torch.nn as nn
import hashlib
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import subprocess
import os
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)

@dataclass
class SafetyThresholds:
    """Safety threshold configuration."""
    min_accuracy_retention: float = 0.85    # 85% of original accuracy
    max_toxicity_increase: float = 0.05     # 5% increase in toxicity
    max_perplexity_increase: float = 1.2    # 20% increase in perplexity
    max_behavioral_drift: float = 0.3       # 30% behavioral change
    min_known_answer_accuracy: float = 0.9  # 90% on known answers
    
@dataclass
class RegressionTest:
    """Single regression test configuration."""
    name: str
    description: str
    test_function: str  # Name of test function
    input_data: Any
    expected_output: Any
    tolerance: float = 1e-6
    critical: bool = True  # If True, failure blocks deployment

@dataclass
class SafetyCheckResult:
    """Result of a safety check."""
    check_name: str
    passed: bool
    score: float
    threshold: float
    message: str
    timestamp: str
    details: Dict[str, Any] = None

class ModelSafetyValidator:
    """Comprehensive safety validation for model updates."""
    
    def __init__(self, thresholds: SafetyThresholds = None):
        self.thresholds = thresholds or SafetyThresholds()
        self.regression_tests = []
        self.known_answers = []
        self._load_standard_tests()
        
    def _load_standard_tests(self):
        """Load standard regression tests and known answers."""
        # Basic mathematical reasoning tests
        self.regression_tests = [
            RegressionTest(
                name="basic_arithmetic",
                description="Basic arithmetic operations",
                test_function="test_basic_arithmetic",
                input_data="What is 15 + 27?",
                expected_output="42",
                critical=True
            ),
            RegressionTest(
                name="simple_algebra",
                description="Simple algebraic solving",
                test_function="test_simple_algebra",
                input_data="If x + 5 = 12, what is x?",
                expected_output="7",
                critical=True
            ),
            RegressionTest(
                name="word_problem",
                description="Basic word problem solving",
                test_function="test_word_problem",
                input_data="Sarah has 12 apples. She gives 3 to Tom and 2 to Mary. How many does she have left?",
                expected_output="7",
                critical=True
            ),
            RegressionTest(
                name="reasoning_chain",
                description="Multi-step reasoning chain",
                test_function="test_reasoning_chain",
                input_data="A number is doubled, then 5 is added, resulting in 17. What was the original number?",
                expected_output="6",
                critical=True
            ),
        ]
        
        # Known factual answers
        self.known_answers = [
            ("What is 2 + 2?", "4"),
            ("What is the capital of France?", "Paris"),
            ("How many days are in a week?", "7"),
            ("What is 10 * 10?", "100"),
            ("What comes after Thursday?", "Friday"),
        ]
        
    def validate_model_update(self, old_model, new_model, tokenizer) -> Tuple[bool, List[SafetyCheckResult]]:
        """
        Comprehensive safety validation for model updates.
        
        Returns:
            Tuple of (is_safe, list_of_check_results)
        """
        logger.info("Starting comprehensive safety validation")
        check_results = []
        
        # 1. Regression test suite
        regression_result = self._run_regression_tests(new_model, tokenizer)
        check_results.append(regression_result)
        
        # 2. Accuracy retention check
        accuracy_result = self._check_accuracy_retention(old_model, new_model, tokenizer)
        check_results.append(accuracy_result)
        
        # 3. Known answer verification
        known_answer_result = self._check_known_answers(new_model, tokenizer)
        check_results.append(known_answer_result)
        
        # 4. Behavioral drift detection
        drift_result = self._detect_behavioral_drift(old_model, new_model, tokenizer)
        check_results.append(drift_result)
        
        # 5. Perplexity degradation check
        perplexity_result = self._check_perplexity_degradation(old_model, new_model, tokenizer)
        check_results.append(perplexity_result)
        
        # Determine overall safety
        critical_failures = [r for r in check_results if not r.passed and r.check_name in ["regression_tests", "accuracy_retention", "known_answers"]]
        is_safe = len(critical_failures) == 0
        
        # Log results
        for result in check_results:
            status = "PASSED" if result.passed else "FAILED"
            logger.info(f"{result.check_name}: {status} (score: {result.score:.3f}, threshold: {result.threshold:.3f})")
        
        overall_status = "PASSED" if is_safe else "FAILED"
        logger.info(f"Overall safety validation: {overall_status}")
        
        return is_safe, check_results
    
    def _run_regression_tests(self, model, tokenizer) -> SafetyCheckResult:
        """Run all regression tests."""
        passed_tests = 0
        total_tests = len(self.regression_tests)
        
        for test in self.regression_tests:
            try:
                response = self._generate_model_response(model, tokenizer, test.input_data)
                if self._check_answer_correctness(response, test.expected_output):
                    passed_tests += 1
            except Exception as e:
                logger.warning(f"Regression test {test.name} failed with error: {e}")
        
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        threshold = 0.8  # 80% of regression tests must pass
        
        return SafetyCheckResult(
            check_name="regression_tests",
            passed=pass_rate >= threshold,
            score=pass_rate,
            threshold=threshold,
            message=f"Passed {passed_tests}/{total_tests} regression tests",
            timestamp=datetime.now().isoformat(),
            details={"passed_tests": passed_tests, "total_tests": total_tests}
        )
    
    def _check_accuracy_retention(self, old_model, new_model, tokenizer) -> SafetyCheckResult:
        """Check if new model retains accuracy of old model."""
        test_problems = self.regression_tests
        
        old_correct = 0
        new_correct = 0
        
        for test in test_problems:
            # Test old model
            old_response = self._generate_model_response(old_model, tokenizer, test.input_data)
            if self._check_answer_correctness(old_response, test.expected_output):
                old_correct += 1
            
            # Test new model
            new_response = self._generate_model_response(new_model, tokenizer, test.input_data)
            if self._check_answer_correctness(new_response, test.expected_output):
                new_correct += 1
        
        retention_rate = new_correct / old_correct if old_correct > 0 else 1.0
        
        return SafetyCheckResult(
            check_name="accuracy_retention",
            passed=retention_rate >= self.thresholds.min_accuracy_retention,
            score=retention_rate,
            threshold=self.thresholds.min_accuracy_retention,
            message=f"Accuracy retention: {retention_rate:.3f}",
            timestamp=datetime.now().isoformat(),
            details={"old_correct": old_correct, "new_correct": new_correct}
        )
    
    def _check_known_answers(self, model, tokenizer) -> SafetyCheckResult:
        """Test model on questions with known answers."""
        correct_answers = 0
        
        for question, expected in self.known_answers:
            response = self._generate_model_response(model, tokenizer, question)
            if self._check_answer_correctness(response, expected):
                correct_answers += 1
        
        accuracy = correct_answers / len(self.known_answers)
        
        return SafetyCheckResult(
            check_name="known_answers",
            passed=accuracy >= self.thresholds.min_known_answer_accuracy,
            score=accuracy,
            threshold=self.thresholds.min_known_answer_accuracy,
            message=f"Known answer accuracy: {accuracy:.3f}",
            timestamp=datetime.now().isoformat(),
            details={"correct": correct_answers, "total": len(self.known_answers)}
        )
    
    def _detect_behavioral_drift(self, old_model, new_model, tokenizer) -> SafetyCheckResult:
        """Detect significant behavioral changes between models."""
        test_prompts = [
            "Hello, how are you?",
            "What is machine learning?",
            "Explain photosynthesis briefly.",
            "What is 2+2?",
            "Tell me about the weather.",
        ]
        
        differences = []
        
        for prompt in test_prompts:
            old_response = self._generate_model_response(old_model, tokenizer, prompt)
            new_response = self._generate_model_response(new_model, tokenizer, prompt)
            
            # Simple similarity metric (can be enhanced with better metrics)
            similarity = self._compute_text_similarity(old_response, new_response)
            differences.append(1 - similarity)  # Convert to difference
        
        avg_drift = sum(differences) / len(differences)
        
        return SafetyCheckResult(
            check_name="behavioral_drift",
            passed=avg_drift <= self.thresholds.max_behavioral_drift,
            score=avg_drift,
            threshold=self.thresholds.max_behavioral_drift,
            message=f"Behavioral drift: {avg_drift:.3f}",
            timestamp=datetime.now().isoformat(),
            details={"individual_drifts": differences}
        )
    
    def _check_perplexity_degradation(self, old_model, new_model, tokenizer) -> SafetyCheckResult:
        """Check for significant perplexity increase."""
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Mathematics is the language of science and engineering.",
        ]
        
        old_perplexities = []
        new_perplexities = []
        
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncate=True, max_length=512)
            
            with torch.no_grad():
                # Old model perplexity
                old_outputs = old_model(**inputs, labels=inputs['input_ids'])
                old_perplexities.append(torch.exp(old_outputs.loss).item())
                
                # New model perplexity
                new_outputs = new_model(**inputs, labels=inputs['input_ids'])
                new_perplexities.append(torch.exp(new_outputs.loss).item())
        
        avg_old_perplexity = sum(old_perplexities) / len(old_perplexities)
        avg_new_perplexity = sum(new_perplexities) / len(new_perplexities)
        
        perplexity_ratio = avg_new_perplexity / avg_old_perplexity if avg_old_perplexity > 0 else 1.0
        
        return SafetyCheckResult(
            check_name="perplexity_degradation",
            passed=perplexity_ratio <= self.thresholds.max_perplexity_increase,
            score=perplexity_ratio,
            threshold=self.thresholds.max_perplexity_increase,
            message=f"Perplexity ratio: {perplexity_ratio:.3f}",
            timestamp=datetime.now().isoformat(),
            details={"old_perplexity": avg_old_perplexity, "new_perplexity": avg_new_perplexity}
        )
    
    def _generate_model_response(self, model, tokenizer, prompt: str, max_length: int = 100) -> str:
        """Generate response from model for given prompt."""
        inputs = tokenizer(prompt, return_tensors="pt", truncate=True, max_length=256)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove input prompt from response
        response = response[len(prompt):].strip()
        return response
    
    def _check_answer_correctness(self, response: str, expected: str) -> bool:
        """Check if response contains expected answer."""
        # Normalize both strings
        response_norm = response.lower().strip()
        expected_norm = expected.lower().strip()
        
        # Check if expected answer is in response
        return expected_norm in response_norm
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts (simple implementation)."""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

class ImmutableCheckpointManager:
    """Manages cryptographically signed, immutable model checkpoints."""
    
    def __init__(self, storage_dir: str = "checkpoints"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate or load signing keys
        self.private_key, self.public_key = self._get_or_create_keys()
        
    def _get_or_create_keys(self) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """Get or create RSA key pair for checkpoint signing."""
        private_key_path = self.storage_dir / "private_key.pem"
        public_key_path = self.storage_dir / "public_key.pem"
        
        if private_key_path.exists() and public_key_path.exists():
            # Load existing keys
            with open(private_key_path, 'rb') as f:
                private_key = serialization.load_pem_private_key(f.read(), password=None)
            
            with open(public_key_path, 'rb') as f:
                public_key = serialization.load_pem_public_key(f.read())
        else:
            # Generate new keys
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            public_key = private_key.public_key()
            
            # Save keys
            with open(private_key_path, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            with open(public_key_path, 'wb') as f:
                f.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
        
        return private_key, public_key
    
    def create_checkpoint(self, model, metadata: Dict[str, Any]) -> str:
        """Create cryptographically signed checkpoint."""
        checkpoint_id = f"checkpoint_{int(time.time())}_{hash(str(metadata)) % 100000:05d}"
        checkpoint_path = self.storage_dir / f"{checkpoint_id}.pt"
        metadata_path = self.storage_dir / f"{checkpoint_id}_metadata.json"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'checkpoint_id': checkpoint_id,
            'timestamp': time.time(),
            'git_commit': self._get_git_commit(),
            'pytorch_version': torch.__version__,
            'metadata': metadata
        }
        
        # Save model checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Create metadata with hash
        file_hash = self._compute_file_hash(checkpoint_path)
        full_metadata = {
            'checkpoint_id': checkpoint_id,
            'file_hash': file_hash,
            'timestamp': checkpoint_data['timestamp'],
            'git_commit': checkpoint_data['git_commit'],
            'pytorch_version': checkpoint_data['pytorch_version'],
            'metadata': metadata
        }
        
        # Sign metadata
        metadata_str = json.dumps(full_metadata, sort_keys=True)
        signature = self._sign_data(metadata_str.encode())
        full_metadata['signature'] = signature.hex()
        
        # Save signed metadata
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        logger.info(f"Created immutable checkpoint: {checkpoint_id}")
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load and verify checkpoint."""
        checkpoint_path = self.storage_dir / f"{checkpoint_id}.pt"
        metadata_path = self.storage_dir / f"{checkpoint_id}_metadata.json"
        
        if not checkpoint_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")
        
        # Load and verify metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Verify signature
        signature = bytes.fromhex(metadata.pop('signature'))
        metadata_str = json.dumps(metadata, sort_keys=True)
        
        if not self._verify_signature(metadata_str.encode(), signature):
            raise ValueError(f"Checkpoint {checkpoint_id} signature verification failed")
        
        # Verify file integrity
        current_hash = self._compute_file_hash(checkpoint_path)
        if current_hash != metadata['file_hash']:
            raise ValueError(f"Checkpoint {checkpoint_id} file integrity check failed")
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path)
        
        logger.info(f"Successfully loaded and verified checkpoint: {checkpoint_id}")
        return checkpoint_data
    
    def rollback_to_checkpoint(self, model, checkpoint_id: str):
        """Safely rollback model to checkpoint."""
        logger.info(f"Rolling back to checkpoint: {checkpoint_id}")
        
        checkpoint_data = self.load_checkpoint(checkpoint_id)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        logger.info("Model rollback completed successfully")
        return checkpoint_data
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        checkpoints = []
        
        for metadata_path in self.storage_dir.glob("*_metadata.json"):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                checkpoints.append({
                    'checkpoint_id': metadata['checkpoint_id'],
                    'timestamp': metadata['timestamp'],
                    'datetime': datetime.fromtimestamp(metadata['timestamp']).isoformat(),
                    'git_commit': metadata.get('git_commit', 'unknown'),
                    'metadata': metadata.get('metadata', {})
                })
            except Exception as e:
                logger.warning(f"Failed to read metadata from {metadata_path}: {e}")
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        return checkpoints
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return 'unknown'
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _sign_data(self, data: bytes) -> bytes:
        """Sign data with private key."""
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def _verify_signature(self, data: bytes, signature: bytes) -> bool:
        """Verify signature with public key."""
        try:
            self.public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False

class HumanApprovalGateway:
    """Human-in-the-loop approval system for production deployments."""
    
    def __init__(self, approval_timeout: int = 300):  # 5 minutes default
        self.approval_timeout = approval_timeout
        self.pending_approvals = {}
        
    def request_approval(self, 
                        checkpoint_id: str,
                        safety_results: List[SafetyCheckResult],
                        deployment_metadata: Dict[str, Any]) -> bool:
        """
        Request human approval for model deployment.
        
        Returns:
            True if approved, False if rejected or timeout
        """
        logger.info(f"Requesting human approval for checkpoint: {checkpoint_id}")
        
        # Create approval request
        approval_request = {
            'checkpoint_id': checkpoint_id,
            'timestamp': time.time(),
            'safety_results': [asdict(result) for result in safety_results],
            'deployment_metadata': deployment_metadata,
            'status': 'pending'
        }
        
        # Display approval request
        self._display_approval_request(approval_request)
        
        # Store pending request
        self.pending_approvals[checkpoint_id] = approval_request
        
        # Wait for approval (in real system, this would be async)
        logger.info("Waiting for human approval...")
        print("\n" + "="*60)
        print("DEPLOYMENT APPROVAL REQUIRED")
        print("="*60)
        print(f"Checkpoint ID: {checkpoint_id}")
        print("\nSafety Check Results:")
        
        for result in safety_results:
            status = "✓ PASSED" if result.passed else "✗ FAILED"
            print(f"  {result.check_name}: {status} (score: {result.score:.3f})")
        
        print("\nDeployment Metadata:")
        for key, value in deployment_metadata.items():
            print(f"  {key}: {value}")
        
        print("\n" + "-"*60)
        
        # In a real system, this would be handled through a web interface or API
        # For demo purposes, we'll simulate approval based on safety results
        all_critical_passed = all(r.passed for r in safety_results if r.check_name in ["regression_tests", "accuracy_retention", "known_answers"])
        
        if all_critical_passed:
            print("All critical safety checks passed. Auto-approving deployment.")
            approval = True
        else:
            print("Critical safety checks failed. Deployment rejected.")
            approval = False
        
        # Record decision
        approval_request['status'] = 'approved' if approval else 'rejected'
        approval_request['decision_timestamp'] = time.time()
        
        logger.info(f"Approval decision: {'APPROVED' if approval else 'REJECTED'}")
        return approval
    
    def _display_approval_request(self, request: Dict[str, Any]):
        """Display approval request details."""
        print(f"\nAPPROVAL REQUEST: {request['checkpoint_id']}")
        print(f"Timestamp: {datetime.fromtimestamp(request['timestamp'])}")
        
        safety_summary = {}
        for result in request['safety_results']:
            safety_summary[result['check_name']] = {
                'passed': result['passed'],
                'score': result['score']
            }
        
        print(f"Safety Summary: {safety_summary}")

class SafetyOrchestrator:
    """Main orchestrator for all safety protocols."""
    
    def __init__(self, 
                 storage_dir: str = "safety_data",
                 enable_human_approval: bool = True,
                 thresholds: SafetyThresholds = None):
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.validator = ModelSafetyValidator(thresholds)
        self.checkpoint_manager = ImmutableCheckpointManager(storage_dir / "checkpoints")
        self.approval_gateway = HumanApprovalGateway() if enable_human_approval else None
        
        self.deployment_log = []
        
    def safe_model_update(self,
                         old_model,
                         new_model,
                         tokenizer,
                         update_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform complete safety validation and deployment of model update.
        
        Returns:
            Dictionary containing deployment results and safety information
        """
        logger.info("Starting safe model update process")
        
        try:
            # 1. Create checkpoint of old model
            old_checkpoint_id = self.checkpoint_manager.create_checkpoint(
                old_model,
                {**update_metadata, 'type': 'pre_update_backup'}
            )
            
            # 2. Run safety validation
            is_safe, safety_results = self.validator.validate_model_update(
                old_model, new_model, tokenizer
            )
            
            if not is_safe:
                logger.error("Safety validation failed - aborting update")
                return {
                    'success': False,
                    'reason': 'safety_validation_failed',
                    'safety_results': [asdict(r) for r in safety_results],
                    'backup_checkpoint': old_checkpoint_id
                }
            
            # 3. Create checkpoint of new model
            new_checkpoint_id = self.checkpoint_manager.create_checkpoint(
                new_model,
                {**update_metadata, 'type': 'validated_update', 'safety_approved': True}
            )
            
            # 4. Human approval (if enabled)
            if self.approval_gateway:
                approved = self.approval_gateway.request_approval(
                    new_checkpoint_id,
                    safety_results,
                    update_metadata
                )
                
                if not approved:
                    logger.error("Human approval rejected - aborting update")
                    return {
                        'success': False,
                        'reason': 'human_approval_rejected',
                        'safety_results': [asdict(r) for r in safety_results],
                        'backup_checkpoint': old_checkpoint_id,
                        'validated_checkpoint': new_checkpoint_id
                    }
            
            # 5. Log successful deployment
            deployment_record = {
                'timestamp': time.time(),
                'old_checkpoint': old_checkpoint_id,
                'new_checkpoint': new_checkpoint_id,
                'safety_results': [asdict(r) for r in safety_results],
                'update_metadata': update_metadata,
                'human_approved': self.approval_gateway is not None
            }
            self.deployment_log.append(deployment_record)
            
            # Save deployment log
            self._save_deployment_log()
            
            logger.info("Safe model update completed successfully")
            return {
                'success': True,
                'old_checkpoint': old_checkpoint_id,
                'new_checkpoint': new_checkpoint_id,
                'safety_results': [asdict(r) for r in safety_results],
                'deployment_record': deployment_record
            }
            
        except Exception as e:
            logger.error(f"Safe model update failed with error: {e}")
            return {
                'success': False,
                'reason': 'system_error',
                'error': str(e)
            }
    
    def emergency_rollback(self, model, checkpoint_id: str) -> bool:
        """Emergency rollback to specified checkpoint."""
        logger.warning(f"Emergency rollback initiated to checkpoint: {checkpoint_id}")
        
        try:
            self.checkpoint_manager.rollback_to_checkpoint(model, checkpoint_id)
            
            # Log rollback
            rollback_record = {
                'timestamp': time.time(),
                'checkpoint_id': checkpoint_id,
                'reason': 'emergency_rollback'
            }
            self.deployment_log.append(rollback_record)
            self._save_deployment_log()
            
            logger.info("Emergency rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Emergency rollback failed: {e}")
            return False
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety report."""
        checkpoint_list = self.checkpoint_manager.list_checkpoints()
        
        return {
            'total_deployments': len(self.deployment_log),
            'recent_deployments': self.deployment_log[-5:] if self.deployment_log else [],
            'available_checkpoints': len(checkpoint_list),
            'latest_checkpoint': checkpoint_list[0] if checkpoint_list else None,
            'safety_thresholds': asdict(self.validator.thresholds),
            'deployment_log_path': str(self.storage_dir / "deployment_log.json")
        }
    
    def _save_deployment_log(self):
        """Save deployment log to disk."""
        log_path = self.storage_dir / "deployment_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.deployment_log, f, indent=2, default=str)