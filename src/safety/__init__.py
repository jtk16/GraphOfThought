"""
Safety protocols and frameworks for KVTG+SEAL system.

This package provides comprehensive safety measures for self-modifying AI systems:
- Automated regression testing
- Model safety validation 
- Immutable checkpoint management
- Behavioral drift detection
- Human-in-the-loop approval
- Emergency rollback capabilities
"""

from .safety_protocols import (
    SafetyThresholds,
    SafetyCheckResult, 
    ModelSafetyValidator,
    ImmutableCheckpointManager,
    HumanApprovalGateway,
    SafetyOrchestrator
)

__all__ = [
    'SafetyThresholds',
    'SafetyCheckResult',
    'ModelSafetyValidator', 
    'ImmutableCheckpointManager',
    'HumanApprovalGateway',
    'SafetyOrchestrator'
]