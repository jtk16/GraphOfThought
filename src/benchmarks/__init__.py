"""
Benchmark suite for KVTG+SEAL implementation.

This package provides comprehensive benchmarking tools to validate
system performance claims and provide empirical evidence for:
- KVTGStorage compression performance
- SEAL training stability and efficiency
- End-to-end system latency
- Memory usage patterns
"""

from .benchmark_kvtg_storage import KVTGStorageBenchmark, create_standard_benchmark_config, run_standard_benchmark

__all__ = [
    'KVTGStorageBenchmark',
    'create_standard_benchmark_config', 
    'run_standard_benchmark'
]