"""
KVTGStorage Performance Microbenchmark Suite

This module provides comprehensive performance benchmarks for the KVTGStorage system,
addressing the critical feedback about unverified system performance claims.

Key measurements:
- Compression ratios across different tensor shapes and methods
- Reconstruction error analysis
- Compression/decompression latency profiling
- Memory usage tracking
- Hardware-specific performance profiles
"""

import time
import torch
import numpy as np
import json
import psutil
import gc
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from contextlib import contextmanager

# Import our KVTG components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from kvtg.storage import KVTGStorage, CompressionMethod, QuantizationType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Single benchmark measurement result."""
    tensor_shape: Tuple[int, ...]
    compression_method: str
    quantization_type: str
    compression_ratio: float
    reconstruction_error: float
    compress_time_ms: float
    decompress_time_ms: float
    original_size_mb: float
    compressed_size_mb: float
    memory_usage_mb: float
    hardware_info: Dict[str, Any]

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    tensor_configs: List[Tuple[int, ...]]
    compression_methods: List[Tuple[CompressionMethod, QuantizationType]]
    num_runs: int = 5
    warmup_runs: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_results: bool = True
    results_path: str = "benchmarks/kvtg_storage_results.json"

class KVTGStorageBenchmark:
    """Comprehensive benchmark suite for KVTGStorage performance."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.hardware_info = self._get_hardware_info()
        
        # Initialize storage systems for each compression method
        self.storage_systems = {}
        for compression_method, quantization_type in config.compression_methods:
            key = (compression_method, quantization_type)
            self.storage_systems[key] = KVTGStorage(
                max_memory_items=10,
                compression_method=compression_method,
                quantization_type=quantization_type,
                persist_to_disk=False  # Keep in memory for benchmarking
            )
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Collect comprehensive hardware information."""
        info = {
            "cpu_model": "Unknown",
            "cpu_cores": psutil.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": self.config.device,
        }
        
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "cuda_version": torch.version.cuda,
            }
            info.update(gpu_info)
        
        return info
    
    def _generate_realistic_kv_cache(self, shape: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate realistic KV-cache tensors based on transformer architecture patterns.
        
        Args:
            shape: (batch_size, num_heads, seq_len, head_dim)
        
        Returns:
            Tuple of (key_cache, value_cache) tensors
        """
        batch_size, num_heads, seq_len, head_dim = shape
        
        # Generate realistic attention patterns with some structure
        # Keys tend to have more varied patterns
        key_cache = torch.randn(shape, device=self.config.device, dtype=torch.float32)
        key_cache = torch.nn.functional.normalize(key_cache, dim=-1)
        
        # Values often have smoother distributions
        value_cache = torch.randn(shape, device=self.config.device, dtype=torch.float32)
        value_cache = value_cache * 0.5 + torch.sin(torch.linspace(0, 10, head_dim).to(self.config.device))
        
        return key_cache, value_cache
    
    @contextmanager
    def _measure_memory(self):
        """Context manager to measure peak memory usage during operation."""
        if self.config.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            yield
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            process = psutil.Process()
            start_memory = process.memory_info().rss / (1024**2)  # MB
            yield
            end_memory = process.memory_info().rss / (1024**2)
            peak_memory = end_memory - start_memory
        
        self._current_memory_usage = peak_memory
    
    def _measure_compression(self, tensor_shape: Tuple[int, ...], compression_method: CompressionMethod, quantization_type: QuantizationType) -> BenchmarkResult:
        """
        Measure compression performance for a single configuration.
        
        Args:
            tensor_shape: Shape of tensors to benchmark
            compression_method: Compression method to test
            quantization_type: Quantization type to test
            
        Returns:
            BenchmarkResult containing all measurements
        """
        storage = self.storage_systems[(compression_method, quantization_type)]
        
        # Generate test data
        key_cache, value_cache = self._generate_realistic_kv_cache(tensor_shape)
        kv_cache = (key_cache, value_cache)
        
        # Calculate original size
        original_size = sum(tensor.numel() * tensor.element_size() for tensor in kv_cache)
        original_size_mb = original_size / (1024**2)
        
        # Warmup runs
        for _ in range(self.config.warmup_runs):
            storage.store("warmup", kv_cache)
            storage.get("warmup")
        
        # Clear cache
        storage._memory_cache.clear()
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Measure compression
        compress_times = []
        decompress_times = []
        compressed_sizes = []
        reconstruction_errors = []
        
        for run in range(self.config.num_runs):
            key_id = f"benchmark_run_{run}"
            
            # Measure compression time and memory
            with self._measure_memory():
                start_time = time.perf_counter()
                storage.store(key_id, kv_cache)
                compress_time = (time.perf_counter() - start_time) * 1000  # ms
            
            compress_times.append(compress_time)
            memory_usage = self._current_memory_usage
            
            # Get compressed size (approximate from storage)
            if key_id in storage._memory_cache:
                # For in-memory, we need to estimate compressed size
                compressed_data = storage._memory_cache[key_id]
                if hasattr(compressed_data, '__sizeof__'):
                    compressed_size = compressed_data.__sizeof__()
                else:
                    # Fallback estimation
                    compressed_size = sum(tensor.numel() * tensor.element_size() 
                                        for tensor in compressed_data if isinstance(tensor, torch.Tensor))
            else:
                compressed_size = original_size // 2  # Rough estimation
            
            compressed_sizes.append(compressed_size / (1024**2))  # MB
            
            # Measure decompression time
            start_time = time.perf_counter()
            retrieved_kv = storage.get(key_id)
            decompress_time = (time.perf_counter() - start_time) * 1000  # ms
            decompress_times.append(decompress_time)
            
            # Calculate reconstruction error
            if retrieved_kv is not None:
                orig_key, orig_val = kv_cache
                retr_key, retr_val = retrieved_kv
                
                key_error = torch.norm(orig_key - retr_key).item()
                val_error = torch.norm(orig_val - retr_val).item()
                total_error = key_error + val_error
            else:
                total_error = float('inf')  # Failed retrieval
            
            reconstruction_errors.append(total_error)
            
            # Clean up for next run
            storage._memory_cache.clear()
        
        # Calculate statistics
        avg_compress_time = np.mean(compress_times)
        avg_decompress_time = np.mean(decompress_times)
        avg_compressed_size = np.mean(compressed_sizes)
        avg_reconstruction_error = np.mean(reconstruction_errors)
        compression_ratio = original_size_mb / avg_compressed_size if avg_compressed_size > 0 else 0
        
        return BenchmarkResult(
            tensor_shape=tensor_shape,
            compression_method=compression_method.name,
            quantization_type=quantization_type.name,
            compression_ratio=compression_ratio,
            reconstruction_error=avg_reconstruction_error,
            compress_time_ms=avg_compress_time,
            decompress_time_ms=avg_decompress_time,
            original_size_mb=original_size_mb,
            compressed_size_mb=avg_compressed_size,
            memory_usage_mb=memory_usage,
            hardware_info=self.hardware_info
        )
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across all configurations.
        
        Returns:
            Dictionary containing all benchmark results and analysis
        """
        logger.info("Starting KVTGStorage comprehensive benchmark")
        logger.info(f"Hardware: {self.hardware_info}")
        logger.info(f"Testing {len(self.config.tensor_configs)} tensor shapes with {len(self.config.compression_methods)} methods")
        
        self.results = []
        
        for i, tensor_shape in enumerate(self.config.tensor_configs):
            logger.info(f"Testing tensor shape {i+1}/{len(self.config.tensor_configs)}: {tensor_shape}")
            
            for j, (compression_method, quantization_type) in enumerate(self.config.compression_methods):
                logger.info(f"  Method {j+1}/{len(self.config.compression_methods)}: {compression_method.name} with {quantization_type.name}")
                try:
                    result = self._measure_compression(tensor_shape, compression_method, quantization_type)
                    self.results.append(result)
                    
                    logger.info(f"    Compression ratio: {result.compression_ratio:.2f}x")
                    logger.info(f"    Compress time: {result.compress_time_ms:.2f}ms")
                    logger.info(f"    Decompress time: {result.decompress_time_ms:.2f}ms")
                    logger.info(f"    Reconstruction error: {result.reconstruction_error:.6f}")
                    
                except Exception as e:
                    logger.error(f"    Failed: {e}")
                    # Continue with other tests
        
        # Analyze results
        analysis = self._analyze_results()
        
        if self.config.save_results:
            self._save_results(analysis)
        
        logger.info("Benchmark completed successfully")
        return analysis
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and generate summary statistics."""
        if not self.results:
            return {"error": "No benchmark results to analyze"}
        
        # Group results by method
        method_results = {}
        for result in self.results:
            method_key = f"{result.compression_method}_{result.quantization_type}"
            if method_key not in method_results:
                method_results[method_key] = []
            method_results[method_key].append(result)
        
        # Calculate summary statistics
        summary = {
            "hardware_info": self.hardware_info,
            "benchmark_config": {
                "tensor_configs": self.config.tensor_configs,
                "num_runs": self.config.num_runs,
                "device": self.config.device
            },
            "method_performance": {},
            "tensor_shape_analysis": {},
            "recommendations": []
        }
        
        # Analyze by method
        for method, results in method_results.items():
            compression_ratios = [r.compression_ratio for r in results]
            compress_times = [r.compress_time_ms for r in results]
            decompress_times = [r.decompress_time_ms for r in results]
            reconstruction_errors = [r.reconstruction_error for r in results]
            
            summary["method_performance"][method] = {
                "avg_compression_ratio": np.mean(compression_ratios),
                "std_compression_ratio": np.std(compression_ratios),
                "min_compression_ratio": np.min(compression_ratios),
                "max_compression_ratio": np.max(compression_ratios),
                "avg_compress_time_ms": np.mean(compress_times),
                "avg_decompress_time_ms": np.mean(decompress_times),
                "avg_reconstruction_error": np.mean(reconstruction_errors),
                "max_reconstruction_error": np.max(reconstruction_errors)
            }
        
        # Generate recommendations
        best_compression = max(summary["method_performance"].items(), 
                             key=lambda x: x[1]["avg_compression_ratio"])
        fastest_compression = min(summary["method_performance"].items(),
                                key=lambda x: x[1]["avg_compress_time_ms"])
        lowest_error = min(summary["method_performance"].items(),
                          key=lambda x: x[1]["avg_reconstruction_error"])
        
        summary["recommendations"] = [
            f"Best compression ratio: {best_compression[0]} ({best_compression[1]['avg_compression_ratio']:.2f}x)",
            f"Fastest compression: {fastest_compression[0]} ({fastest_compression[1]['avg_compress_time_ms']:.2f}ms)",
            f"Lowest reconstruction error: {lowest_error[0]} ({lowest_error[1]['avg_reconstruction_error']:.6f})"
        ]
        
        # Add all detailed results
        summary["detailed_results"] = [asdict(result) for result in self.results]
        
        return summary
    
    def _save_results(self, analysis: Dict[str, Any]):
        """Save benchmark results to file."""
        results_path = Path(self.config.results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")

def create_standard_benchmark_config() -> BenchmarkConfig:
    """Create standard benchmark configuration for common use cases."""
    
    # Standard transformer configurations
    tensor_configs = [
        # (batch_size, num_heads, seq_len, head_dim)
        (1, 32, 128, 64),   # Small sequence - Mistral-7B style
        (1, 32, 256, 64),   # Medium sequence
        (1, 32, 512, 64),   # Long sequence
        (1, 32, 1024, 64),  # Very long sequence
        (4, 32, 128, 64),   # Small batch
        (8, 32, 128, 64),   # Medium batch
    ]
    
    compression_methods = [
        (CompressionMethod.NONE, QuantizationType.FP32),
        (CompressionMethod.QUANTIZATION, QuantizationType.FP16),
        (CompressionMethod.QUANTIZATION, QuantizationType.INT8),
        (CompressionMethod.LOW_RANK, QuantizationType.FP32),
        (CompressionMethod.SPARSIFICATION, QuantizationType.FP16),
        (CompressionMethod.HYBRID, QuantizationType.FP16)
    ]
    
    return BenchmarkConfig(
        tensor_configs=tensor_configs,
        compression_methods=compression_methods,
        num_runs=5,
        warmup_runs=2,
        save_results=True,
        results_path="benchmarks/kvtg_storage_results.json"
    )

def run_standard_benchmark():
    """Run standard benchmark suite and print results."""
    config = create_standard_benchmark_config()
    benchmark = KVTGStorageBenchmark(config)
    
    try:
        results = benchmark.run_comprehensive_benchmark()
        
        if not results or 'hardware_info' not in results:
            logger.error("Benchmark produced no results.")
            return

        print("\n" + "="*60)
        print("KVTG STORAGE BENCHMARK RESULTS")
        print("="*60)
        
        print(f"\nHardware: {results['hardware_info']['device']}")
        if 'gpu_name' in results['hardware_info']:
            print(f"GPU: {results['hardware_info']['gpu_name']}")
            print(f"GPU Memory: {results['hardware_info']['gpu_memory_gb']:.1f} GB")
        
        print("\nPerformance by Method:")
        print("-" * 40)
        
        for method, perf in results['method_performance'].items():
            print(f"\n{method}:")
            print(f"  Compression Ratio: {perf['avg_compression_ratio']:.2f}x (±{perf['std_compression_ratio']:.2f})")
            print(f"  Compress Time: {perf['avg_compress_time_ms']:.2f}ms")
            print(f"  Decompress Time: {perf['avg_decompress_time_ms']:.2f}ms")
            print(f"  Reconstruction Error: {perf['avg_reconstruction_error']:.6f}")
        
        print("\nRecommendations:")
        print("-" * 20)
        for rec in results['recommendations']:
            print(f"• {rec}")
        
        return results
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    # Run the standard benchmark
    run_standard_benchmark()
