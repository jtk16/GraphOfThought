import logging
import pickle
import os
import struct
import time
import warnings
from typing import Dict, Optional, Tuple, Any, Union, NamedTuple
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn.functional as F
import numpy as np

# Type alias for KV-cache - tuple of key and value tensors for each layer
KVCacheType = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]

class CompressionMethod(Enum):
    """Available compression methods for KV-caches."""
    NONE = "none"
    PICKLE_ONLY = "pickle_only"
    QUANTIZATION = "quantization"
    LOW_RANK = "low_rank"
    SPARSIFICATION = "sparsification"
    HYBRID = "hybrid"  # Combines multiple techniques

class QuantizationType(Enum):
    """Quantization precision levels."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"

@dataclass
class CompressionStats:
    """Statistics for compression performance analysis."""
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    compression_time_ms: float
    decompression_time_ms: float
    reconstruction_error: float  # L2 norm of difference
    method_used: CompressionMethod
    quantization_type: Optional[QuantizationType] = None
    
    def __post_init__(self):
        if self.original_size_bytes > 0:
            self.compression_ratio = self.original_size_bytes / self.compressed_size_bytes
        else:
            self.compression_ratio = 1.0

class CompressedKVCache(NamedTuple):
    """Container for compressed KV-cache data."""
    data: bytes
    metadata: Dict[str, Any]
    compression_stats: CompressionStats

class AdvancedKVCompressor:
    """Advanced compression algorithms for KV-caches with mathematical optimizations."""
    
    def __init__(self, 
                 compression_method: CompressionMethod = CompressionMethod.HYBRID,
                 quantization_type: QuantizationType = QuantizationType.FP16,
                 svd_rank_ratio: float = 0.8,
                 sparsity_threshold: float = 1e-4,
                 adaptive_compression: bool = True):
        """
        Initialize advanced KV-cache compressor.
        
        Args:
            compression_method: Primary compression technique to use
            quantization_type: Quantization precision for numeric values
            svd_rank_ratio: Ratio of singular values to keep in SVD compression
            sparsity_threshold: Magnitude threshold for sparsification
            adaptive_compression: Whether to adapt compression based on cache characteristics
        """
        self.compression_method = compression_method
        self.quantization_type = quantization_type
        self.svd_rank_ratio = svd_rank_ratio
        self.sparsity_threshold = sparsity_threshold
        self.adaptive_compression = adaptive_compression
        
        # Cache tensor statistics for adaptive compression
        self._tensor_stats = {}
        
    def _compute_tensor_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Compute statistical properties of tensor for compression decisions."""
        with torch.no_grad():
            flattened = tensor.flatten()
            stats = {
                'mean': float(flattened.mean()),
                'std': float(flattened.std()),
                'min': float(flattened.min()),
                'max': float(flattened.max()),
                'sparsity': float((flattened.abs() < self.sparsity_threshold).float().mean()),
                'kurtosis': self._compute_kurtosis(flattened),
                'effective_rank': self._estimate_rank(tensor)
            }
        return stats
    
    def _compute_kurtosis(self, tensor: torch.Tensor) -> float:
        """Compute kurtosis to understand tail distribution."""
        mean = tensor.mean()
        std = tensor.std()
        if std == 0:
            return 0.0
        normalized = (tensor - mean) / std
        return float(normalized.pow(4).mean() - 3.0)
    
    def _estimate_rank(self, tensor: torch.Tensor) -> float:
        """Estimate effective rank using nuclear norm approximation."""
        if tensor.dim() < 2:
            return float(tensor.numel())
        
        # For high-dimensional tensors, estimate rank on reshaped 2D version
        original_shape = tensor.shape
        if tensor.dim() > 2:
            # Reshape to 2D: (batch*heads*seq, head_dim) or similar
            tensor_2d = tensor.view(-1, original_shape[-1])
        else:
            tensor_2d = tensor
            
        # Use SVD on a sample if tensor is very large
        if tensor_2d.numel() > 1e6:
            # Sample a subset for rank estimation
            sample_size = min(1000, tensor_2d.size(0))
            indices = torch.randperm(tensor_2d.size(0))[:sample_size]
            tensor_2d = tensor_2d[indices]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, s, _ = torch.svd(tensor_2d.float())
                # Estimate effective rank using 99% energy criterion
                cumsum_ratio = s.cumsum(0) / s.sum()
                effective_rank = float((cumsum_ratio < 0.99).sum() + 1)
            return min(effective_rank, float(min(tensor_2d.shape)))
        except:
            return float(min(tensor_2d.shape))
    
    def _quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply quantization with dynamic range optimization."""
        metadata = {}
        
        if self.quantization_type == QuantizationType.FP32:
            return tensor, metadata
        
        # Compute dynamic range for optimal quantization
        tensor_abs = tensor.abs()
        percentile_99 = torch.quantile(tensor_abs.flatten(), 0.99)
        max_val = tensor_abs.max()
        
        # Use 99th percentile for clipping to reduce outlier impact
        clip_val = min(percentile_99 * 1.2, max_val)
        
        if self.quantization_type == QuantizationType.FP16:
            # Clip to prevent overflow in FP16
            tensor_clipped = torch.clamp(tensor, -clip_val, clip_val)
            quantized = tensor_clipped.half()
            metadata = {'clip_val': float(clip_val), 'dtype': 'fp16'}
            
        elif self.quantization_type == QuantizationType.INT8:
            # Symmetric quantization with zero-point at 0
            scale = clip_val / 127.0
            tensor_clipped = torch.clamp(tensor, -clip_val, clip_val)
            quantized = torch.round(tensor_clipped / scale).clamp(-128, 127).to(torch.int8)
            metadata = {'scale': float(scale), 'clip_val': float(clip_val), 'dtype': 'int8'}
        
        return quantized, metadata
    
    def _dequantize_tensor(self, tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """Reconstruct tensor from quantized representation."""
        if metadata.get('dtype') == 'fp16':
            return tensor.float()
        elif metadata.get('dtype') == 'int8':
            scale = metadata['scale']
            return tensor.float() * scale
        return tensor
    
    def _apply_svd_compression(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Apply SVD-based low-rank approximation."""
        original_shape = tensor.shape
        
        # Reshape to 2D for SVD: (batch*heads*seq, head_dim)
        if tensor.dim() > 2:
            tensor_2d = tensor.view(-1, original_shape[-1])
        else:
            tensor_2d = tensor
        
        try:
            U, s, V = torch.svd(tensor_2d.float())
            
            # Determine rank to keep based on energy or fixed ratio
            if self.adaptive_compression:
                # Keep components that contain 99% of the energy
                energy_cumsum = s.pow(2).cumsum(0)
                total_energy = energy_cumsum[-1]
                rank = int((energy_cumsum / total_energy < 0.99).sum()) + 1
                rank = min(rank, int(self.svd_rank_ratio * s.size(0)))
            else:
                rank = int(self.svd_rank_ratio * s.size(0))
            
            rank = max(1, min(rank, s.size(0)))
            
            # Truncate to rank-r approximation
            U_r = U[:, :rank]
            s_r = s[:rank]
            V_r = V[:, :rank]
            
            return U_r, s_r, V_r, rank
            
        except Exception as e:
            logging.warning(f"SVD compression failed: {e}. Using original tensor.")
            # Return identity decomposition
            return tensor_2d, torch.ones(1), torch.eye(tensor_2d.size(1)), tensor_2d.size(1)
    
    def _reconstruct_from_svd(self, U: torch.Tensor, s: torch.Tensor, V: torch.Tensor, 
                             original_shape: Tuple[int, ...]) -> torch.Tensor:
        """Reconstruct tensor from SVD components."""
        # Reconstruct 2D tensor: U @ diag(s) @ V^T
        tensor_2d = U @ torch.diag(s) @ V.T
        
        # Reshape back to original dimensions
        return tensor_2d.view(original_shape)
    
    def _apply_sparsification(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply magnitude-based sparsification."""
        mask = tensor.abs() >= self.sparsity_threshold
        sparse_tensor = tensor * mask.float()
        return sparse_tensor, mask
    
    def _serialize_compressed_data(self, compressed_layers: list) -> bytes:
        """Custom serialization optimized for compressed KV-cache structure."""
        # Use efficient binary format with minimal overhead
        buffer = bytearray()
        
        # Header: number of layers
        buffer.extend(struct.pack('I', len(compressed_layers)))
        
        for layer_data in compressed_layers:
            # Serialize each layer's compressed data
            layer_bytes = pickle.dumps(layer_data, protocol=pickle.HIGHEST_PROTOCOL)
            # Length prefix for each layer
            buffer.extend(struct.pack('I', len(layer_bytes)))
            buffer.extend(layer_bytes)
        
        return bytes(buffer)
    
    def _deserialize_compressed_data(self, data: bytes) -> list:
        """Deserialize compressed KV-cache data."""
        buffer = data
        offset = 0
        
        # Read number of layers
        num_layers = struct.unpack('I', buffer[offset:offset+4])[0]
        offset += 4
        
        layers = []
        for _ in range(num_layers):
            # Read layer data length
            layer_len = struct.unpack('I', buffer[offset:offset+4])[0]
            offset += 4
            
            # Read and deserialize layer data
            layer_bytes = buffer[offset:offset+layer_len]
            layer_data = pickle.loads(layer_bytes)
            layers.append(layer_data)
            offset += layer_len
        
        return layers
    
    def compress(self, kv_cache: KVCacheType) -> CompressedKVCache:
        """Main compression method combining multiple techniques."""
        start_time = time.time()
        
        # Calculate original size
        original_size = sum(
            key.numel() * key.element_size() + value.numel() * value.element_size()
            for key, value in kv_cache
        )
        
        compressed_layers = []
        total_reconstruction_error = 0.0
        
        for layer_idx, (key_tensor, value_tensor) in enumerate(kv_cache):
            layer_data = {'layer_idx': layer_idx}
            
            # Process key and value tensors separately
            for tensor_name, tensor in [('key', key_tensor), ('value', value_tensor)]:
                tensor_data = {'original_shape': tensor.shape}
                
                # Apply compression pipeline based on method
                if self.compression_method == CompressionMethod.HYBRID:
                    # Adaptive hybrid approach
                    stats = self._compute_tensor_stats(tensor)
                    
                    # Decide compression strategy based on tensor characteristics
                    if stats['effective_rank'] / min(tensor.shape[-2:]) < 0.7:
                        # Low rank - use SVD
                        U, s, V, rank = self._apply_svd_compression(tensor)
                        U_q, U_meta = self._quantize_tensor(U)
                        s_q, s_meta = self._quantize_tensor(s)
                        V_q, V_meta = self._quantize_tensor(V)
                        
                        tensor_data.update({
                            'method': 'svd_quantized',
                            'U': U_q, 'U_meta': U_meta,
                            's': s_q, 's_meta': s_meta,
                            'V': V_q, 'V_meta': V_meta,
                            'rank': rank
                        })
                        
                        # Compute reconstruction error
                        reconstructed = self._reconstruct_from_svd(
                            self._dequantize_tensor(U_q, U_meta),
                            self._dequantize_tensor(s_q, s_meta),
                            self._dequantize_tensor(V_q, V_meta),
                            tensor.shape
                        )
                        
                    elif stats['sparsity'] > 0.3:
                        # Sparse - use sparsification + quantization
                        sparse_tensor, mask = self._apply_sparsification(tensor)
                        sparse_q, sparse_meta = self._quantize_tensor(sparse_tensor)
                        
                        tensor_data.update({
                            'method': 'sparse_quantized',
                            'tensor': sparse_q,
                            'meta': sparse_meta,
                            'mask': mask
                        })
                        
                        reconstructed = self._dequantize_tensor(sparse_q, sparse_meta)
                        
                    else:
                        # Dense - use quantization only
                        quantized, meta = self._quantize_tensor(tensor)
                        tensor_data.update({
                            'method': 'quantized',
                            'tensor': quantized,
                            'meta': meta
                        })
                        
                        reconstructed = self._dequantize_tensor(quantized, meta)
                    
                    # Accumulate reconstruction error
                    error = torch.norm(tensor.float() - reconstructed.float()).item()
                    total_reconstruction_error += error
                    
                elif self.compression_method == CompressionMethod.QUANTIZATION:
                    quantized, meta = self._quantize_tensor(tensor)
                    tensor_data.update({
                        'method': 'quantized',
                        'tensor': quantized,
                        'meta': meta
                    })
                    
                elif self.compression_method == CompressionMethod.LOW_RANK:
                    U, s, V, rank = self._apply_svd_compression(tensor)
                    tensor_data.update({
                        'method': 'svd',
                        'U': U, 's': s, 'V': V, 'rank': rank
                    })
                    
                elif self.compression_method == CompressionMethod.SPARSIFICATION:
                    sparse_tensor, mask = self._apply_sparsification(tensor)
                    tensor_data.update({
                        'method': 'sparse',
                        'tensor': sparse_tensor,
                        'mask': mask
                    })
                
                layer_data[tensor_name] = tensor_data
            
            compressed_layers.append(layer_data)
        
        # Serialize compressed data
        compressed_data = self._serialize_compressed_data(compressed_layers)
        compressed_size = len(compressed_data)
        
        compression_time = (time.time() - start_time) * 1000  # ms
        
        # Create compression statistics
        stats = CompressionStats(
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size,
            compression_ratio=original_size / compressed_size if compressed_size > 0 else 1.0,
            compression_time_ms=compression_time,
            decompression_time_ms=0.0,  # Will be filled during decompression
            reconstruction_error=total_reconstruction_error,
            method_used=self.compression_method,
            quantization_type=self.quantization_type
        )
        
        metadata = {
            'compression_method': self.compression_method.value,
            'quantization_type': self.quantization_type.value,
            'svd_rank_ratio': self.svd_rank_ratio,
            'sparsity_threshold': self.sparsity_threshold,
            'num_layers': len(kv_cache)
        }
        
        return CompressedKVCache(compressed_data, metadata, stats)
    
    def decompress(self, compressed_cache: CompressedKVCache) -> KVCacheType:
        """Decompress KV-cache back to original format."""
        start_time = time.time()
        
        # Deserialize compressed data
        compressed_layers = self._deserialize_compressed_data(compressed_cache.data)
        
        reconstructed_layers = []
        
        for layer_data in compressed_layers:
            key_data = layer_data['key']
            value_data = layer_data['value']
            
            # Reconstruct key tensor
            key_tensor = self._reconstruct_tensor(key_data)
            
            # Reconstruct value tensor
            value_tensor = self._reconstruct_tensor(value_data)
            
            reconstructed_layers.append((key_tensor, value_tensor))
        
        decompression_time = (time.time() - start_time) * 1000  # ms
        compressed_cache.compression_stats.decompression_time_ms = decompression_time
        
        return tuple(reconstructed_layers)
    
    def _reconstruct_tensor(self, tensor_data: Dict[str, Any]) -> torch.Tensor:
        """Reconstruct individual tensor from compressed representation."""
        method = tensor_data['method']
        original_shape = tensor_data['original_shape']
        
        if method == 'quantized':
            tensor = self._dequantize_tensor(tensor_data['tensor'], tensor_data['meta'])
            
        elif method == 'svd':
            tensor = self._reconstruct_from_svd(
                tensor_data['U'], tensor_data['s'], tensor_data['V'], original_shape
            )
            
        elif method == 'svd_quantized':
            U = self._dequantize_tensor(tensor_data['U'], tensor_data['U_meta'])
            s = self._dequantize_tensor(tensor_data['s'], tensor_data['s_meta'])
            V = self._dequantize_tensor(tensor_data['V'], tensor_data['V_meta'])
            tensor = self._reconstruct_from_svd(U, s, V, original_shape)
            
        elif method == 'sparse':
            tensor = tensor_data['tensor']
            
        elif method == 'sparse_quantized':
            tensor = self._dequantize_tensor(tensor_data['tensor'], tensor_data['meta'])
        
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")
        
        return tensor.view(original_shape)

class KVTGStorage:
    """
    Manages storage and retrieval of KV-cache snapshots for KVTG nodes.
    
    Provides memory management with LRU eviction, optional disk persistence,
    and compression capabilities for efficient storage of transformer KV-caches.
    """
    
    def __init__(self, max_memory_items: int = 100, persist_to_disk: bool = True, 
                 storage_dir: str = "./kvtg_cache", compress: bool = True,
                 compression_method: CompressionMethod = CompressionMethod.HYBRID,
                 quantization_type: QuantizationType = QuantizationType.FP16,
                 svd_rank_ratio: float = 0.8,
                 sparsity_threshold: float = 1e-4,
                 enable_benchmarking: bool = True):
        """
        Initialize KV-cache storage system with advanced compression.
        
        Args:
            max_memory_items: Maximum number of KV-cache items to keep in memory
            persist_to_disk: Whether to save evicted items to disk
            storage_dir: Directory for disk persistence
            compress: Whether to compress KV-caches when storing (legacy parameter)
            compression_method: Advanced compression technique to use
            quantization_type: Quantization precision for numeric values
            svd_rank_ratio: Ratio of singular values to keep in SVD compression
            sparsity_threshold: Magnitude threshold for sparsification
            enable_benchmarking: Whether to collect detailed compression metrics
        """
        self.max_memory_items = max_memory_items
        self.persist_to_disk = persist_to_disk
        self.storage_dir = storage_dir
        self.compress = compress or compression_method != CompressionMethod.NONE
        self.enable_benchmarking = enable_benchmarking
        
        # Initialize advanced compressor
        self.compressor = AdvancedKVCompressor(
            compression_method=compression_method,
            quantization_type=quantization_type,
            svd_rank_ratio=svd_rank_ratio,
            sparsity_threshold=sparsity_threshold,
            adaptive_compression=True
        )
        
        # In-memory cache using OrderedDict for LRU behavior
        self._memory_cache: OrderedDict[str, KVCacheType] = OrderedDict()
        
        # Compression statistics tracking
        self._compression_stats: Dict[str, CompressionStats] = {}
        self._global_stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'total_compression_time_ms': 0.0,
            'total_decompression_time_ms': 0.0,
            'average_compression_ratio': 0.0,
            'total_original_bytes': 0,
            'total_compressed_bytes': 0
        }
        
        # Track which items are on disk
        self._disk_items: set[str] = set()
        
        if self.persist_to_disk:
            os.makedirs(self.storage_dir, exist_ok=True)
            # Load existing disk items
            self._load_disk_inventory()
        
        logging.info(f"KVTGStorage initialized: max_memory={max_memory_items}, "
                    f"disk_persist={persist_to_disk}, compress={self.compress}, "
                    f"compression_method={compression_method.value}, "
                    f"quantization={quantization_type.value}")
    
    def _load_disk_inventory(self):
        """Load list of items available on disk."""
        if os.path.exists(self.storage_dir):
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.pkl'):
                    cache_id = filename[:-4]  # Remove .pkl extension
                    self._disk_items.add(cache_id)
            logging.info(f"Found {len(self._disk_items)} cached items on disk")
    
    def _get_disk_path(self, cache_id: str) -> str:
        """Get filesystem path for a cache ID."""
        return os.path.join(self.storage_dir, f"{cache_id}.pkl")
    
    def _compress_kv_cache(self, kv_cache: KVCacheType) -> bytes:
        """Compress KV-cache using advanced compression techniques."""
        if not self.compress:
            return pickle.dumps(kv_cache, protocol=pickle.HIGHEST_PROTOCOL)
        
        try:
            # Use advanced compression
            compressed_cache = self.compressor.compress(kv_cache)
            
            # Store compression statistics if benchmarking enabled
            if self.enable_benchmarking:
                self._update_global_stats(compressed_cache.compression_stats)
            
            # Serialize the entire compressed cache object
            return pickle.dumps(compressed_cache, protocol=pickle.HIGHEST_PROTOCOL)
            
        except Exception as e:
            logging.warning(f"Advanced compression failed: {e}. Falling back to pickle.")
            return pickle.dumps(kv_cache, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _decompress_kv_cache(self, compressed_data: bytes) -> KVCacheType:
        """Decompress KV-cache from storage."""
        try:
            # Try to load as compressed cache first
            loaded_data = pickle.loads(compressed_data)
            
            if isinstance(loaded_data, CompressedKVCache):
                # Advanced decompression
                kv_cache = self.compressor.decompress(loaded_data)
                
                # Update statistics if benchmarking enabled
                if self.enable_benchmarking:
                    self._global_stats['total_decompressions'] += 1
                    self._global_stats['total_decompression_time_ms'] += loaded_data.compression_stats.decompression_time_ms
                
                return kv_cache
            else:
                # Legacy format - direct KV-cache tuple
                return loaded_data
                
        except Exception as e:
            logging.error(f"Decompression failed: {e}")
            # Attempt to load as legacy format
            return pickle.loads(compressed_data)
    
    def _evict_lru(self):
        """Evict least recently used item from memory to make space."""
        if not self._memory_cache:
            return
        
        # Remove oldest item (first in OrderedDict)
        lru_id, lru_cache = self._memory_cache.popitem(last=False)
        
        if self.persist_to_disk:
            # Save to disk before evicting
            try:
                compressed_data = self._compress_kv_cache(lru_cache)
                disk_path = self._get_disk_path(lru_id)
                with open(disk_path, 'wb') as f:
                    f.write(compressed_data)
                self._disk_items.add(lru_id)
                
                # Log compression statistics for debugging
                if self.enable_benchmarking and hasattr(self, '_compression_stats'):
                    original_size = sum(
                        key.numel() * key.element_size() + value.numel() * value.element_size()
                        for key, value in lru_cache
                    )
                    compressed_size = len(compressed_data)
                    ratio = original_size / compressed_size if compressed_size > 0 else 1.0
                    logging.debug(f"Compressed cache {lru_id}: {original_size} -> {compressed_size} bytes (ratio: {ratio:.2f}x)")
                logging.debug(f"Evicted KV-cache {lru_id} to disk")
            except Exception as e:
                logging.error(f"Failed to persist KV-cache {lru_id} to disk: {e}")
        else:
            logging.debug(f"Evicted KV-cache {lru_id} from memory (no disk persistence)")
    
    def store(self, cache_id: str, kv_cache: KVCacheType) -> bool:
        """
        Store a KV-cache snapshot.
        
        Args:
            cache_id: Unique identifier for this cache snapshot
            kv_cache: The KV-cache data to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        try:
            # Remove if already exists (update scenario)
            if cache_id in self._memory_cache:
                del self._memory_cache[cache_id]
            
            # Make space if needed
            while len(self._memory_cache) >= self.max_memory_items:
                self._evict_lru()
            
            # Store in memory cache
            self._memory_cache[cache_id] = kv_cache
            
            # Store compression statistics if available and benchmarking enabled
            if self.enable_benchmarking and hasattr(kv_cache, '__compressed_stats__'):
                self._compression_stats[cache_id] = kv_cache.__compressed_stats__
            
            # Remove from disk items if it was there
            if cache_id in self._disk_items:
                self._disk_items.remove(cache_id)
                disk_path = self._get_disk_path(cache_id)
                if os.path.exists(disk_path):
                    os.remove(disk_path)
            
            logging.debug(f"Stored KV-cache {cache_id} in memory")
            return True
            
        except Exception as e:
            logging.error(f"Failed to store KV-cache {cache_id}: {e}")
            return False
    
    def get(self, cache_id: str) -> Optional[KVCacheType]:
        """
        Retrieve a KV-cache snapshot.
        
        Args:
            cache_id: Unique identifier for the cache snapshot
            
        Returns:
            The KV-cache data if found, None otherwise
        """
        # Check memory cache first
        if cache_id in self._memory_cache:
            # Move to end (mark as recently used)
            kv_cache = self._memory_cache.pop(cache_id)
            self._memory_cache[cache_id] = kv_cache
            logging.debug(f"Retrieved KV-cache {cache_id} from memory")
            return kv_cache
        
        # Check disk cache
        if cache_id in self._disk_items:
            try:
                disk_path = self._get_disk_path(cache_id)
                if os.path.exists(disk_path):
                    with open(disk_path, 'rb') as f:
                        compressed_data = f.read()
                    kv_cache = self._decompress_kv_cache(compressed_data)
                    
                    # Move back to memory cache
                    if len(self._memory_cache) >= self.max_memory_items:
                        self._evict_lru()
                    
                    self._memory_cache[cache_id] = kv_cache
                    self._disk_items.remove(cache_id)
                    os.remove(disk_path)
                    
                    logging.debug(f"Retrieved KV-cache {cache_id} from disk and moved to memory")
                    return kv_cache
                else:
                    # File missing, remove from disk items
                    self._disk_items.remove(cache_id)
                    
            except Exception as e:
                logging.error(f"Failed to load KV-cache {cache_id} from disk: {e}")
                if cache_id in self._disk_items:
                    self._disk_items.remove(cache_id)
        
        logging.debug(f"KV-cache {cache_id} not found")
        return None
    
    def exists(self, cache_id: str) -> bool:
        """Check if a KV-cache exists (in memory or on disk)."""
        return cache_id in self._memory_cache or cache_id in self._disk_items
    
    def remove(self, cache_id: str) -> bool:
        """
        Remove a KV-cache from storage.
        
        Args:
            cache_id: Unique identifier for the cache to remove
            
        Returns:
            True if removal was successful, False if not found
        """
        found = False
        
        # Remove from memory
        if cache_id in self._memory_cache:
            del self._memory_cache[cache_id]
            found = True
        
        # Remove from disk
        if cache_id in self._disk_items:
            try:
                disk_path = self._get_disk_path(cache_id)
                if os.path.exists(disk_path):
                    os.remove(disk_path)
                self._disk_items.remove(cache_id)
                found = True
            except Exception as e:
                logging.error(f"Failed to remove KV-cache {cache_id} from disk: {e}")
        
        return found
    
    def clear(self):
        """Clear all cached items from memory and optionally disk."""
        self._memory_cache.clear()
        
        if self.persist_to_disk:
            for cache_id in list(self._disk_items):
                try:
                    disk_path = self._get_disk_path(cache_id)
                    if os.path.exists(disk_path):
                        os.remove(disk_path)
                except Exception as e:
                    logging.error(f"Failed to remove disk cache {cache_id}: {e}")
            
            self._disk_items.clear()
        
        logging.info("Cleared all KV-cache storage")
    
    def _update_global_stats(self, stats: CompressionStats):
        """Update global compression statistics."""
        self._global_stats['total_compressions'] += 1
        self._global_stats['total_compression_time_ms'] += stats.compression_time_ms
        self._global_stats['total_original_bytes'] += stats.original_size_bytes
        self._global_stats['total_compressed_bytes'] += stats.compressed_size_bytes
        
        # Update average compression ratio
        if self._global_stats['total_original_bytes'] > 0:
            self._global_stats['average_compression_ratio'] = (
                self._global_stats['total_original_bytes'] / 
                self._global_stats['total_compressed_bytes']
            )
    
    def get_compression_stats(self, cache_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed compression statistics."""
        if cache_id and cache_id in self._compression_stats:
            stats = self._compression_stats[cache_id]
            return {
                'cache_id': cache_id,
                'original_size_bytes': stats.original_size_bytes,
                'compressed_size_bytes': stats.compressed_size_bytes,
                'compression_ratio': stats.compression_ratio,
                'compression_time_ms': stats.compression_time_ms,
                'decompression_time_ms': stats.decompression_time_ms,
                'reconstruction_error': stats.reconstruction_error,
                'method_used': stats.method_used.value,
                'quantization_type': stats.quantization_type.value if stats.quantization_type else None
            }
        else:
            return dict(self._global_stats)
    
    def benchmark_compression_methods(self, kv_cache: KVCacheType) -> Dict[CompressionMethod, Dict[str, Any]]:
        """Benchmark different compression methods on a given KV-cache."""
        results = {}
        
        methods_to_test = [
            (CompressionMethod.NONE, QuantizationType.FP32),
            (CompressionMethod.QUANTIZATION, QuantizationType.FP16),
            (CompressionMethod.QUANTIZATION, QuantizationType.INT8),
            (CompressionMethod.LOW_RANK, QuantizationType.FP32),
            (CompressionMethod.SPARSIFICATION, QuantizationType.FP16),
            (CompressionMethod.HYBRID, QuantizationType.FP16)
        ]
        
        for method, quant_type in methods_to_test:
            try:
                # Create temporary compressor with specific settings
                temp_compressor = AdvancedKVCompressor(
                    compression_method=method,
                    quantization_type=quant_type
                )
                
                # Compress and decompress
                compressed = temp_compressor.compress(kv_cache)
                reconstructed = temp_compressor.decompress(compressed)
                
                # Verify reconstruction quality
                total_error = 0.0
                for (orig_k, orig_v), (rec_k, rec_v) in zip(kv_cache, reconstructed):
                    total_error += torch.norm(orig_k.float() - rec_k.float()).item()
                    total_error += torch.norm(orig_v.float() - rec_v.float()).item()
                
                results[f"{method.value}_{quant_type.value}"] = {
                    'compression_ratio': compressed.compression_stats.compression_ratio,
                    'compression_time_ms': compressed.compression_stats.compression_time_ms,
                    'decompression_time_ms': compressed.compression_stats.decompression_time_ms,
                    'reconstruction_error': total_error,
                    'original_size_bytes': compressed.compression_stats.original_size_bytes,
                    'compressed_size_bytes': compressed.compression_stats.compressed_size_bytes
                }
                
            except Exception as e:
                results[f"{method.value}_{quant_type.value}"] = {'error': str(e)}
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage and compression statistics."""
        base_stats = {
            "memory_items": len(self._memory_cache),
            "disk_items": len(self._disk_items),
            "total_items": len(self._memory_cache) + len(self._disk_items),
            "max_memory": self.max_memory_items,
            "persist_to_disk": self.persist_to_disk,
            "compress": self.compress,
            "compression_method": self.compressor.compression_method.value,
            "quantization_type": self.compressor.quantization_type.value
        }
        
        if self.enable_benchmarking:
            base_stats.update({
                "compression_stats": dict(self._global_stats)
            })
        
        return base_stats
    
    def validate_reconstruction(self, cache_id: str, tolerance: float = 1e-3) -> Dict[str, Any]:
        """
        Validate that a compressed KV-cache can be perfectly reconstructed.
        
        Args:
            cache_id: ID of the cache to validate
            tolerance: Maximum allowed reconstruction error
            
        Returns:
            Dictionary with validation results
        """
        if cache_id not in self._memory_cache:
            return {'error': f'Cache {cache_id} not found in memory'}
        
        try:
            # Get original cache
            original_cache = self._memory_cache[cache_id]
            
            # Compress and decompress
            compressed = self.compressor.compress(original_cache)
            reconstructed_cache = self.compressor.decompress(compressed)
            
            # Compute reconstruction errors per layer
            layer_errors = []
            max_error = 0.0
            
            for i, ((orig_k, orig_v), (rec_k, rec_v)) in enumerate(zip(original_cache, reconstructed_cache)):
                key_error = torch.norm(orig_k.float() - rec_k.float()).item()
                value_error = torch.norm(orig_v.float() - rec_v.float()).item()
                layer_error = key_error + value_error
                layer_errors.append({
                    'layer': i,
                    'key_error': key_error,
                    'value_error': value_error,
                    'total_error': layer_error
                })
                max_error = max(max_error, layer_error)
            
            validation_passed = max_error <= tolerance
            
            return {
                'validation_passed': validation_passed,
                'max_reconstruction_error': max_error,
                'tolerance': tolerance,
                'layer_errors': layer_errors,
                'compression_stats': {
                    'compression_ratio': compressed.compression_stats.compression_ratio,
                    'original_size_bytes': compressed.compression_stats.original_size_bytes,
                    'compressed_size_bytes': compressed.compression_stats.compressed_size_bytes,
                    'method_used': compressed.compression_stats.method_used.value
                }
            }
            
        except Exception as e:
            return {'error': f'Validation failed: {str(e)}'}
    
    def optimize_compression_settings(self, sample_cache: KVCacheType, 
                                    target_compression_ratio: float = 10.0,
                                    max_reconstruction_error: float = 1e-3) -> Dict[str, Any]:
        """
        Automatically optimize compression settings for given constraints.
        
        Args:
            sample_cache: Sample KV-cache to use for optimization
            target_compression_ratio: Desired compression ratio
            max_reconstruction_error: Maximum allowed reconstruction error
            
        Returns:
            Dictionary with optimized settings and performance metrics
        """
        best_settings = None
        best_score = float('-inf')
        results = []
        
        # Parameter grid for optimization
        quantization_types = [QuantizationType.FP32, QuantizationType.FP16, QuantizationType.INT8]
        svd_ratios = [0.5, 0.7, 0.8, 0.9, 0.95]
        sparsity_thresholds = [1e-5, 1e-4, 1e-3, 1e-2]
        
        for quant_type in quantization_types:
            for svd_ratio in svd_ratios:
                for sparsity_thresh in sparsity_thresholds:
                    try:
                        # Create temporary compressor
                        temp_compressor = AdvancedKVCompressor(
                            compression_method=CompressionMethod.HYBRID,
                            quantization_type=quant_type,
                            svd_rank_ratio=svd_ratio,
                            sparsity_threshold=sparsity_thresh,
                            adaptive_compression=True
                        )
                        
                        # Test compression
                        compressed = temp_compressor.compress(sample_cache)
                        reconstructed = temp_compressor.decompress(compressed)
                        
                        # Calculate reconstruction error
                        total_error = 0.0
                        for (orig_k, orig_v), (rec_k, rec_v) in zip(sample_cache, reconstructed):
                            total_error += torch.norm(orig_k.float() - rec_k.float()).item()
                            total_error += torch.norm(orig_v.float() - rec_v.float()).item()
                        
                        # Check constraints
                        ratio = compressed.compression_stats.compression_ratio
                        meets_ratio_target = ratio >= target_compression_ratio * 0.8  # 20% tolerance
                        meets_error_target = total_error <= max_reconstruction_error
                        
                        if meets_error_target:  # Error constraint is hard
                            # Score based on compression ratio (higher is better)
                            score = ratio
                            
                            if meets_ratio_target:
                                score += 100  # Bonus for meeting both constraints
                            
                            if score > best_score:
                                best_score = score
                                best_settings = {
                                    'quantization_type': quant_type,
                                    'svd_rank_ratio': svd_ratio,
                                    'sparsity_threshold': sparsity_thresh,
                                    'compression_ratio': ratio,
                                    'reconstruction_error': total_error,
                                    'compression_time_ms': compressed.compression_stats.compression_time_ms,
                                    'meets_targets': meets_ratio_target
                                }
                        
                        results.append({
                            'settings': {
                                'quantization_type': quant_type.value,
                                'svd_rank_ratio': svd_ratio,
                                'sparsity_threshold': sparsity_thresh
                            },
                            'performance': {
                                'compression_ratio': ratio,
                                'reconstruction_error': total_error,
                                'compression_time_ms': compressed.compression_stats.compression_time_ms,
                                'meets_ratio_target': meets_ratio_target,
                                'meets_error_target': meets_error_target
                            }
                        })
                        
                    except Exception as e:
                        logging.warning(f"Optimization failed for settings {quant_type.value}, {svd_ratio}, {sparsity_thresh}: {e}")
                        continue
        
        return {
            'best_settings': best_settings,
            'all_results': results,
            'target_compression_ratio': target_compression_ratio,
            'max_reconstruction_error': max_reconstruction_error
        }
    
    def __len__(self) -> int:
        """Return total number of cached items."""
        return len(self._memory_cache) + len(self._disk_items)
    
    def __contains__(self, cache_id: str) -> bool:
        """Support 'in' operator for checking cache existence."""
        return self.exists(cache_id)