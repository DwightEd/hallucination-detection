"""GPU memory management utilities.

Provides functions for:
- Clearing GPU memory cache
- Getting GPU memory statistics
- Tracking memory usage during operations
"""
import gc
import logging
from typing import Dict, Optional
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


def clear_gpu_memory() -> None:
    """Clear GPU memory cache.
    
    Performs garbage collection and clears CUDA cache if available.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_info(device_id: int = 0) -> Dict[str, float]:
    """Get GPU memory usage info in GB.
    
    Args:
        device_id: CUDA device ID
        
    Returns:
        Dictionary with memory info:
        - allocated: Memory currently allocated by tensors
        - reserved: Memory reserved by the caching allocator
        - total: Total GPU memory
        - free: Available memory (total - reserved)
    """
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "total": 0.0, "free": 0.0}
    
    if device_id >= torch.cuda.device_count():
        logger.warning(f"Device {device_id} not available, using device 0")
        device_id = 0
    
    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
    total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
    free = total - reserved
    
    return {
        "allocated": round(allocated, 2),
        "reserved": round(reserved, 2),
        "total": round(total, 2),
        "free": round(free, 2),
    }


def log_gpu_memory(prefix: str = "", device_id: int = 0) -> None:
    """Log current GPU memory usage.
    
    Args:
        prefix: Optional prefix for log message
        device_id: CUDA device ID
    """
    if not torch.cuda.is_available():
        return
    
    info = get_gpu_memory_info(device_id)
    msg = f"{prefix} GPU Memory: {info['allocated']:.2f}GB allocated, {info['free']:.2f}GB free"
    logger.info(msg)


class MemoryTracker:
    """Context manager for tracking GPU memory usage during operations.
    
    Example:
        with MemoryTracker("Feature extraction") as tracker:
            features = extract_features(sample)
        # Logs memory delta after operation
    """
    
    def __init__(self, name: str = "Operation", device_id: int = 0, log: bool = True):
        """Initialize memory tracker.
        
        Args:
            name: Name of the operation being tracked
            device_id: CUDA device ID to track
            log: Whether to log memory delta on exit
        """
        self.name = name
        self.device_id = device_id
        self.log = log
        self.start_memory = 0.0
        self.end_memory = 0.0
        self.delta = 0.0
    
    def __enter__(self) -> "MemoryTracker":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated(self.device_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.end_memory = torch.cuda.memory_allocated(self.device_id)
            self.delta = (self.end_memory - self.start_memory) / 1024**2  # MB
            
            if self.log:
                logger.debug(f"[{self.name}] Memory delta: {self.delta:+.1f} MB")
    
    def get_delta_mb(self) -> float:
        """Get memory delta in MB."""
        return self.delta
    
    def get_delta_gb(self) -> float:
        """Get memory delta in GB."""
        return self.delta / 1024


@contextmanager
def gpu_memory_scope(clear_after: bool = True, clear_before: bool = False):
    """Context manager that optionally clears GPU memory before/after execution.
    
    Args:
        clear_after: Whether to clear memory after the block
        clear_before: Whether to clear memory before the block
        
    Example:
        with gpu_memory_scope(clear_after=True):
            # Memory-intensive operations
            pass
        # Memory is cleared here
    """
    if clear_before:
        clear_gpu_memory()
    
    try:
        yield
    finally:
        if clear_after:
            clear_gpu_memory()


def get_tensor_memory_mb(tensor: torch.Tensor) -> float:
    """Get memory usage of a tensor in MB.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        Memory usage in MB
    """
    return tensor.element_size() * tensor.nelement() / 1024**2


def estimate_batch_memory(
    sample_memory_mb: float,
    batch_size: int,
    overhead_factor: float = 1.5
) -> float:
    """Estimate memory needed for a batch.
    
    Args:
        sample_memory_mb: Memory for a single sample in MB
        batch_size: Number of samples in batch
        overhead_factor: Factor for gradients and intermediate values
        
    Returns:
        Estimated memory in MB
    """
    return sample_memory_mb * batch_size * overhead_factor
