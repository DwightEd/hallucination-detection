"""Performance Metrics Tracker for Hallucination Detection.

Records and reports:
- Training time
- Peak memory usage (CPU/GPU)
- Model size (serialized pickle file)

Usage:
    from src.utils.metrics_tracker import MetricsTracker
    
    tracker = MetricsTracker()
    tracker.start()
    
    # ... training code ...
    
    metrics = tracker.stop()
    tracker.set_model_path("model.pkl")
    report = tracker.get_report()
"""
import os
import time
import gc
import json
import logging
import tracemalloc
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    # Time metrics (seconds)
    training_time_seconds: float = 0.0
    
    # Memory metrics (MB)
    peak_cpu_memory_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    
    # Model size (bytes and human-readable)
    model_size_bytes: int = 0
    model_size_mb: float = 0.0
    
    # Additional info
    n_samples: int = 0
    n_features: int = 0
    method_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: Path) -> "PerformanceMetrics":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    def summary(self) -> str:
        lines = [
            "=" * 50,
            "Performance Metrics Summary",
            "=" * 50,
            f"Method: {self.method_name}" if self.method_name else "",
            f"Samples: {self.n_samples}",
            f"Features: {self.n_features}",
            "",
            "Time:",
            f"  Training: {self.training_time_seconds:.2f}s",
            "",
            "Memory:",
            f"  Peak CPU: {self.peak_cpu_memory_mb:.1f} MB",
            f"  Peak GPU: {self.peak_gpu_memory_mb:.1f} MB",
            "",
            "Model Size:",
            f"  {self.model_size_mb:.2f} MB ({self.model_size_bytes:,} bytes)",
            "=" * 50,
        ]
        return "\n".join(line for line in lines if line is not None)


class MetricsTracker:
    """Track performance metrics during training."""
    
    def __init__(self, method_name: str = ""):
        self.method_name = method_name
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._peak_cpu_memory: float = 0.0
        self._peak_gpu_memory: float = 0.0
        self._initial_cpu_memory: float = 0.0
        self._initial_gpu_memory: float = 0.0
        self._model_path: Optional[Path] = None
        self._n_samples: int = 0
        self._n_features: int = 0
        self._is_tracking: bool = False
    
    def start(self) -> "MetricsTracker":
        """Start tracking metrics."""
        self._start_time = time.time()
        self._is_tracking = True
        
        # Start CPU memory tracking
        tracemalloc.start()
        self._initial_cpu_memory = self._get_current_cpu_memory()
        
        # Record initial GPU memory
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self._initial_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        
        logger.debug("Metrics tracking started")
        return self
    
    def stop(self) -> PerformanceMetrics:
        """Stop tracking and return metrics."""
        if not self._is_tracking:
            logger.warning("Metrics tracking was not started")
            return self.get_metrics()
        
        self._end_time = time.time()
        self._is_tracking = False
        
        # Get peak CPU memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._peak_cpu_memory = peak / (1024 ** 2)  # Convert to MB
        
        # Get peak GPU memory
        if HAS_TORCH and torch.cuda.is_available():
            self._peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        logger.debug("Metrics tracking stopped")
        return self.get_metrics()
    
    def set_model_path(self, path: Path) -> None:
        """Set the model file path for size calculation."""
        self._model_path = Path(path)
    
    def set_sample_info(self, n_samples: int, n_features: int = 0) -> None:
        """Set sample information."""
        self._n_samples = n_samples
        self._n_features = n_features
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics."""
        training_time = 0.0
        if self._start_time:
            end = self._end_time or time.time()
            training_time = end - self._start_time
        
        model_size_bytes = 0
        if self._model_path and self._model_path.exists():
            model_size_bytes = os.path.getsize(self._model_path)
        
        return PerformanceMetrics(
            training_time_seconds=training_time,
            peak_cpu_memory_mb=self._peak_cpu_memory,
            peak_gpu_memory_mb=self._peak_gpu_memory,
            model_size_bytes=model_size_bytes,
            model_size_mb=model_size_bytes / (1024 ** 2),
            n_samples=self._n_samples,
            n_features=self._n_features,
            method_name=self.method_name,
        )
    
    def get_report(self) -> str:
        """Get human-readable report."""
        return self.get_metrics().summary()
    
    def _get_current_cpu_memory(self) -> float:
        """Get current CPU memory usage in MB."""
        if HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 ** 2)
        return 0.0
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """Get GPU memory information."""
        if not HAS_TORCH or not torch.cuda.is_available():
            return {"available": False}
        
        info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
        }
        
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
            total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
            
            info[f"gpu_{i}_allocated_mb"] = allocated
            info[f"gpu_{i}_reserved_mb"] = reserved
            info[f"gpu_{i}_total_mb"] = total
        
        return info


@contextmanager
def track_metrics(method_name: str = ""):
    """Context manager for tracking metrics.
    
    Usage:
        with track_metrics("lapeigvals") as tracker:
            # training code
            tracker.set_sample_info(n_samples=1000, n_features=512)
        
        metrics = tracker.get_metrics()
    """
    tracker = MetricsTracker(method_name)
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()


def measure_model_size(model_path: Path) -> Dict[str, Any]:
    """Measure model file size.
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        Dict with size information
    """
    model_path = Path(model_path)
    if not model_path.exists():
        return {"exists": False, "path": str(model_path)}
    
    size_bytes = os.path.getsize(model_path)
    
    return {
        "exists": True,
        "path": str(model_path),
        "size_bytes": size_bytes,
        "size_kb": size_bytes / 1024,
        "size_mb": size_bytes / (1024 ** 2),
    }


def format_metrics_table(metrics_list: list) -> str:
    """Format multiple metrics as a comparison table.
    
    Args:
        metrics_list: List of PerformanceMetrics objects
        
    Returns:
        Formatted table string
    """
    if not metrics_list:
        return "No metrics to display"
    
    # Header
    header = f"{'Method':<20} {'Time(s)':<10} {'CPU(MB)':<10} {'GPU(MB)':<10} {'Model(MB)':<10}"
    separator = "-" * len(header)
    
    lines = [separator, header, separator]
    
    for m in metrics_list:
        row = f"{m.method_name:<20} {m.training_time_seconds:<10.2f} {m.peak_cpu_memory_mb:<10.1f} {m.peak_gpu_memory_mb:<10.1f} {m.model_size_mb:<10.2f}"
        lines.append(row)
    
    lines.append(separator)
    
    return "\n".join(lines)
