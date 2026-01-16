"""Utility functions for hallucination detection framework.

Provides framework-level utilities:
- Logging setup
- Custom exceptions
- Progress tracking
- Randomness control
- Import helpers
"""
from __future__ import annotations
import logging
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, TypeVar, Optional, Any, Type

import numpy as np
import torch

T = TypeVar("T")


# ==============================================================================
# Logging
# ==============================================================================

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for logging
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger by name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# ==============================================================================
# Errors
# ==============================================================================

class HallucDetectError(Exception):
    """Base exception for hallucination detection framework."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{super().__str__()} ({details_str})"
        return super().__str__()


class DatasetError(HallucDetectError):
    """Dataset-related errors."""
    pass


class ModelError(HallucDetectError):
    """Model-related errors."""
    pass


class FeatureError(HallucDetectError):
    """Feature extraction errors."""
    pass


class MethodError(HallucDetectError):
    """Detection method errors."""
    pass


class APIError(HallucDetectError):
    """API-related errors."""
    pass


# ==============================================================================
# Progress & Timing
# ==============================================================================

class Progress:
    """Simple progress tracker with ETA estimation.
    
    Example:
        with Progress(100, "Processing") as pbar:
            for item in items:
                process(item)
                pbar.update()
    """
    
    def __init__(self, total: int, desc: str = "Progress"):
        """Initialize progress tracker.
        
        Args:
            total: Total number of items
            desc: Description for progress bar
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1) -> None:
        """Update progress.
        
        Args:
            n: Number of items completed
        """
        self.current += n
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        
        pct = 100 * self.current / self.total if self.total > 0 else 0
        print(f"\r{self.desc}: {self.current}/{self.total} ({pct:.1f}%) "
              f"[{elapsed:.1f}s elapsed, {eta:.1f}s remaining]", end="", flush=True)
        
        if self.current >= self.total:
            print()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if self.current < self.total:
            print()


@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing operations.
    
    Example:
        with timer("Feature extraction"):
            features = extract_features(samples)
        # Logs: "Feature extraction completed in X.XXs"
    """
    start = time.time()
    logger = logging.getLogger(__name__)
    logger.info(f"{name} started...")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"{name} completed in {elapsed:.2f}s")


# ==============================================================================
# Sample ID Utilities
# ==============================================================================

def sanitize_sample_id(sample_id: str, max_length: int = 100) -> str:
    """将 sample_id 转换为安全的文件名格式。
    
    统一的 sample_id 清理函数，避免在多处重复实现。
    
    转换规则：
    - 将 "/" 和 "\\" 替换为 "_"
    - 截断到 max_length 长度
    
    Args:
        sample_id: 原始样本 ID
        max_length: 最大长度（默认 100）
        
    Returns:
        安全的文件名字符串
        
    Example:
        >>> sanitize_sample_id("path/to/sample")
        'path_to_sample'
        >>> sanitize_sample_id("a" * 200)
        'aaa...aaa'  # 100 chars
    """
    safe_id = str(sample_id).replace("/", "_").replace("\\", "_")
    return safe_id[:max_length]


# ==============================================================================
# File & Directory Utilities
# ==============================================================================

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        The path (for chaining)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ==============================================================================
# Batch Utilities
# ==============================================================================

def batch_iter(items: List[T], batch_size: int) -> Iterator[List[T]]:
    """Iterate over items in batches.
    
    Args:
        items: List of items
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# ==============================================================================
# Randomness
# ==============================================================================

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Sets seed for Python random, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# Device Utilities
# ==============================================================================

def get_device(device: Optional[str] = None) -> str:
    """Get compute device.
    
    Args:
        device: Explicit device string (if None, auto-detect)
        
    Returns:
        Device string ("cuda" or "cpu")
    """
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# Import Utilities
# ==============================================================================

def import_class_from_path(cls_path: str) -> Type:
    """Import a class from a dot-separated path.
    
    Args:
        cls_path: Full path to class, e.g., 'src.data.ragtruth.RAGTruthDataset'
        
    Returns:
        The imported class
        
    Raises:
        ImportError: If the class cannot be imported
    """
    import importlib
    
    parts = cls_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ImportError(f"Invalid class path: {cls_path}")
    
    module_path, class_name = parts
    
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Cannot import '{class_name}' from '{module_path}': {e}")


def import_function_from_path(func_path: str):
    """Import a function from a dot-separated path.
    
    Args:
        func_path: Full path to function
        
    Returns:
        The imported function
    """
    return import_class_from_path(func_path)  # Same logic works for functions
