"""Utility functions for hallucination detection framework."""
from __future__ import annotations
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Any, Dict, Iterator, TypeVar
from contextlib import contextmanager

T = TypeVar('T')

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    """Get logger by name."""
    return logging.getLogger(name)


# ==============================================================================
# Exceptions
# ==============================================================================

class HallucDetectError(Exception):
    """Base exception for framework errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)
    
    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class DatasetError(HallucDetectError):
    """Dataset-related errors."""
    pass


class ModelError(HallucDetectError):
    """Model loading/inference errors."""
    pass


class FeatureError(HallucDetectError):
    """Feature extraction errors."""
    pass


class MethodError(HallucDetectError):
    """Detection method errors."""
    pass


class APIError(HallucDetectError):
    """LLM API errors."""
    pass


# ==============================================================================
# Progress Tracking
# ==============================================================================

class Progress:
    """Simple progress tracker for batch operations."""
    
    def __init__(self, total: int, desc: str = "Progress", logger: Optional[logging.Logger] = None, log_interval: int = 10):
        self.total = total
        self.desc = desc
        self.logger = logger or logging.getLogger(__name__)
        self.log_interval = log_interval
        self.current = 0
        self.start_time = None
        self._last_logged_pct = -1
    
    def __enter__(self) -> "Progress":
        self.start_time = time.time()
        self.logger.info(f"{self.desc}: Starting ({self.total} items)")
        return self
    
    def __exit__(self, *args) -> None:
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        self.logger.info(f"{self.desc}: Completed {self.current}/{self.total} in {elapsed:.1f}s ({rate:.1f} it/s)")
    
    def update(self, n: int = 1) -> None:
        self.current += n
        if self.total > 0:
            pct = int(100 * self.current / self.total)
            if pct >= self._last_logged_pct + self.log_interval:
                elapsed = time.time() - self.start_time
                rate = self.current / elapsed if elapsed > 0 else 0
                eta = (self.total - self.current) / rate if rate > 0 else 0
                self.logger.info(f"{self.desc}: {self.current}/{self.total} ({pct}%) | {rate:.1f} it/s | ETA: {eta:.0f}s")
                self._last_logged_pct = pct


@contextmanager
def timer(desc: str, logger: Optional[logging.Logger] = None):
    """Context manager for timing operations."""
    logger = logger or logging.getLogger(__name__)
    logger.info(f"{desc}: Starting...")
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"{desc}: Completed in {elapsed:.2f}s")


# ==============================================================================
# Helper Functions
# ==============================================================================

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def batch_iter(items: list, batch_size: int) -> Iterator[list]:
    """Iterate over items in batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> str:
    """Get available device."""
    import torch
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device
