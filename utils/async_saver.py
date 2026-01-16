"""Asynchronous feature saving with ThreadPoolExecutor.

Provides non-blocking saving of features while GPU continues processing.
Includes immediate tensor cleanup to prevent memory accumulation.
"""
from __future__ import annotations
import logging
import queue
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
import torch
import gc

from src.core import sanitize_sample_id

logger = logging.getLogger(__name__)


@dataclass
class SaveTask:
    """Represents a save task."""
    sample_id: str
    features: Dict[str, Any]
    metadata: Dict[str, Any]
    output_path: Path


class AsyncFeatureSaver:
    """Asynchronous feature saver using ThreadPoolExecutor.
    
    Saves features in background thread while main thread continues
    GPU processing. Ensures tensors are moved to CPU and original
    GPU tensors are immediately freed.
    
    Usage:
        saver = AsyncFeatureSaver(output_dir, max_workers=2)
        
        for sample in samples:
            features = extract_features(sample)  # GPU operation
            
            # Non-blocking save, frees GPU memory immediately
            saver.submit(sample.id, features, metadata)
            
        saver.wait_all()  # Wait for all saves to complete
        saver.shutdown()
    """
    
    def __init__(
        self, 
        output_dir: Path,
        max_workers: int = 2,
        on_save_complete: Optional[Callable[[str], None]] = None,
        on_save_error: Optional[Callable[[str, Exception], None]] = None
    ):
        """Initialize async saver.
        
        Args:
            output_dir: Directory to save features
            max_workers: Number of worker threads
            on_save_complete: Callback when save completes (sample_id)
            on_save_error: Callback on save error (sample_id, exception)
        """
        self.output_dir = Path(output_dir)
        self.features_dir = self.output_dir / "features_individual"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        
        self.on_save_complete = on_save_complete
        self.on_save_error = on_save_error
        
        self._completed_count = 0
        self._error_count = 0
    
    def submit(
        self,
        sample_id: str,
        features: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Future:
        """Submit features for async saving.
        
        IMPORTANT: This method immediately copies tensors to CPU and
        clears the original references to free GPU memory.
        
        Args:
            sample_id: Sample identifier
            features: Dict of tensor features (will be moved to CPU)
            metadata: Optional metadata
            
        Returns:
            Future for the save operation
        """
        # Immediately move tensors to CPU (this is synchronous but fast)
        cpu_features = self._prepare_features(features)
        
        # Submit to thread pool
        task = SaveTask(
            sample_id=sample_id,
            features=cpu_features,
            metadata=metadata or {},
            output_path=self._get_save_path(sample_id)
        )
        
        future = self.executor.submit(self._save_task, task)
        future.add_done_callback(lambda f: self._on_complete(sample_id, f))
        
        with self._lock:
            self.pending_futures[sample_id] = future
        
        return future
    
    def _prepare_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Move features to CPU and prepare for saving.
        
        This ensures GPU memory is freed as soon as possible.
        """
        cpu_features = {}
        
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                # Clone to CPU, detach from computation graph
                cpu_features[key] = value.detach().cpu().clone()
                
                # Clear original tensor to free GPU memory
                if value.device.type == "cuda":
                    del value
            elif isinstance(value, dict):
                cpu_features[key] = self._prepare_features(value)
            elif isinstance(value, list):
                cpu_features[key] = [
                    v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
            else:
                cpu_features[key] = value
        
        return cpu_features
    
    def _get_save_path(self, sample_id: str) -> Path:
        """Generate save path for sample."""
        safe_id = sanitize_sample_id(sample_id)
        return self.features_dir / f"{safe_id}.pt"
    
    def _save_task(self, task: SaveTask):
        """Execute save task (runs in thread pool)."""
        save_data = {
            "sample_id": task.sample_id,
            "features": task.features,
            "metadata": task.metadata,
        }
        
        # Atomic save with temp file
        temp_path = task.output_path.with_suffix(".tmp")
        torch.save(save_data, temp_path)
        temp_path.replace(task.output_path)
        
        return task.sample_id
    
    def _on_complete(self, sample_id: str, future: Future):
        """Handle save completion."""
        with self._lock:
            self.pending_futures.pop(sample_id, None)
        
        try:
            future.result()
            self._completed_count += 1
            if self.on_save_complete:
                self.on_save_complete(sample_id)
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to save {sample_id}: {e}")
            if self.on_save_error:
                self.on_save_error(sample_id, e)
    
    def wait_all(self, timeout: Optional[float] = None) -> bool:
        """Wait for all pending saves to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all completed, False if timeout
        """
        from concurrent.futures import wait, FIRST_EXCEPTION
        
        with self._lock:
            futures = list(self.pending_futures.values())
        
        if not futures:
            return True
        
        done, not_done = wait(futures, timeout=timeout)
        return len(not_done) == 0
    
    def get_pending_count(self) -> int:
        """Get number of pending save operations."""
        with self._lock:
            return len(self.pending_futures)
    
    def get_stats(self) -> Dict[str, int]:
        """Get save statistics."""
        return {
            "completed": self._completed_count,
            "errors": self._error_count,
            "pending": self.get_pending_count(),
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor.
        
        Args:
            wait: Whether to wait for pending tasks to complete
        """
        if wait:
            self.wait_all()
        self.executor.shutdown(wait=wait)


class MemoryEfficientSaver:
    """Memory-efficient saver that immediately releases tensors.
    
    Combines checkpoint management with async saving for maximum
    memory efficiency during large-scale feature extraction.
    """
    
    def __init__(
        self,
        output_dir: Path,
        max_workers: int = 2,
        checkpoint_interval: int = 10
    ):
        """Initialize memory-efficient saver.
        
        Args:
            output_dir: Output directory
            max_workers: Number of save workers
            checkpoint_interval: How often to force checkpoint save
        """
        self.output_dir = Path(output_dir)
        self.async_saver = AsyncFeatureSaver(
            output_dir,
            max_workers=max_workers,
            on_save_complete=self._on_save
        )
        
        self.checkpoint_interval = checkpoint_interval
        self._save_count = 0
        self._completed_ids = set()
        self._lock = threading.Lock()
    
    def _on_save(self, sample_id: str):
        """Callback when save completes."""
        with self._lock:
            self._completed_ids.add(sample_id)
    
    def save_and_release(
        self,
        sample_id: str,
        features: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
        force_gc: bool = True
    ):
        """Save features and immediately release GPU memory.
        
        Args:
            sample_id: Sample identifier
            features: Features dict (will be cleared after submission)
            metadata: Optional metadata
            force_gc: Whether to force garbage collection (now forces every time if True)
        """
        # Submit for async save
        self.async_saver.submit(sample_id, features, metadata)
        
        # Clear the original feature dict to release references
        features.clear()
        
        self._save_count += 1
        if force_gc:
            self._force_cleanup()
    
    def _force_cleanup(self):
        """Force GPU and CPU memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def is_saved(self, sample_id: str) -> bool:
        """Check if sample has been saved."""
        with self._lock:
            return sample_id in self._completed_ids
    
    def finalize(self) -> Dict[str, int]:
        """Wait for all saves and return stats."""
        self.async_saver.wait_all()
        self._force_cleanup()
        return self.async_saver.get_stats()
    
    def shutdown(self):
        """Shutdown the saver."""
        self.async_saver.shutdown()


def create_feature_saver(
    output_dir: Path,
    use_async: bool = True,
    max_workers: int = 2
) -> MemoryEfficientSaver:
    """Create a memory-efficient feature saver.
    
    Args:
        output_dir: Output directory
        use_async: Whether to use async saving
        max_workers: Number of worker threads
        
    Returns:
        MemoryEfficientSaver instance
    """
    return MemoryEfficientSaver(
        output_dir=output_dir,
        max_workers=max_workers if use_async else 1
    )
