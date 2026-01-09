"""Checkpoint management for incremental processing with resume support.

Provides:
- Per-sample saving with immediate GPU memory release
- Resume from last checkpoint on failure
- Progress tracking and state persistence
"""
from __future__ import annotations
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Set, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import torch
import threading

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Tracks processing state for resume capability."""
    total_samples: int = 0
    processed_samples: int = 0
    processed_ids: Set[str] = field(default_factory=set)
    failed_ids: Set[str] = field(default_factory=set)
    start_time: str = ""
    last_update: str = ""
    config_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "processed_samples": self.processed_samples,
            "processed_ids": list(self.processed_ids),
            "failed_ids": list(self.failed_ids),
            "start_time": self.start_time,
            "last_update": self.last_update,
            "config_hash": self.config_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        return cls(
            total_samples=data.get("total_samples", 0),
            processed_samples=data.get("processed_samples", 0),
            processed_ids=set(data.get("processed_ids", [])),
            failed_ids=set(data.get("failed_ids", [])),
            start_time=data.get("start_time", ""),
            last_update=data.get("last_update", ""),
            config_hash=data.get("config_hash", ""),
        )


class CheckpointManager:
    """Manages checkpoints for incremental processing with resume support.
    
    Features:
    - Saves processing state after each sample
    - Supports resume from interruption
    - Tracks failed samples for retry
    - Thread-safe state updates
    
    Usage:
        manager = CheckpointManager(output_dir)
        manager.initialize(total_samples, config)
        
        for sample in samples:
            if manager.is_processed(sample.id):
                continue
            
            try:
                features = extract(sample)
                manager.save_sample_features(sample.id, features)
                manager.mark_completed(sample.id)
            except Exception as e:
                manager.mark_failed(sample.id, str(e))
    """
    
    def __init__(self, output_dir: Path, checkpoint_file: str = "checkpoint.json"):
        """Initialize checkpoint manager.
        
        Args:
            output_dir: Directory for checkpoints and features
            checkpoint_file: Name of checkpoint state file
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_file = self.output_dir / checkpoint_file
        self.features_dir = self.output_dir / "features_individual"
        self.state = CheckpointState()
        self._lock = threading.Lock()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
    
    def initialize(
        self, 
        total_samples: int, 
        config: Optional[Dict[str, Any]] = None,
        force_restart: bool = False
    ) -> bool:
        """Initialize or resume checkpoint state.
        
        Args:
            total_samples: Total number of samples to process
            config: Configuration dict for hash comparison
            force_restart: Force restart even if checkpoint exists
            
        Returns:
            True if resuming from checkpoint, False if starting fresh
        """
        config_hash = self._hash_config(config) if config else ""
        
        if not force_restart and self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    data = json.load(f)
                self.state = CheckpointState.from_dict(data)
                
                # Verify config compatibility
                if config_hash and self.state.config_hash and config_hash != self.state.config_hash:
                    logger.warning("Config changed since last run, starting fresh")
                    self._reset_state(total_samples, config_hash)
                    return False
                
                logger.info(f"Resuming from checkpoint: {self.state.processed_samples}/{self.state.total_samples}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, starting fresh")
        
        self._reset_state(total_samples, config_hash)
        return False
    
    def _reset_state(self, total_samples: int, config_hash: str):
        """Reset state for fresh start."""
        self.state = CheckpointState(
            total_samples=total_samples,
            start_time=datetime.now().isoformat(),
            config_hash=config_hash,
        )
        self._save_state()
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate hash of config for comparison."""
        import hashlib
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def is_processed(self, sample_id: str) -> bool:
        """Check if sample has been processed."""
        with self._lock:
            return sample_id in self.state.processed_ids
    
    def mark_completed(self, sample_id: str):
        """Mark sample as successfully processed."""
        with self._lock:
            self.state.processed_ids.add(sample_id)
            self.state.processed_samples = len(self.state.processed_ids)
            self.state.last_update = datetime.now().isoformat()
            
            # Remove from failed if it was there
            self.state.failed_ids.discard(sample_id)
            
            self._save_state()
    
    def mark_failed(self, sample_id: str, error: str = ""):
        """Mark sample as failed."""
        with self._lock:
            self.state.failed_ids.add(sample_id)
            self.state.last_update = datetime.now().isoformat()
            self._save_state()
            
            if error:
                logger.warning(f"Sample {sample_id} failed: {error}")
    
    def _save_state(self):
        """Save checkpoint state to file."""
        try:
            temp_file = self.checkpoint_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
            temp_file.replace(self.checkpoint_file)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def save_sample_features(
        self, 
        sample_id: str, 
        features: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save features for a single sample.
        
        Args:
            sample_id: Sample identifier
            features: Dict of tensor features
            metadata: Optional metadata dict
            
        Returns:
            Path to saved feature file
        """
        # Sanitize sample_id for filename
        safe_id = sample_id.replace("/", "_").replace("\\", "_")[:100]
        feature_path = self.features_dir / f"{safe_id}.pt"
        
        # Move tensors to CPU before saving
        cpu_features = {}
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                cpu_features[key] = tensor.detach().cpu()
            else:
                cpu_features[key] = tensor
        
        save_data = {
            "sample_id": sample_id,
            "features": cpu_features,
            "metadata": metadata or {},
        }
        
        # Save with temp file for atomicity
        temp_path = feature_path.with_suffix(".tmp")
        torch.save(save_data, temp_path)
        temp_path.replace(feature_path)
        
        return feature_path
    
    def load_sample_features(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Load features for a single sample."""
        safe_id = sample_id.replace("/", "_").replace("\\", "_")[:100]
        feature_path = self.features_dir / f"{safe_id}.pt"
        
        if not feature_path.exists():
            return None
        
        try:
            return torch.load(feature_path, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning(f"Failed to load features for {sample_id}: {e}")
            return None
    
    def get_all_feature_paths(self) -> List[Path]:
        """Get paths to all saved feature files."""
        return sorted(self.features_dir.glob("*.pt"))
    
    def consolidate_features(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Consolidate all individual feature files into combined format.
        
        Args:
            output_path: Optional path to save consolidated features
            
        Returns:
            Dict with combined features, each feature type is a dict keyed by sample_id
        """
        feature_paths = self.get_all_feature_paths()
        
        # 使用 dict 格式，以 sample_id 为键（与 train_probe.py 期望的格式一致）
        # 注意：存储键使用 full_attentions（复数），与文件名一致
        consolidated = {
            "attn_diags": {},
            "laplacian_diags": {},
            "attn_entropy": {},
            "hidden_states": {},
            "token_probs": {},
            "token_entropy": {},
            "full_attentions": {},  # 使用复数形式，与存储文件名一致
            "sample_ids": [],
            "metadata": [],
        }
        
        # 支持的特征键列表（包括 full_attentions 复数形式）
        feature_keys = [
            "attn_diags", "laplacian_diags", "attn_entropy",
            "hidden_states", "token_probs", "token_entropy", 
            "full_attentions"
        ]
        
        for path in feature_paths:
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                features = data.get("features", {})
                sample_id = data.get("sample_id", path.stem)
                
                consolidated["sample_ids"].append(sample_id)
                consolidated["metadata"].append(data.get("metadata", {}))
                
                # 使用 sample_id 作为键存储特征
                for key in feature_keys:
                    if key in features and features[key] is not None:
                        consolidated[key][sample_id] = features[key]
                        
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
        
        # 移除空的 dict（只保留非空的特征类型）
        result = {}
        for k, v in consolidated.items():
            if k in ["sample_ids", "metadata"]:
                if v:  # 保留非空列表
                    result[k] = v
            elif isinstance(v, dict) and v:  # 保留非空 dict
                result[k] = v
        
        if output_path:
            torch.save(result, output_path)
            logger.info(f"Consolidated {len(feature_paths)} features to {output_path}")
        
        return result
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress info."""
        return {
            "total": self.state.total_samples,
            "processed": self.state.processed_samples,
            "failed": len(self.state.failed_ids),
            "remaining": self.state.total_samples - self.state.processed_samples,
            "progress_pct": (self.state.processed_samples / max(1, self.state.total_samples)) * 100,
        }
    
    def cleanup_temp_files(self):
        """Remove temporary files."""
        for temp_file in self.output_dir.glob("*.tmp"):
            try:
                temp_file.unlink()
            except Exception:
                pass


def get_pending_samples(samples: List[Any], checkpoint_manager: CheckpointManager) -> List[Any]:
    """Filter samples to only those not yet processed.
    
    Args:
        samples: List of samples (must have .id attribute)
        checkpoint_manager: Checkpoint manager instance
        
    Returns:
        List of unprocessed samples
    """
    return [s for s in samples if not checkpoint_manager.is_processed(s.id)]