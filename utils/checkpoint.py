"""Checkpoint management for incremental processing with resume support.

Provides:
- Per-sample saving with immediate GPU memory release
- Resume from last checkpoint on failure
- Progress tracking and state persistence
- Memory-efficient streaming consolidation (fixes OOM issue)
"""
from __future__ import annotations
import gc
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Set, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import torch
import threading

from src.core import sanitize_sample_id

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
        force_restart: bool = False,
        sample_ids: Optional[List[str]] = None,  # 新增：期望的样本ID列表
    ) -> bool:
        """Initialize or resume checkpoint state.
        
        核心逻辑：
        1. 扫描 features_individual/ 目录中已存在的特征文件
        2. 与期望的样本列表对比
        3. 如果全部齐全 → 返回 True（跳过提取）
        4. 如果有缺失 → 只处理缺失的样本
        
        Args:
            total_samples: Total number of samples to process
            config: Configuration dict (仅用于日志记录，不影响是否重新提取)
            force_restart: Force restart even if checkpoint exists
            sample_ids: 期望的样本ID列表，用于完整性检查
            
        Returns:
            True if resuming from checkpoint, False if starting fresh
        """
        config_hash = self._hash_config(config) if config else ""
        
        if force_restart:
            logger.info("Force restart requested, clearing all progress")
            self._reset_state(total_samples, config_hash)
            return False
        
        # 核心逻辑：基于文件存在性检查
        existing_ids = self._scan_existing_features()
        
        if existing_ids:
            logger.info(f"Found {len(existing_ids)} existing feature files in {self.features_dir}")
            
            # 如果提供了期望的样本列表，检查完整性
            if sample_ids:
                expected_ids = set(sample_ids)
                missing_ids = expected_ids - existing_ids
                extra_ids = existing_ids - expected_ids
                
                if missing_ids:
                    logger.info(f"Missing {len(missing_ids)} samples, will extract them")
                else:
                    logger.info("✅ All samples already extracted, skipping extraction")
                
                if extra_ids:
                    logger.debug(f"Found {len(extra_ids)} extra files not in current sample list")
            
            # 重建状态
            self.state = CheckpointState(
                total_samples=total_samples,
                processed_samples=len(existing_ids),
                processed_ids=existing_ids,
                start_time=datetime.now().isoformat(),
                last_update=datetime.now().isoformat(),
                config_hash=config_hash,
            )
            self._save_state()
            return True
        
        # 没有已存在的文件，从头开始
        logger.info("No existing feature files found, starting fresh")
        self._reset_state(total_samples, config_hash)
        return False
    
    def _scan_existing_features(self) -> Set[str]:
        """扫描 features_individual 目录，返回已存在的样本ID集合。
        
        Returns:
            已存在特征文件的样本ID集合
        """
        existing_ids = set()
        
        if not self.features_dir.exists():
            logger.info(f"Features directory does not exist: {self.features_dir}")
            return existing_ids
        
        feature_files = list(self.features_dir.glob("*.pt"))
        logger.info(f"Scanning {self.features_dir}: found {len(feature_files)} .pt files")
        
        for f in feature_files:
            try:
                # 尝试从文件中读取真实的 sample_id
                data = torch.load(f, map_location="cpu", weights_only=False)
                sample_id = data.get("sample_id", f.stem)
                existing_ids.add(sample_id)
                del data
            except Exception as e:
                # 如果无法读取，使用文件名作为ID
                logger.debug(f"Failed to read {f}: {e}, using filename as ID")
                existing_ids.add(f.stem)
        
        return existing_ids
    
    def get_missing_samples(self, all_sample_ids: List[str]) -> List[str]:
        """获取缺失的样本ID列表。
        
        Args:
            all_sample_ids: 期望的所有样本ID
            
        Returns:
            缺失的样本ID列表
        """
        existing_ids = self._scan_existing_features()
        missing = [sid for sid in all_sample_ids if sid not in existing_ids]
        return missing
    
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
        safe_id = sanitize_sample_id(sample_id)
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
        safe_id = sanitize_sample_id(sample_id)
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
        
        ⚠️ MEMORY-OPTIMIZED VERSION: Processes one feature type at a time
        to avoid OOM when handling large feature sets (e.g., full_attentions).
        
        Args:
            output_path: Optional path to save consolidated features
            
        Returns:
            Dict with combined features, each feature type is a dict keyed by sample_id
        """
        feature_paths = self.get_all_feature_paths()
        
        if not feature_paths:
            logger.warning("No feature files found to consolidate")
            return {"sample_ids": [], "metadata": []}
        
        logger.info(f"Consolidating {len(feature_paths)} feature files (memory-optimized mode)...")
        
        # Feature keys to process (including full_attentions plural form)
        feature_keys = [
            "attn_diags", "laplacian_diags", "attn_entropy",
            "hidden_states", "token_probs", "token_entropy", 
            "full_attentions"
        ]
        
        # =========================================================================
        # Pass 1: Collect sample_ids and metadata only (low memory footprint)
        # =========================================================================
        logger.info("  Pass 1/2: Collecting sample IDs and metadata...")
        sample_ids = []
        metadata_list = []
        
        for path in feature_paths:
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                sample_id = data.get("sample_id", path.stem)
                sample_ids.append(sample_id)
                metadata_list.append(data.get("metadata", {}))
                # Clear immediately
                del data
            except Exception as e:
                logger.warning(f"Failed to load metadata from {path}: {e}")
        
        gc.collect()
        logger.info(f"  Found {len(sample_ids)} samples")
        
        # =========================================================================
        # Pass 2: Process each feature type SEPARATELY to avoid OOM
        # =========================================================================
        logger.info("  Pass 2/2: Processing features by type (streaming)...")
        
        result = {
            "sample_ids": sample_ids,
            "metadata": metadata_list,
        }
        
        for feature_key in feature_keys:
            feature_dict = {}
            found_count = 0
            
            for path in feature_paths:
                try:
                    data = torch.load(path, map_location="cpu", weights_only=False)
                    features = data.get("features", {})
                    sample_id = data.get("sample_id", path.stem)
                    
                    if feature_key in features and features[feature_key] is not None:
                        feature_dict[sample_id] = features[feature_key]
                        found_count += 1
                    
                    # Clear loaded data immediately to free memory
                    del data
                    del features
                    
                except Exception:
                    pass  # Skip individual file errors silently
            
            # Only add non-empty feature dicts to result
            if feature_dict:
                result[feature_key] = feature_dict
                logger.info(f"    {feature_key}: {found_count} samples")
            
            # Clear and force garbage collection before next feature type
            del feature_dict
            gc.collect()
            
            # Force CUDA cache clear if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"  Consolidation complete: {len(sample_ids)} samples total")
        
        if output_path:
            torch.save(result, output_path)
            logger.info(f"Saved consolidated features to {output_path}")
        
        return result
    
    def consolidate_features_streaming(
        self, 
        output_dir: Path,
        batch_size_for_large_features: int = 50,
    ) -> Dict[str, Any]:
        """Stream-consolidate features directly to disk, one type at a time.
        
        ⚠️ MEMORY-OPTIMIZED: 
        - Regular features: merge all samples into one file per feature type
        - Large features (full_attentions): keep as individual files or batch
        
        Args:
            output_dir: Directory to save consolidated feature files
            batch_size_for_large_features: Batch size for large features like full_attentions
            
        Returns:
            Dict with sample_ids, metadata, and info about saved files
        """
        feature_paths = self.get_all_feature_paths()
        
        if not feature_paths:
            logger.warning("No feature files found to consolidate")
            return {"sample_ids": [], "metadata": [], "saved_features": []}
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Streaming consolidation of {len(feature_paths)} files to {output_dir}...")
        
        # Regular features that can be merged (small per sample)
        # These are typically O(n_layers * n_heads * seq_len) or smaller
        regular_features = [
            "attn_diags", "laplacian_diags", "attn_entropy",
            "token_probs", "token_entropy",
            "hallucination_labels", "hallucination_token_spans",  # Token-level labels
        ]
        
        # Large features that should NOT be merged (huge per sample)
        # - hidden_states: O(n_layers * seq_len * hidden_dim) ≈ 256MB per sample for 7B model
        # - full_attentions: O(n_layers * n_heads * seq_len^2) ≈ GBs per sample
        # Merging 800+ samples = guaranteed OOM!
        large_features = ["hidden_states", "full_attentions"]
        
        # First pass: collect sample_ids and metadata
        sample_ids = []
        metadata_list = []
        
        logger.info("  Pass 1: Collecting sample IDs and metadata...")
        for path in feature_paths:
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                sample_ids.append(data.get("sample_id", path.stem))
                metadata_list.append(data.get("metadata", {}))
                del data
            except Exception as e:
                logger.warning(f"Failed to read {path}: {e}")
        
        gc.collect()
        logger.info(f"  Found {len(sample_ids)} samples")
        
        saved_features = []
        
        # =================================================================
        # Process REGULAR features (can be merged into single file)
        # =================================================================
        logger.info("  Pass 2: Processing regular features...")
        for feature_key in regular_features:
            feature_dict = {}
            found_count = 0
            
            for path in feature_paths:
                try:
                    data = torch.load(path, map_location="cpu", weights_only=False)
                    features = data.get("features", {})
                    sample_id = data.get("sample_id", path.stem)
                    
                    if feature_key in features and features[feature_key] is not None:
                        feature_dict[sample_id] = features[feature_key]
                        found_count += 1
                    
                    del data
                    del features
                except Exception:
                    pass
            
            # Save this feature type directly to disk
            if feature_dict:
                output_file = output_dir / f"{feature_key}.pt"
                torch.save(feature_dict, output_file)
                saved_features.append(feature_key)
                logger.info(f"    {feature_key}: {found_count} samples")
            
            # Clear memory before next feature type
            del feature_dict
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # =================================================================
        # Process LARGE features (keep as index pointing to individual files)
        # =================================================================
        logger.info("  Pass 3: Processing large features (index only)...")
        for feature_key in large_features:
            # Check if any sample has this feature
            has_feature = False
            feature_index = {}  # Maps sample_id -> original file path
            
            for path in feature_paths:
                try:
                    # Only load metadata to check if feature exists
                    data = torch.load(path, map_location="cpu", weights_only=False)
                    features = data.get("features", {})
                    sample_id = data.get("sample_id", path.stem)
                    
                    if feature_key in features and features[feature_key] is not None:
                        has_feature = True
                        # Store the path to the original file (for lazy loading)
                        feature_index[sample_id] = str(path)
                    
                    del data
                    del features
                except Exception:
                    pass
            
            if has_feature:
                # Save an INDEX file instead of merged data
                # This allows downstream code to load individual samples on demand
                index_file = output_dir / f"{feature_key}_index.json"
                import json
                with open(index_file, 'w') as f:
                    json.dump({
                        "type": "index",
                        "feature_key": feature_key,
                        "sample_count": len(feature_index),
                        "index": feature_index,
                        "note": "Large feature - load individual files on demand"
                    }, f, indent=2)
                
                saved_features.append(f"{feature_key}_index")
                logger.info(f"    {feature_key}: {len(feature_index)} samples (INDEX ONLY - files kept separate)")
            
            gc.collect()
        
        logger.info(f"  Consolidation complete")
        
        return {
            "sample_ids": sample_ids,
            "metadata": metadata_list,
            "saved_features": saved_features,
        }
    
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


def get_failed_samples(samples: List[Any], checkpoint_manager: CheckpointManager) -> List[Any]:
    """Get samples that previously failed.
    
    Args:
        samples: List of samples (must have .id attribute)
        checkpoint_manager: Checkpoint manager instance
        
    Returns:
        List of failed samples
    """
    failed_ids = checkpoint_manager.state.failed_ids
    return [s for s in samples if s.id in failed_ids]


def clear_failed_samples(checkpoint_manager: CheckpointManager) -> int:
    """Clear failed samples so they can be retried.
    
    Returns:
        Number of samples cleared
    """
    count = len(checkpoint_manager.state.failed_ids)
    checkpoint_manager.state.failed_ids.clear()
    checkpoint_manager._save_state()
    return count