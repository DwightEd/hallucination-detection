"""Core data types for hallucination detection framework.

All components use these unified data structures to ensure consistency.
Design principle: Simple, flat structures that are easy to serialize.

修复:
- 改进懒加载方法的错误处理
- 添加更好的日志
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pathlib import Path
import json
import logging
import torch

logger = logging.getLogger(__name__)


# ==============================================================================
# Enums
# ==============================================================================

class TaskType(str, Enum):
    """Task type for dataset samples."""
    QA = "QA"
    SUMMARY = "Summary"
    DATA2TXT = "Data2txt"
    DIALOGUE = "Dialogue"
    MATH = "Math"
    OTHER = "Other"


class SplitType(str, Enum):
    """Data split type."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ExtractionMode(str, Enum):
    """Feature extraction mode."""
    TEACHER_FORCING = "teacher_forcing"
    GENERATION = "generation"


class ClassificationLevel(str, Enum):
    """Classification level for hallucination detection."""
    SAMPLE = "sample"     # Sample-level classification (whole response)
    TOKEN = "token"       # Token-level classification (per token)
    BOTH = "both"         # Train both classifiers


class StorageMode(str, Enum):
    """Feature storage mode."""
    DIAGONAL = "diagonal"  # Only store diagonals (lapeigvals style, memory efficient)
    FULL = "full"          # Store full matrices


# ==============================================================================
# Data Samples - Unified format for all datasets
# ==============================================================================

@dataclass
class Sample:
    """Unified sample format for all datasets.
    
    All dataset parsers MUST output this format for consistency.
    This ensures any dataset can work with any detection method.
    
    Attributes:
        id: Unique sample identifier
        prompt: Complete prompt (including context if any)
        response: Model's response/answer
        reference: Ground truth answer (if available)
        label: Hallucination label (0=correct, 1=hallucinated, None=unknown)
        task_type: Type of task
        split: Data split (train/val/test)
        metadata: Dataset-specific additional information
    """
    id: str
    prompt: str
    response: str
    reference: str = ""
    label: Optional[int] = None
    task_type: TaskType = TaskType.QA
    split: Optional[SplitType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "response": self.response,
            "reference": self.reference,
            "label": self.label,
            "task_type": self.task_type.value,
            "split": self.split.value if self.split else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sample":
        """Create from dictionary."""
        return cls(
            id=str(data.get("id", "")),
            prompt=str(data.get("prompt", "")),
            response=str(data.get("response", "")),
            reference=str(data.get("reference", "")),
            label=data.get("label"),
            task_type=TaskType(data.get("task_type", "qa")),
            split=SplitType(data["split"]) if data.get("split") else None,
            metadata=data.get("metadata", {}),
        )
    
    def __repr__(self) -> str:
        prompt_preview = self.prompt[:50] if len(self.prompt) > 50 else self.prompt
        return f"Sample(id={self.id!r}, label={self.label}, prompt={prompt_preview!r}...)"


# ==============================================================================
# Feature Data Structures
# ==============================================================================

@dataclass
class ExtractedFeatures:
    """Extracted features from a single sample.
    
    Contains raw data that can be used by any detection method.
    Storage format follows lapeigvals convention for compatibility.
    
    Lazy Loading Support:
        Large features (hidden_states, full_attentions) can be stored on disk
        and loaded on demand via get_hidden_states() and get_full_attention().
        File paths are stored in metadata["_feature_paths"].
    """
    sample_id: str
    prompt_len: int
    response_len: int
    
    # Attention (stored as diagonal only by default)
    attn_diags: Optional[torch.Tensor] = None
    attn_row_sums: Optional[torch.Tensor] = None  # NEW: for lapeigvals
    laplacian_diags: Optional[torch.Tensor] = None
    attn_entropy: Optional[torch.Tensor] = None
    full_attention: Optional[torch.Tensor] = None  # [n_layers, n_heads, seq_len, seq_len]
    
    # Hidden states (pooled)
    hidden_states: Optional[torch.Tensor] = None
    
    # Token probabilities
    token_probs: Optional[torch.Tensor] = None
    token_entropy: Optional[torch.Tensor] = None
    top_k_probs: Optional[torch.Tensor] = None
    top_k_indices: Optional[torch.Tensor] = None
    perplexity: Optional[float] = None
    
    # Token-level hallucination labels (following RAGTruth format)
    # 0 = not hallucinated, 1 = hallucinated, for each token in input
    hallucination_labels: Optional[List[int]] = None
    # Token-level hallucination spans [[start_token_idx, end_token_idx_exclusive], ...]
    hallucination_token_spans: Optional[List[List[int]]] = None
    
    # Metadata
    label: Optional[int] = None
    layers: List[int] = field(default_factory=list)
    model_name: str = ""
    mode: ExtractionMode = ExtractionMode.TEACHER_FORCING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save features to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "sample_id": self.sample_id,
            "prompt_len": self.prompt_len,
            "response_len": self.response_len,
            "label": self.label,
            "layers": self.layers,
            "model_name": self.model_name,
            "mode": self.mode.value,
            "metadata": self.metadata,
        }
        
        # Save tensors
        tensors = {}
        for name in ["attn_diags", "attn_row_sums", "laplacian_diags", "attn_entropy", "full_attention",
                     "hidden_states", "token_probs", "token_entropy",
                     "top_k_probs", "top_k_indices"]:
            val = getattr(self, name)
            if val is not None:
                tensors[name] = val.cpu()
        
        if self.perplexity is not None:
            data["perplexity"] = self.perplexity
        
        # Save hallucination labels
        if self.hallucination_labels is not None:
            data["hallucination_labels"] = self.hallucination_labels
        if self.hallucination_token_spans is not None:
            data["hallucination_token_spans"] = self.hallucination_token_spans
        
        torch.save({"info": data, "tensors": tensors}, path)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExtractedFeatures":
        """Load features from file."""
        saved = torch.load(path, map_location="cpu")
        info = saved["info"]
        tensors = saved["tensors"]
        
        return cls(
            sample_id=info["sample_id"],
            prompt_len=info["prompt_len"],
            response_len=info["response_len"],
            label=info.get("label"),
            layers=info.get("layers", []),
            model_name=info.get("model_name", ""),
            mode=ExtractionMode(info.get("mode", "teacher_forcing")),
            metadata=info.get("metadata", {}),
            perplexity=info.get("perplexity"),
            hallucination_labels=info.get("hallucination_labels"),
            hallucination_token_spans=info.get("hallucination_token_spans"),
            attn_diags=tensors.get("attn_diags"),
            attn_row_sums=tensors.get("attn_row_sums"),
            laplacian_diags=tensors.get("laplacian_diags"),
            attn_entropy=tensors.get("attn_entropy"),
            full_attention=tensors.get("full_attention"),
            hidden_states=tensors.get("hidden_states"),
            token_probs=tensors.get("token_probs"),
            token_entropy=tensors.get("token_entropy"),
            top_k_probs=tensors.get("top_k_probs"),
            top_k_indices=tensors.get("top_k_indices"),
        )
    
    # ==========================================================================
    # Lazy Loading Methods
    # ==========================================================================
    
    def load_large_feature(self, feature_key: str) -> Optional[torch.Tensor]:
        """Lazy-load a large feature from file on demand.
        
        Large features (hidden_states, full_attentions) are not loaded into memory
        during dataset loading to avoid OOM. Instead, file paths are stored in
        metadata["_feature_paths"]. This method loads them when actually needed.
        
        Args:
            feature_key: "hidden_states" or "full_attentions"
            
        Returns:
            The feature tensor, or None if not available
        """
        # First check if already loaded
        attr_name = "full_attention" if feature_key == "full_attentions" else feature_key
        current_value = getattr(self, attr_name, None)
        if current_value is not None:
            return current_value
        
        # Check if path is stored in metadata
        feature_paths = self.metadata.get("_feature_paths", {})
        if feature_key not in feature_paths:
            logger.debug(f"No path found for {feature_key} in sample {self.sample_id}")
            return None
        
        file_path = feature_paths[feature_key]
        
        # 验证文件路径
        if not file_path:
            logger.debug(f"Empty path for {feature_key} in sample {self.sample_id}")
            return None
        
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"Feature file not found: {file_path}")
            return None
        
        try:
            logger.debug(f"Loading {feature_key} from {file_path}")
            data = torch.load(file_path, map_location="cpu", weights_only=False)
            
            # 尝试多种可能的数据结构
            value = None
            
            # 格式 1: {"features": {"full_attentions": tensor}}
            if "features" in data:
                features = data["features"]
                if isinstance(features, dict):
                    value = features.get(feature_key)
            
            # 格式 2: 直接是 tensor
            if value is None and isinstance(data, torch.Tensor):
                value = data
            
            # 格式 3: {"full_attentions": tensor}
            if value is None and feature_key in data:
                value = data[feature_key]
            
            # Cache the loaded value
            if value is not None:
                if isinstance(value, torch.Tensor):
                    setattr(self, attr_name, value)
                    logger.debug(f"Loaded {feature_key} with shape {value.shape}")
                else:
                    logger.warning(f"Loaded {feature_key} is not a tensor: {type(value)}")
                    return None
            else:
                logger.warning(f"Could not find {feature_key} in file {file_path}")
                logger.debug(f"File keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
            
            # Clean up
            del data
            
            return value
            
        except Exception as e:
            logger.warning(f"Failed to load {feature_key} for sample {self.sample_id}: {e}")
            return None
    
    def get_hidden_states(self) -> Optional[torch.Tensor]:
        """Get hidden_states, loading from file if necessary.
        
        Returns:
            Hidden states tensor, or None if not available
        """
        if self.hidden_states is not None:
            return self.hidden_states
        return self.load_large_feature("hidden_states")
    
    def get_full_attention(self) -> Optional[torch.Tensor]:
        """Get full_attention, loading from file if necessary.
        
        Returns:
            Full attention tensor, or None if not available
        """
        if self.full_attention is not None:
            return self.full_attention
        return self.load_large_feature("full_attentions")
    
    def release_large_features(self) -> None:
        """Release large features from memory after use.
        
        Call this after processing to free up memory.
        The features can be reloaded later via get_hidden_states() or get_full_attention().
        """
        self.hidden_states = None
        self.full_attention = None


# ==============================================================================
# Evaluation Results
# ==============================================================================

@dataclass
class Prediction:
    """Single prediction result."""
    sample_id: str
    score: float       # Probability of hallucination [0, 1]
    label: int         # Binary prediction (0 or 1)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "score": self.score,
            "label": self.label,
            "confidence": self.confidence,
        }


@dataclass
class HallucinationSpan:
    """A span of hallucinated text within a response.
    
    Character positions are relative to the response text.
    Following RAGTruth format for consistency.
    """
    start: int           # Start character position (inclusive)
    end: int             # End character position (exclusive)
    text: str = ""       # The hallucinated text content
    label_type: str = "" # Type: "intrinsic", "extrinsic", "fabrication", etc.
    explanation: str = "" # Why this is hallucinated
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "label_type": self.label_type,
            "explanation": self.explanation,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HallucinationSpan":
        return cls(
            start=int(data.get("start", 0)),
            end=int(data.get("end", 0)),
            text=str(data.get("text", "")),
            label_type=str(data.get("label_type", data.get("type", ""))),
            explanation=str(data.get("explanation", "")),
        )


@dataclass
class JudgeResult:
    """LLM-as-Judge evaluation result.
    
    Format aligned with RAGTruth labels for consistency.
    Includes hallucination spans with character positions.
    """
    sample_id: str
    label: int           # 0=correct, 1=hallucinated (same as Sample.label)
    confidence: float    # Judge's confidence [0, 1]
    explanation: str     # Reasoning from judge
    raw_response: str    # Raw API response
    model: str = ""      # Judge model name
    hallucination_spans: List["HallucinationSpan"] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "label": self.label,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "raw_response": self.raw_response,
            "model": self.model,
            "hallucination_spans": [s.to_dict() for s in self.hallucination_spans],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JudgeResult":
        spans = [
            HallucinationSpan.from_dict(s)
            for s in data.get("hallucination_spans", [])
        ]
        return cls(
            sample_id=str(data["sample_id"]),
            label=int(data["label"]),
            confidence=float(data.get("confidence", 1.0)),
            explanation=str(data.get("explanation", "")),
            raw_response=str(data.get("raw_response", "")),
            model=str(data.get("model", "")),
            hallucination_spans=spans,
        )


@dataclass
class EvalMetrics:
    """Evaluation metrics container."""
    auroc: float = 0.0
    auprc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0
    threshold: float = 0.5
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    
    # Optional confidence intervals
    auroc_ci: Optional[tuple] = None
    auprc_ci: Optional[tuple] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "threshold": self.threshold,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "auroc_ci": self.auroc_ci,
            "auprc_ci": self.auprc_ci,
        }
    
    def __repr__(self) -> str:
        return (
            f"EvalMetrics(AUROC={self.auroc:.4f}, AUPRC={self.auprc:.4f}, "
            f"F1={self.f1:.4f}, n={self.n_samples})"
        )
