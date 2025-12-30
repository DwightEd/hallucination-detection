"""Evaluation metrics for hallucination detection.

Provides:
- Binary classification metrics (AUROC, F1, Precision, Recall, Accuracy)
- Threshold-based metrics
- Calibration metrics
- Comparison utilities
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
import logging
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)

from src.core import Prediction, EvalMetrics

logger = logging.getLogger(__name__)


@dataclass
class DetailedMetrics:
    """Detailed evaluation metrics with confidence intervals."""
    
    # Core metrics
    auroc: float = 0.0
    auprc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0
    
    # Threshold-specific
    threshold: float = 0.5
    
    # Confusion matrix
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    
    # Additional
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    
    # Per-class metrics
    precision_per_class: Dict[int, float] = field(default_factory=dict)
    recall_per_class: Dict[int, float] = field(default_factory=dict)
    
    # Calibration
    ece: Optional[float] = None  # Expected Calibration Error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "threshold": self.threshold,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "ece": self.ece,
        }
    
    def __str__(self) -> str:
        """Format as string."""
        return (
            f"AUROC={self.auroc:.4f} | AUPRC={self.auprc:.4f} | "
            f"F1={self.f1:.4f} | P={self.precision:.4f} | R={self.recall:.4f} | "
            f"Acc={self.accuracy:.4f}"
        )


def compute_metrics(
    predictions: List[Prediction],
    labels: Optional[List[int]] = None,
    threshold: float = 0.5,
) -> DetailedMetrics:
    """Compute comprehensive evaluation metrics.
    
    Args:
        predictions: List of predictions with scores
        labels: Ground truth labels (if not in predictions)
        threshold: Classification threshold
        
    Returns:
        DetailedMetrics instance
    """
    # Extract scores and labels
    scores = np.array([p.score for p in predictions])
    
    if labels is not None:
        y_true = np.array(labels)
    else:
        y_true = np.array([p.label for p in predictions])
    
    # Binary predictions
    y_pred = (scores >= threshold).astype(int)
    
    metrics = DetailedMetrics(
        n_samples=len(predictions),
        n_positive=int(y_true.sum()),
        n_negative=int(len(y_true) - y_true.sum()),
        threshold=threshold,
    )
    
    # Check for valid data
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in labels, some metrics undefined")
        metrics.accuracy = accuracy_score(y_true, y_pred)
        return metrics
    
    # Core metrics
    try:
        metrics.auroc = float(roc_auc_score(y_true, scores))
    except Exception as e:
        logger.warning(f"Failed to compute AUROC: {e}")
        metrics.auroc = 0.0
    
    try:
        metrics.auprc = float(average_precision_score(y_true, scores))
    except Exception as e:
        logger.warning(f"Failed to compute AUPRC: {e}")
        metrics.auprc = 0.0
    
    metrics.f1 = float(f1_score(y_true, y_pred, zero_division=0))
    metrics.precision = float(precision_score(y_true, y_pred, zero_division=0))
    metrics.recall = float(recall_score(y_true, y_pred, zero_division=0))
    metrics.accuracy = float(accuracy_score(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics.tn, metrics.fp, metrics.fn, metrics.tp = cm.ravel()
    
    # Per-class metrics
    for cls in [0, 1]:
        cls_mask = y_true == cls
        if cls_mask.sum() > 0:
            cls_pred = y_pred[cls_mask]
            metrics.precision_per_class[cls] = float((cls_pred == cls).mean())
            metrics.recall_per_class[cls] = float(recall_score(
                y_true == cls, y_pred == cls, zero_division=0
            ))
    
    # Expected Calibration Error
    metrics.ece = compute_ece(scores, y_true)
    
    return metrics


def compute_ece(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.
    
    Args:
        scores: Predicted probabilities
        labels: Ground truth labels
        n_bins: Number of bins
        
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_mask = (scores >= bin_boundaries[i]) & (scores < bin_boundaries[i + 1])
        if bin_mask.sum() > 0:
            bin_conf = scores[bin_mask].mean()
            bin_acc = labels[bin_mask].mean()
            ece += bin_mask.sum() * abs(bin_acc - bin_conf)
    
    return float(ece / len(scores)) if len(scores) > 0 else 0.0


def find_optimal_threshold(
    predictions: List[Prediction],
    labels: Optional[List[int]] = None,
    metric: str = "f1",
) -> Tuple[float, float]:
    """Find optimal classification threshold.
    
    Args:
        predictions: List of predictions
        labels: Ground truth labels
        metric: Metric to optimize (f1, precision, recall, accuracy)
        
    Returns:
        (optimal_threshold, best_metric_value)
    """
    scores = np.array([p.score for p in predictions])
    
    if labels is not None:
        y_true = np.array(labels)
    else:
        y_true = np.array([p.label for p in predictions])
    
    best_threshold = 0.5
    best_value = 0.0
    
    # Try different thresholds
    thresholds = np.linspace(0.01, 0.99, 99)
    
    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        
        if metric == "f1":
            value = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            value = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            value = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "accuracy":
            value = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if value > best_value:
            best_value = value
            best_threshold = thresh
    
    return float(best_threshold), float(best_value)


def compute_roc_curve(
    predictions: List[Prediction],
    labels: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """Compute ROC curve data.
    
    Args:
        predictions: List of predictions
        labels: Ground truth labels
        
    Returns:
        Dictionary with fpr, tpr, thresholds
    """
    scores = np.array([p.score for p in predictions])
    
    if labels is not None:
        y_true = np.array(labels)
    else:
        y_true = np.array([p.label for p in predictions])
    
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def compute_pr_curve(
    predictions: List[Prediction],
    labels: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """Compute Precision-Recall curve data.
    
    Args:
        predictions: List of predictions
        labels: Ground truth labels
        
    Returns:
        Dictionary with precision, recall, thresholds
    """
    scores = np.array([p.score for p in predictions])
    
    if labels is not None:
        y_true = np.array(labels)
    else:
        y_true = np.array([p.label for p in predictions])
    
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    
    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
    }


def compare_methods(
    method_predictions: Dict[str, List[Prediction]],
    labels: List[int],
) -> Dict[str, DetailedMetrics]:
    """Compare multiple methods.
    
    Args:
        method_predictions: Dictionary of method_name -> predictions
        labels: Ground truth labels
        
    Returns:
        Dictionary of method_name -> metrics
    """
    results = {}
    
    for method_name, predictions in method_predictions.items():
        metrics = compute_metrics(predictions, labels)
        results[method_name] = metrics
        logger.info(f"{method_name}: {metrics}")
    
    return results


def format_metrics_table(
    method_metrics: Dict[str, DetailedMetrics],
    metrics_to_show: List[str] = None,
) -> str:
    """Format metrics as a table string.
    
    Args:
        method_metrics: Dictionary of method_name -> metrics
        metrics_to_show: Which metrics to include
        
    Returns:
        Formatted table string
    """
    if metrics_to_show is None:
        metrics_to_show = ["auroc", "auprc", "f1", "precision", "recall", "accuracy"]
    
    # Header
    header = "Method" + "".join(f"\t{m.upper()}" for m in metrics_to_show)
    lines = [header, "-" * len(header)]
    
    # Rows
    for method_name, metrics in method_metrics.items():
        row = method_name
        for m in metrics_to_show:
            value = getattr(metrics, m, 0.0)
            row += f"\t{value:.4f}"
        lines.append(row)
    
    return "\n".join(lines)


class MetricsTracker:
    """Track metrics across multiple evaluations."""
    
    def __init__(self):
        self.history: List[DetailedMetrics] = []
        self.method_history: Dict[str, List[DetailedMetrics]] = {}
    
    def add(self, metrics: DetailedMetrics, method_name: str = "default") -> None:
        """Add metrics to history."""
        self.history.append(metrics)
        
        if method_name not in self.method_history:
            self.method_history[method_name] = []
        self.method_history[method_name].append(metrics)
    
    def get_summary(self, method_name: str = None) -> Dict[str, Dict[str, float]]:
        """Get summary statistics across evaluations.
        
        Returns:
            Dictionary with mean and std for each metric
        """
        if method_name:
            history = self.method_history.get(method_name, [])
        else:
            history = self.history
        
        if not history:
            return {}
        
        metrics_names = ["auroc", "auprc", "f1", "precision", "recall", "accuracy"]
        summary = {}
        
        for m in metrics_names:
            values = [getattr(h, m, 0.0) for h in history]
            summary[m] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        
        return summary
    
    def to_dataframe(self) -> Any:
        """Convert history to pandas DataFrame."""
        try:
            import pandas as pd
            records = [m.to_dict() for m in self.history]
            return pd.DataFrame(records)
        except ImportError:
            logger.warning("pandas not installed, returning list of dicts")
            return [m.to_dict() for m in self.history]
