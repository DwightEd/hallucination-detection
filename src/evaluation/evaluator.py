"""Unified evaluator for hallucination detection methods.

Provides:
- Method evaluation against ground truth
- LLM-as-Judge evaluation
- Cross-validation
- Comparison across methods
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import logging
import json

from src.core import Sample, ExtractedFeatures, Prediction, JudgeResult, LLMAPIConfig
from .metrics import DetailedMetrics, compute_metrics, find_optimal_threshold, MetricsTracker
from .llm_judge import LLMJudge, create_judge, JudgeMode

logger = logging.getLogger(__name__)


@dataclass 
class EvaluationResult:
    """Complete evaluation result."""
    
    method_name: str
    metrics: DetailedMetrics
    predictions: List[Prediction]
    optimal_threshold: float = 0.5
    judge_results: Optional[List[JudgeResult]] = None
    judge_metrics: Optional[DetailedMetrics] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "method_name": self.method_name,
            "metrics": self.metrics.to_dict(),
            "optimal_threshold": self.optimal_threshold,
            "n_predictions": len(self.predictions),
        }
        
        if self.judge_metrics:
            result["judge_metrics"] = self.judge_metrics.to_dict()
        
        return result
    
    def save(self, path: Union[str, Path]) -> None:
        """Save evaluation result to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved evaluation result to {path}")


class Evaluator:
    """Unified evaluator for hallucination detection.
    
    Supports both method-based and LLM-based evaluation.
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMAPIConfig] = None,
        judge_mode: str = "binary",
    ):
        """Initialize evaluator.
        
        Args:
            llm_config: LLM API config for judge (optional)
            judge_mode: Judge evaluation mode
        """
        self.llm_judge: Optional[LLMJudge] = None
        
        if llm_config:
            self.llm_judge = create_judge(llm_config, mode=judge_mode)
        
        self.tracker = MetricsTracker()
    
    def evaluate_method(
        self,
        method,  # BaseMethod
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
        method_name: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate a detection method.
        
        Args:
            method: Trained detection method
            features_list: List of extracted features
            labels: Ground truth labels
            method_name: Name for this evaluation
            
        Returns:
            EvaluationResult instance
        """
        method_name = method_name or method.__class__.__name__
        
        # Get predictions
        predictions = method.predict_batch(features_list)
        
        # Get labels
        if labels is None:
            labels = [f.label for f in features_list]
        
        # Compute metrics
        metrics = compute_metrics(predictions, labels)
        
        # Find optimal threshold
        opt_threshold, _ = find_optimal_threshold(predictions, labels)
        
        # Recompute metrics with optimal threshold
        metrics_opt = compute_metrics(predictions, labels, threshold=opt_threshold)
        
        logger.info(f"{method_name} evaluation: {metrics}")
        logger.info(f"Optimal threshold: {opt_threshold:.3f}, F1@opt: {metrics_opt.f1:.4f}")
        
        # Track metrics
        self.tracker.add(metrics, method_name)
        
        return EvaluationResult(
            method_name=method_name,
            metrics=metrics,
            predictions=predictions,
            optimal_threshold=opt_threshold,
        )
    
    def evaluate_with_judge(
        self,
        samples: List[Sample],
        predictions: Optional[List[Prediction]] = None,
    ) -> Dict[str, Any]:
        """Evaluate using LLM judge.
        
        Args:
            samples: Samples to evaluate
            predictions: Method predictions to compare (optional)
            
        Returns:
            Evaluation results with judge assessments
        """
        if self.llm_judge is None:
            raise ValueError("LLM judge not configured. Provide llm_config to Evaluator.")
        
        # Get judge results
        judge_results = self.llm_judge.judge_batch(samples)
        
        result = {
            "judge_results": judge_results,
            "n_hallucinated": sum(r.hallucinated for r in judge_results),
            "n_clean": sum(not r.hallucinated for r in judge_results),
        }
        
        # Compare with method predictions if provided
        if predictions:
            # Convert judge results to predictions
            judge_predictions = [
                Prediction(
                    sample_id=r.sample_id,
                    score=r.confidence if r.hallucinated else 1 - r.confidence,
                    label=1 if r.hallucinated else 0,
                )
                for r in judge_results
            ]
            
            # Use judge as ground truth
            judge_labels = [1 if r.hallucinated else 0 for r in judge_results]
            method_labels = [p.label for p in predictions]
            
            # Agreement metrics
            agreement = sum(j == m for j, m in zip(judge_labels, method_labels)) / len(judge_labels)
            result["agreement_with_method"] = agreement
            
            # Method metrics using judge as ground truth
            method_metrics = compute_metrics(predictions, judge_labels)
            result["method_vs_judge_metrics"] = method_metrics
        
        return result
    
    def cross_validate(
        self,
        method_class,  # Type[BaseMethod]
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
        n_folds: int = 5,
        method_config = None,
    ) -> Dict[str, Any]:
        """Cross-validate a method.
        
        Args:
            method_class: Method class to instantiate
            features_list: All features
            labels: All labels
            n_folds: Number of CV folds
            method_config: Config for method
            
        Returns:
            CV results with mean and std metrics
        """
        from sklearn.model_selection import StratifiedKFold
        import numpy as np
        
        if labels is None:
            labels = [f.label for f in features_list]
        
        labels_array = np.array(labels)
        features_array = np.array(features_list, dtype=object)
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(features_array, labels_array)):
            logger.info(f"CV Fold {fold + 1}/{n_folds}")
            
            # Split data
            train_features = [features_list[i] for i in train_idx]
            test_features = [features_list[i] for i in test_idx]
            train_labels = [labels[i] for i in train_idx]
            test_labels = [labels[i] for i in test_idx]
            
            # Train method
            method = method_class(method_config) if method_config else method_class()
            method.fit(train_features, train_labels, cv=False)
            
            # Evaluate
            predictions = method.predict_batch(test_features)
            metrics = compute_metrics(predictions, test_labels)
            fold_metrics.append(metrics)
        
        # Aggregate metrics
        metric_names = ["auroc", "auprc", "f1", "precision", "recall", "accuracy"]
        cv_results = {}
        
        for m in metric_names:
            values = [getattr(fm, m) for fm in fold_metrics]
            cv_results[m] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
        
        logger.info(f"CV Results: AUROC={cv_results['auroc']['mean']:.4f}±{cv_results['auroc']['std']:.4f}")
        
        return cv_results
    
    def compare_methods(
        self,
        methods: Dict[str, Any],  # Dict[str, BaseMethod]
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
    ) -> Dict[str, EvaluationResult]:
        """Compare multiple methods.
        
        Args:
            methods: Dictionary of method_name -> method instance
            features_list: Test features
            labels: Ground truth labels
            
        Returns:
            Dictionary of method_name -> EvaluationResult
        """
        results = {}
        
        for name, method in methods.items():
            result = self.evaluate_method(method, features_list, labels, name)
            results[name] = result
        
        # Log comparison
        logger.info("\n" + "=" * 60)
        logger.info("Method Comparison:")
        logger.info("=" * 60)
        
        for name, result in sorted(results.items(), key=lambda x: -x[1].metrics.auroc):
            logger.info(f"{name:20s} | AUROC={result.metrics.auroc:.4f} | "
                       f"F1={result.metrics.f1:.4f} | "
                       f"Acc={result.metrics.accuracy:.4f}")
        
        return results
    
    def generate_report(
        self,
        results: Dict[str, EvaluationResult],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate evaluation report.
        
        Args:
            results: Evaluation results
            output_path: Optional path to save report
            
        Returns:
            Report string
        """
        lines = [
            "# Hallucination Detection Evaluation Report",
            "",
            "## Summary",
            "",
            "| Method | AUROC | AUPRC | F1 | Precision | Recall | Accuracy |",
            "|--------|-------|-------|----|-----------| -------|----------|",
        ]
        
        for name, result in sorted(results.items(), key=lambda x: -x[1].metrics.auroc):
            m = result.metrics
            lines.append(
                f"| {name} | {m.auroc:.4f} | {m.auprc:.4f} | {m.f1:.4f} | "
                f"{m.precision:.4f} | {m.recall:.4f} | {m.accuracy:.4f} |"
            )
        
        lines.extend([
            "",
            "## Detailed Results",
            "",
        ])
        
        for name, result in results.items():
            m = result.metrics
            lines.extend([
                f"### {name}",
                "",
                f"- Optimal Threshold: {result.optimal_threshold:.3f}",
                f"- True Positives: {m.tp}",
                f"- False Positives: {m.fp}",
                f"- True Negatives: {m.tn}",
                f"- False Negatives: {m.fn}",
                f"- ECE (Calibration): {m.ece:.4f}" if m.ece else "",
                "",
            ])
        
        report = "\n".join(lines)
        
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {path}")
        
        return report


def create_evaluator(
    llm_config: Optional[LLMAPIConfig] = None,
    judge_mode: str = "binary",
) -> Evaluator:
    """Create evaluator instance.
    
    Args:
        llm_config: LLM API config for judge
        judge_mode: Judge mode (binary, score, detailed)
        
    Returns:
        Evaluator instance
    """
    return Evaluator(llm_config, judge_mode)
