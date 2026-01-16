"""Evaluation module for hallucination detection.

Provides:
- Metrics: AUROC, F1, Precision, Recall, Calibration
- LLM-as-Judge: Qwen/OpenAI API support
- Unified Evaluator: Method evaluation, cross-validation, comparison

Example:
    from src.evaluation import Evaluator, compute_metrics, LLMJudge
    from src.core import LLMAPIConfig
    
    # Simple metrics computation
    metrics = compute_metrics(predictions, labels)
    print(f"AUROC: {metrics.auroc:.4f}")
    
    # With LLM Judge
    evaluator = Evaluator(
        llm_config=LLMAPIConfig(
            provider="qwen",
            model="qwen-turbo",
            api_key="your-key",
        )
    )
    
    # Evaluate method
    result = evaluator.evaluate_method(method, features_list)
    
    # Compare methods
    results = evaluator.compare_methods({
        "lapeigvals": method1,
        "entropy": method2,
    }, features_list)
    
    # Generate report
    report = evaluator.generate_report(results, "report.md")
"""

from .metrics import (
    DetailedMetrics,
    compute_metrics,
    compute_ece,
    find_optimal_threshold,
    compute_roc_curve,
    compute_pr_curve,
    compare_methods,
    format_metrics_table,
    MetricsTracker,
)

from .llm_judge import (
    JudgeMode,
    BaseLLMClient,
    QwenClient,
    OpenAICompatibleClient,
    create_llm_client,
    LLMJudge,
    create_judge,
    evaluate_with_judge,
    BINARY_PROMPT,
    SCORE_PROMPT,
    DETAILED_PROMPT,
)

from .evaluator import (
    EvaluationResult,
    Evaluator,
    create_evaluator,
)

__all__ = [
    # Metrics
    "DetailedMetrics",
    "compute_metrics",
    "compute_ece",
    "find_optimal_threshold",
    "compute_roc_curve",
    "compute_pr_curve",
    "compare_methods",
    "format_metrics_table",
    "MetricsTracker",
    
    # LLM Judge
    "JudgeMode",
    "BaseLLMClient",
    "QwenClient",
    "OpenAICompatibleClient",
    "create_llm_client",
    "LLMJudge",
    "create_judge",
    "evaluate_with_judge",
    "BINARY_PROMPT",
    "SCORE_PROMPT",
    "DETAILED_PROMPT",
    
    # Evaluator
    "EvaluationResult",
    "Evaluator",
    "create_evaluator",
]
