#!/usr/bin/env python3
"""Run LLM-as-Judge evaluation on samples with hallucination span detection.

All parameters are configured via Hydra configuration files.

This script:
1. Evaluates samples using LLM judge
2. Detects hallucination spans with character positions (RAGTruth format)
3. Outputs structured JSON with span annotations

Usage:
    # Default config (detailed mode with span detection)
    python scripts/llm_judge.py
    
    # Override LLM API
    python scripts/llm_judge.py llm_api=openai
    
    # Override model
    python scripts/llm_judge.py llm_api.model=gpt-4o
    
    # Specify judge mode
    python scripts/llm_judge.py judge_mode=detailed
"""
import sys
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, List

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    DatasetConfig, LLMAPIConfig, Sample, JudgeResult, HallucinationSpan,
    set_seed, setup_logging,
)
from src.data import get_dataset
from src.evaluation import create_judge, compute_metrics
from src.features import needs_llm_judge_for_spans

logger = logging.getLogger(__name__)


def find_output_dir(cfg: DictConfig) -> Path:
    """Find/create output directory for judge results."""
    base_dir = Path(cfg.results_dir) / "llm_judge"
    dataset_name = cfg.dataset.name
    llm_name = cfg.llm_api.model.replace("/", "_")
    
    output_dir = base_dir / dataset_name / llm_name / f"seed_{cfg.seed}"
    return output_dir


def get_api_key(cfg: DictConfig) -> str:
    """Get API key from environment."""
    env_var = cfg.llm_api.api_key_env
    api_key = os.environ.get(env_var, "")
    
    if not api_key:
        raise ValueError(
            f"API key not found. Please set {env_var} environment variable."
        )
    
    return api_key


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for LLM judge evaluation with hallucination span detection."""
    
    # Setup
    setup_logging(level=logging.INFO)
    set_seed(cfg.seed)
    
    logger.info("=" * 60)
    logger.info("LLM-as-Judge Evaluation (with Hallucination Span Detection)")
    logger.info("=" * 60)
    
    # Get API key
    try:
        api_key = get_api_key(cfg)
    except ValueError as e:
        logger.error(str(e))
        return
    
    # Build configs
    from src.core.config import get_dataset_config
    dataset_config = get_dataset_config(OmegaConf.to_container(cfg.dataset, resolve=True))
    llm_config_dict = OmegaConf.to_container(cfg.llm_api, resolve=True)
    # Set API key from environment
    llm_config_dict["api_key"] = api_key
    llm_config = LLMAPIConfig(**llm_config_dict)
    
    # Get judge mode from config, default to 'detailed' for span detection
    judge_mode = cfg.get("judge_mode", "detailed")
    
    logger.info(f"Dataset: {dataset_config.name}")
    logger.info(f"LLM: {llm_config.provider}/{llm_config.model}")
    logger.info(f"Judge Mode: {judge_mode}")
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_config.name}")
    dataset = get_dataset(config=dataset_config, split=dataset_config.test_split_name)
    samples = dataset.load(max_samples=dataset_config.max_samples)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Filter samples that need LLM judge for span annotation
    # (samples labeled as hallucinated but missing span annotations)
    samples_needing_spans = []
    samples_with_spans = []
    for sample in samples:
        if sample.label == 1 and needs_llm_judge_for_spans(sample.metadata):
            samples_needing_spans.append(sample)
        else:
            samples_with_spans.append(sample)
    
    logger.info(f"Samples needing span annotation: {len(samples_needing_spans)}")
    logger.info(f"Samples with existing spans: {len(samples_with_spans)}")
    
    # Create judge with detailed mode for span detection
    logger.info("Creating LLM judge...")
    judge = create_judge(llm_config, mode=judge_mode)
    
    # Run evaluation on all samples
    logger.info("Running LLM judge evaluation...")
    results: List[JudgeResult] = []
    
    for i, sample in enumerate(samples):
        try:
            result = judge.judge(sample)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(samples)} samples")
                
        except Exception as e:
            logger.warning(f"Failed to judge sample {sample.id}: {e}")
            # Create a failed result
            results.append(JudgeResult(
                sample_id=sample.id,
                label=-1,  # Unknown
                confidence=0.0,
                explanation=f"Error: {str(e)}",
                raw_response="",
                model=llm_config.model,
                hallucination_spans=[],
            ))
    
    logger.info(f"Judged {len(results)} samples")
    
    # Statistics
    n_hallucinated = sum(1 for r in results if r.label == 1)
    n_clean = sum(1 for r in results if r.label == 0)
    n_failed = sum(1 for r in results if r.label == -1)
    n_with_spans = sum(1 for r in results if r.hallucination_spans)
    
    logger.info(f"Results: {n_hallucinated} hallucinated, {n_clean} clean, {n_failed} failed")
    logger.info(f"Results with span annotations: {n_with_spans}")
    
    # Compare with ground truth if available
    has_labels = any(s.label is not None for s in samples)
    metrics = None
    
    if has_labels:
        true_labels = [s.label for s in samples]
        pred_labels = [r.label for r in results]
        
        # Filter out failed predictions
        valid_idx = [i for i, r in enumerate(results) if r.label != -1]
        valid_true = [true_labels[i] for i in valid_idx]
        valid_pred = [pred_labels[i] for i in valid_idx]
        
        if valid_true and valid_pred:
            # Use confidence as score
            valid_scores = [results[i].confidence for i in valid_idx]
            metrics = compute_metrics(valid_scores, valid_true)
            
            logger.info("=" * 40)
            logger.info("Comparison with Ground Truth:")
            logger.info("=" * 40)
            logger.info(f"  AUROC:     {metrics.auroc:.4f}")
            logger.info(f"  AUPRC:     {metrics.auprc:.4f}")
            logger.info(f"  F1:        {metrics.f1:.4f}")
            logger.info(f"  Accuracy:  {metrics.accuracy:.4f}")
            logger.info("=" * 40)
    
    # Save results
    output_dir = find_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save judge results with hallucination spans
    results_data = [r.to_dict() for r in results]
    results_path = output_dir / "judge_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Save samples with span annotations in RAGTruth format
    # This creates a unified data structure that can be used by detection methods
    samples_with_annotations = []
    for sample, result in zip(samples, results):
        sample_data = sample.to_dict()
        # Add judge annotations
        sample_data["judge_label"] = result.label
        sample_data["judge_confidence"] = result.confidence
        sample_data["judge_explanation"] = result.explanation
        # Add hallucination spans from judge (RAGTruth format)
        if result.hallucination_spans:
            sample_data["labels"] = [
                {
                    "start": span.start,
                    "end": span.end,
                    "text": span.text,
                    "label_type": span.label_type,
                    "explanation": span.explanation,
                }
                for span in result.hallucination_spans
            ]
        else:
            # Keep existing spans from metadata if available
            existing_spans = sample.metadata.get("hallucination_spans", [])
            sample_data["labels"] = existing_spans
        samples_with_annotations.append(sample_data)
    
    annotated_samples_path = output_dir / "samples_annotated.json"
    with open(annotated_samples_path, "w", encoding="utf-8") as f:
        json.dump({
            "extraction_info": {
                "judge_model": llm_config.model,
                "judge_mode": judge_mode,
                "dataset": dataset_config.name,
            },
            "samples": samples_with_annotations,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Annotated samples saved to {annotated_samples_path}")
    
    # Save metrics if available
    if metrics:
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
    
    # Save config
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()