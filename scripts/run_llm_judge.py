#!/usr/bin/env python3
"""Run LLM-as-Judge evaluation on samples.

Usage:
    python scripts/run_llm_judge.py \
        --dataset ragtruth \
        --dataset-path ./data/RAGTruth \
        --provider qwen \
        --model qwen-turbo \
        --mode binary \
        --output ./outputs/judge_results.json \
        --max-samples 100
"""
import argparse
import sys
import json
import os
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import DatasetConfig, LLMAPIConfig, setup_logging
from src.data import create_dataset
from src.evaluation import create_judge, JudgeMode

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM-as-Judge evaluation")
    
    # Dataset
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    
    # LLM API
    parser.add_argument("--provider", type=str, default="qwen",
                       choices=["qwen", "openai", "openai_compatible"])
    parser.add_argument("--model", type=str, default="qwen-turbo")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key (or use DASHSCOPE_API_KEY/OPENAI_API_KEY env)")
    parser.add_argument("--base-url", type=str, default=None)
    
    # Judge settings
    parser.add_argument("--mode", type=str, default="binary",
                       choices=["binary", "score", "detailed"])
    parser.add_argument("--rate-limit", type=int, default=60,
                       help="Requests per minute")
    
    # Output
    parser.add_argument("--output", type=str, required=True)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    setup_logging(level=logging.INFO)
    
    # Get API key
    api_key = args.api_key
    if not api_key:
        if args.provider == "qwen":
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if not api_key:
        logger.error("API key not provided. Set via --api-key or environment variable.")
        return
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset_config = DatasetConfig(name=args.dataset, path=args.dataset_path)
    dataset = create_dataset(args.dataset, config=dataset_config)
    samples = dataset.load(max_samples=args.max_samples)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Create judge
    llm_config = LLMAPIConfig(
        provider=args.provider,
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        rate_limit=args.rate_limit,
    )
    
    judge = create_judge(llm_config, mode=args.mode)
    
    # Run evaluation
    logger.info(f"Running LLM Judge ({args.mode} mode)...")
    results = judge.judge_batch(samples)
    
    # Statistics
    n_hallucinated = sum(r.hallucinated for r in results)
    logger.info(f"Results: {n_hallucinated}/{len(results)} hallucinated")
    
    # Compare with ground truth
    if any(s.label is not None for s in samples):
        ground_truth = [s.label for s in samples]
        judge_labels = [1 if r.hallucinated else 0 for r in results]
        
        agreement = sum(g == j for g, j in zip(ground_truth, judge_labels)) / len(ground_truth)
        logger.info(f"Agreement with ground truth: {agreement:.4f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_data = {
        "config": {
            "dataset": args.dataset,
            "provider": args.provider,
            "model": args.model,
            "mode": args.mode,
        },
        "statistics": {
            "total": len(results),
            "hallucinated": n_hallucinated,
            "clean": len(results) - n_hallucinated,
        },
        "results": [
            {
                "sample_id": r.sample_id,
                "hallucinated": r.hallucinated,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
                "score": r.score,
            }
            for r in results
        ],
    }
    
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
