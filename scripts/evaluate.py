#!/usr/bin/env python3
"""Evaluate a trained method on test data.

Usage:
    python scripts/evaluate.py \
        --model ./outputs/model.pkl \
        --features ./outputs/test_features.pkl \
        --output ./outputs/evaluation.json
"""
import argparse
import sys
import pickle
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import setup_logging
from src.methods import BaseMethod
from src.evaluation import compute_metrics, find_optimal_threshold, EvaluationResult

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate detection method")
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--features", type=str, required=True,
                       help="Path to test features")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for evaluation results")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Classification threshold (auto if not specified)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    setup_logging(level=logging.INFO)
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    with open(args.model, "rb") as f:
        state = pickle.load(f)
    
    # Reconstruct method
    from src.methods.base import BaseMethod
    method = BaseMethod.__new__(BaseMethod)
    method.__dict__.update(state)
    
    # Load features
    logger.info(f"Loading features from {args.features}")
    with open(args.features, "rb") as f:
        data = pickle.load(f)
    
    features = data["features"]
    labels = [f.label for f in features]
    
    logger.info(f"Evaluating on {len(features)} samples")
    
    # Predict
    predictions = method.predict_batch(features)
    
    # Compute metrics
    if args.threshold:
        threshold = args.threshold
    else:
        threshold, _ = find_optimal_threshold(predictions, labels)
        logger.info(f"Optimal threshold: {threshold:.3f}")
    
    metrics = compute_metrics(predictions, labels, threshold=threshold)
    
    logger.info(f"Evaluation Results:")
    logger.info(f"  AUROC:     {metrics.auroc:.4f}")
    logger.info(f"  AUPRC:     {metrics.auprc:.4f}")
    logger.info(f"  F1:        {metrics.f1:.4f}")
    logger.info(f"  Precision: {metrics.precision:.4f}")
    logger.info(f"  Recall:    {metrics.recall:.4f}")
    logger.info(f"  Accuracy:  {metrics.accuracy:.4f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "metrics": metrics.to_dict(),
        "threshold": threshold,
        "predictions": [
            {"sample_id": p.sample_id, "score": p.score, "label": p.label}
            for p in predictions
        ],
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
