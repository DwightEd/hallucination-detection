#!/usr/bin/env python3
"""Train a detection method on extracted features.

Usage:
    python scripts/train_method.py \
        --features ./outputs/features.pkl \
        --method lapeigvals \
        --output ./outputs/model.pkl
"""
import argparse
import sys
import pickle
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import MethodConfig, setup_logging, set_seed
from src.methods import create_method

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train detection method")
    
    parser.add_argument("--features", type=str, required=True,
                       help="Path to extracted features")
    parser.add_argument("--method", type=str, required=True,
                       choices=["lapeigvals", "lookback_lens", "entropy", 
                               "perplexity", "ensemble", "auto_ensemble"],
                       help="Method to train")
    parser.add_argument("--classifier", type=str, default="logistic",
                       choices=["logistic", "random_forest", "mlp"])
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for trained model")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    setup_logging(level=logging.INFO)
    set_seed(args.seed)
    
    # Load features
    logger.info(f"Loading features from {args.features}")
    with open(args.features, "rb") as f:
        data = pickle.load(f)
    
    features = data["features"]
    samples = data.get("samples", [])
    
    # Get labels
    labels = [f.label for f in features]
    
    logger.info(f"Loaded {len(features)} features")
    logger.info(f"Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
    
    # Create and train method
    config = MethodConfig(
        name=args.method,
        classifier=args.classifier,
        cv_folds=args.cv_folds,
        random_seed=args.seed,
    )
    
    method = create_method(args.method, config=config)
    
    logger.info(f"Training {args.method}...")
    metrics = method.fit(features, labels, cv=True)
    
    logger.info(f"Training metrics: {metrics}")
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    method.save(output_path)
    
    # Save metrics
    metrics_path = output_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Model saved to {output_path}")
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
