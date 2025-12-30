#!/usr/bin/env python3
"""Standalone feature extraction script.

Usage:
    python scripts/extract_features.py \
        --dataset ragtruth \
        --dataset-path ./data/RAGTruth \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output ./outputs/features.pkl \
        --layers last_n:4 \
        --max-samples 1000
"""
import argparse
import sys
import pickle
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    DatasetConfig, ModelConfig, FeaturesConfig,
    setup_logging, set_seed,
)
from src.data import create_dataset
from src.models import load_model, unload_all_models
from src.features import FeatureExtractor

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract features for hallucination detection")
    
    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["ragtruth", "truthfulqa", "halueval"],
                       help="Dataset name")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to dataset")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to process")
    
    # Model
    parser.add_argument("--model", type=str, required=True,
                       help="Model name or path")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="Use 4-bit quantization")
    
    # Features
    parser.add_argument("--layers", type=str, default="last_n:4",
                       help="Layer selection (e.g., 'last_n:4', 'all', '[0,1,2]')")
    parser.add_argument("--mode", type=str, default="teacher_forcing",
                       choices=["teacher_forcing", "generation"])
    parser.add_argument("--max-length", type=int, default=2048)
    
    # Output
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for features")
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    setup_logging(level=logging.INFO)
    set_seed(args.seed)
    
    logger.info(f"Extracting features from {args.dataset}")
    
    # Build configs
    dataset_config = DatasetConfig(
        name=args.dataset,
        path=args.dataset_path,
        max_samples=args.max_samples,
    )
    
    model_config = ModelConfig(
        name=args.model,
        attn_implementation="eager",
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )
    
    features_config = FeaturesConfig(
        mode=args.mode,
        attention_layers=args.layers,
        hidden_states_layers=args.layers,
        max_length=args.max_length,
    )
    
    # Load dataset
    dataset = create_dataset(args.dataset, config=dataset_config)
    samples = dataset.load(max_samples=args.max_samples)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model = load_model(model_config)
    
    # Extract features
    logger.info("Extracting features...")
    extractor = FeatureExtractor(model, features_config)
    features = extractor.extract_batch(samples)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump({
            "features": features,
            "samples": samples,
            "config": {
                "dataset": args.dataset,
                "model": args.model,
                "layers": args.layers,
            }
        }, f)
    
    logger.info(f"Features saved to {output_path}")
    
    # Cleanup
    unload_all_models()
    logger.info("Done!")


if __name__ == "__main__":
    main()
