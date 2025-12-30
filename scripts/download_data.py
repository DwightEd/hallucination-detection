#!/usr/bin/env python3
"""Download datasets for hallucination detection.

Usage:
    python scripts/download_data.py --dataset truthfulqa --output ./data
    python scripts/download_data.py --all --output ./data
"""
import argparse
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import setup_logging

logger = logging.getLogger(__name__)


def download_truthfulqa(output_dir: Path):
    """Download TruthfulQA from HuggingFace."""
    try:
        from datasets import load_dataset
        
        logger.info("Downloading TruthfulQA from HuggingFace...")
        dataset = load_dataset("truthful_qa", "generation")
        
        output_path = output_dir / "TruthfulQA"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        import json
        for split in dataset:
            split_path = output_path / f"{split}.json"
            data = [dict(item) for item in dataset[split]]
            with open(split_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {split} split to {split_path}")
        
        logger.info(f"TruthfulQA downloaded to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download TruthfulQA: {e}")
        return False


def download_halueval(output_dir: Path):
    """Download HaluEval dataset."""
    try:
        from datasets import load_dataset
        
        logger.info("Downloading HaluEval from HuggingFace...")
        
        output_path = output_dir / "HaluEval"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download each subtask
        import json
        for subtask in ["qa", "summarization", "dialogue"]:
            try:
                dataset = load_dataset("pminervini/HaluEval", subtask)
                
                for split in dataset:
                    split_path = output_path / f"{subtask}_{split}.json"
                    data = [dict(item) for item in dataset[split]]
                    with open(split_path, "w") as f:
                        json.dump(data, f, indent=2)
                    logger.info(f"Saved {subtask}/{split} to {split_path}")
                    
            except Exception as e:
                logger.warning(f"Failed to download {subtask}: {e}")
        
        logger.info(f"HaluEval downloaded to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download HaluEval: {e}")
        return False


def download_ragtruth(output_dir: Path):
    """Instructions for RAGTruth (manual download required)."""
    logger.info("=" * 60)
    logger.info("RAGTruth Dataset")
    logger.info("=" * 60)
    logger.info("RAGTruth requires manual download:")
    logger.info("")
    logger.info("1. Visit: https://github.com/ParticleMedia/RAGTruth")
    logger.info("2. Clone the repository")
    logger.info("3. Copy the data files to: " + str(output_dir / "RAGTruth"))
    logger.info("")
    logger.info("Expected structure:")
    logger.info("  RAGTruth/")
    logger.info("    response.jsonl")
    logger.info("    source_info.jsonl")
    logger.info("=" * 60)
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Download datasets")
    
    parser.add_argument("--dataset", type=str, 
                       choices=["truthfulqa", "halueval", "ragtruth"],
                       help="Dataset to download")
    parser.add_argument("--all", action="store_true",
                       help="Download all available datasets")
    parser.add_argument("--output", type=str, default="./data",
                       help="Output directory")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    setup_logging(level=logging.INFO)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        download_truthfulqa(output_dir)
        download_halueval(output_dir)
        download_ragtruth(output_dir)
    elif args.dataset == "truthfulqa":
        download_truthfulqa(output_dir)
    elif args.dataset == "halueval":
        download_halueval(output_dir)
    elif args.dataset == "ragtruth":
        download_ragtruth(output_dir)
    else:
        logger.error("Please specify --dataset or --all")


if __name__ == "__main__":
    main()
