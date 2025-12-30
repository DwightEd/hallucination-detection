#!/usr/bin/env python3
"""Main experiment runner with Hydra configuration.

Usage:
    # Default experiment
    python scripts/run_experiment.py
    
    # Override config
    python scripts/run_experiment.py experiment=full_benchmark
    
    # Override specific values
    python scripts/run_experiment.py dataset=truthfulqa method=ensemble
    
    # Multiple runs
    python scripts/run_experiment.py --multirun method=lapeigvals,entropy,ensemble
"""
import sys
import logging
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    DatasetConfig, ModelConfig, FeaturesConfig, MethodConfig, LLMAPIConfig,
    setup_logging, set_seed, ensure_dir,
)
from src.data import create_dataset
from src.models import load_model, unload_all_models
from src.features import FeatureExtractor
from src.methods import create_method
from src.evaluation import Evaluator, create_evaluator

logger = logging.getLogger(__name__)


def build_configs(cfg: DictConfig):
    """Build typed configs from Hydra DictConfig."""
    
    dataset_config = DatasetConfig(
        name=cfg.dataset.name,
        path=cfg.dataset.path,
        splits=cfg.dataset.get("splits"),
        task_types=cfg.dataset.get("task_types"),
        max_samples=cfg.dataset.get("max_samples"),
        random_seed=cfg.get("seed", 42),
    )
    
    model_config = ModelConfig(
        name=cfg.model.name,
        attn_implementation=cfg.model.get("attn_implementation", "eager"),
        dtype=cfg.model.get("dtype", "bfloat16"),
        load_in_4bit=cfg.model.get("load_in_4bit", False),
        load_in_8bit=cfg.model.get("load_in_8bit", False),
        device_map=cfg.model.get("device_map", "auto"),
        n_layers=cfg.model.get("n_layers", 32),
        n_heads=cfg.model.get("n_heads", 32),
        hidden_size=cfg.model.get("hidden_size", 4096),
        trust_remote_code=cfg.model.get("trust_remote_code", True),
    )
    
    features_config = FeaturesConfig(
        mode=cfg.features.get("mode", "teacher_forcing"),
        attention_layers=cfg.features.get("attention_layers", "last_n:4"),
        hidden_states_layers=cfg.features.get("hidden_states_layers", "last_n:4"),
        attention_enabled=cfg.features.get("attention_enabled", True),
        hidden_states_enabled=cfg.features.get("hidden_states_enabled", True),
        token_probs_enabled=cfg.features.get("token_probs_enabled", True),
        token_probs_top_k=cfg.features.get("token_probs_top_k", 10),
        max_length=cfg.features.get("max_length", 2048),
        hidden_states_pooling=cfg.features.get("hidden_states_pooling", "last_token"),
    )
    
    method_config = MethodConfig(
        name=cfg.method.name,
        classifier=cfg.method.get("classifier", "logistic"),
        params=OmegaConf.to_container(cfg.method.get("params", {})),
        cv_folds=cfg.method.get("cv_folds", 5),
        random_seed=cfg.get("seed", 42),
    )
    
    llm_config = None
    if cfg.get("llm_api") and cfg.evaluation.get("use_llm_judge", False):
        llm_config = LLMAPIConfig(
            provider=cfg.llm_api.provider,
            model=cfg.llm_api.model,
            api_key=cfg.llm_api.get("api_key", ""),
            base_url=cfg.llm_api.get("base_url"),
            temperature=cfg.llm_api.get("temperature", 0.1),
            max_tokens=cfg.llm_api.get("max_tokens", 1024),
            rate_limit=cfg.llm_api.get("rate_limit", 60),
            system_prompt=cfg.llm_api.get("system_prompt"),
        )
    
    return dataset_config, model_config, features_config, method_config, llm_config


@hydra.main(version_base=None, config_path="../config/experiment", config_name="default")
def main(cfg: DictConfig) -> Optional[float]:
    """Main experiment entry point."""
    
    # Setup
    setup_logging(level=logging.INFO)
    set_seed(cfg.get("seed", 42))
    
    logger.info("=" * 60)
    logger.info(f"Experiment: {cfg.name}")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Build configs
    dataset_cfg, model_cfg, features_cfg, method_cfg, llm_cfg = build_configs(cfg)
    
    # Output directory
    output_dir = Path(cfg.output.dir)
    ensure_dir(output_dir)
    
    # Save config
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    logger.info(f"Config saved to {config_path}")
    
    try:
        # 1. Load dataset
        logger.info(f"Loading dataset: {dataset_cfg.name}")
        dataset = create_dataset(dataset_cfg.name, config=dataset_cfg)
        samples = dataset.load(max_samples=dataset_cfg.max_samples)
        logger.info(f"Loaded {len(samples)} samples")
        
        # Split train/test
        from sklearn.model_selection import train_test_split
        test_split = cfg.training.get("test_split", 0.2)
        labels = [s.label for s in samples]
        
        train_samples, test_samples = train_test_split(
            samples, 
            test_size=test_split,
            stratify=labels,
            random_state=cfg.seed,
        )
        logger.info(f"Train: {len(train_samples)}, Test: {len(test_samples)}")
        
        # 2. Load model
        logger.info(f"Loading model: {model_cfg.name}")
        model = load_model(model_cfg)
        
        # 3. Extract features
        logger.info("Extracting features...")
        extractor = FeatureExtractor(model, features_cfg)
        
        train_features = extractor.extract_batch(train_samples)
        test_features = extractor.extract_batch(test_samples)
        
        logger.info(f"Extracted: {len(train_features)} train, {len(test_features)} test features")
        
        # Save features if requested
        if cfg.output.get("save_features", False):
            import pickle
            features_path = output_dir / "features.pkl"
            with open(features_path, "wb") as f:
                pickle.dump({"train": train_features, "test": test_features}, f)
            logger.info(f"Features saved to {features_path}")
        
        # Unload model to free memory
        unload_all_models()
        
        # 4. Train method
        logger.info(f"Training method: {method_cfg.name}")
        method = create_method(method_cfg.name, config=method_cfg)
        
        train_labels = [s.label for s in train_samples]
        train_metrics = method.fit(train_features, train_labels, cv=True)
        logger.info(f"Training metrics: {train_metrics}")
        
        # Save model if requested
        if cfg.output.get("save_model", False):
            model_path = output_dir / f"{method_cfg.name}_model.pkl"
            method.save(model_path)
            logger.info(f"Model saved to {model_path}")
        
        # 5. Evaluate
        logger.info("Evaluating...")
        evaluator = create_evaluator(
            llm_config=llm_cfg,
            judge_mode=cfg.evaluation.get("judge_mode", "binary"),
        )
        
        test_labels = [s.label for s in test_samples]
        result = evaluator.evaluate_method(
            method, 
            test_features, 
            test_labels,
            method_name=method_cfg.name,
        )
        
        logger.info(f"Test metrics: {result.metrics}")
        
        # Save predictions
        if cfg.output.get("save_predictions", False):
            import json
            predictions_path = output_dir / "predictions.json"
            preds_data = [
                {"sample_id": p.sample_id, "score": p.score, "label": p.label}
                for p in result.predictions
            ]
            with open(predictions_path, "w") as f:
                json.dump(preds_data, f, indent=2)
            logger.info(f"Predictions saved to {predictions_path}")
        
        # LLM Judge evaluation
        if llm_cfg and cfg.evaluation.get("use_llm_judge", False):
            logger.info("Running LLM Judge evaluation...")
            judge_results = evaluator.evaluate_with_judge(
                test_samples,
                result.predictions,
            )
            logger.info(f"Judge agreement: {judge_results.get('agreement_with_method', 0):.4f}")
        
        # Generate report
        report = evaluator.generate_report(
            {method_cfg.name: result},
            output_dir / "report.md",
        )
        
        # Save final metrics
        result.save(output_dir / "evaluation_result.json")
        
        logger.info("=" * 60)
        logger.info("Experiment completed!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 60)
        
        # Return AUROC for hyperparameter optimization
        return result.metrics.auroc
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
