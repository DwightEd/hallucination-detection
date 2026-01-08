#!/usr/bin/env python3
"""extract_base_features.py - Stage 1: 基础特征提取脚本。

提取模型推理所需的基础特征：
- full_attention: 完整注意力矩阵
- hidden_states: 隐藏状态
- token_probs: Token 概率

Usage:
    # 提取所有基础特征
    python scripts/features/extract_base_features.py \
        dataset.name=ragtruth \
        dataset.task_type=QA \
        model=mistral_7b \
        seed=42
    
    # 只提取 hidden_states（用于 haloscope）
    python scripts/features/extract_base_features.py \
        dataset.name=ragtruth \
        model=mistral_7b \
        features.full_attention=false \
        features.hidden_states=true \
        features.token_probs=false
    
    # 指定方法，自动确定需要的特征
    python scripts/features/extract_base_features.py \
        methods=[lapeigvals,lookback_lens] \
        dataset.name=ragtruth \
        model=mistral_7b
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm
import json

from src.models import load_model
from src.data import load_dataset
from src.features.feature_registry import (
    compute_union_requirements,
    get_base_features_needed,
    should_store_full_attention,
    describe_requirements,
    BaseFeatureType,
)
from src.features.base.attention import AttentionExtractor, AttentionExtractionConfig
from src.features.base.hidden_states import HiddenStatesExtractor, HiddenStatesExtractionConfig
from src.features.base.token_probs import TokenProbsExtractor, TokenProbsExtractionConfig

logger = logging.getLogger(__name__)


def build_output_dir(cfg: DictConfig) -> Path:
    """构建输出目录路径。"""
    base_dir = Path(cfg.get("output_dir", "outputs/features"))
    
    dataset_name = cfg.dataset.name
    model_name = cfg.model.short_name if hasattr(cfg.model, "short_name") else cfg.model.name
    seed = cfg.get("seed", 42)
    task_type = cfg.dataset.get("task_type", "default")
    
    output_dir = base_dir / dataset_name / model_name / f"seed_{seed}" / task_type / "base"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def determine_features_needed(cfg: DictConfig) -> dict:
    """确定需要提取的特征。"""
    # 如果指定了 methods，自动确定
    if "methods" in cfg and cfg.methods:
        methods = list(cfg.methods) if hasattr(cfg.methods, "__iter__") else [cfg.methods]
        requirements = compute_union_requirements(methods)
        base_needed = get_base_features_needed(requirements)
        
        return {
            "full_attention": BaseFeatureType.FULL_ATTENTION in base_needed,
            "hidden_states": BaseFeatureType.HIDDEN_STATES in base_needed,
            "token_probs": BaseFeatureType.TOKEN_PROBS in base_needed,
            "store_full_attention": should_store_full_attention(requirements),
        }
    
    # 否则使用配置
    features_cfg = cfg.get("features", {})
    return {
        "full_attention": features_cfg.get("full_attention", True),
        "hidden_states": features_cfg.get("hidden_states", True),
        "token_probs": features_cfg.get("token_probs", True),
        "store_full_attention": features_cfg.get("store_full_attention", False),
    }


def extract_sample_features(
    model,
    sample,
    features_needed: dict,
    device: str,
) -> dict:
    """提取单个样本的基础特征。"""
    # 分词
    prompt_ids = model.encode(sample.prompt, add_special_tokens=True)
    response_ids = model.encode(sample.response, add_special_tokens=False)
    
    prompt_len = prompt_ids.size(1)
    response_len = response_ids.size(1)
    
    # 拼接
    input_ids = torch.cat([prompt_ids, response_ids], dim=1).to(device)
    seq_len = input_ids.size(1)
    
    # 前向传播
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            output_attentions=features_needed["full_attention"],
            output_hidden_states=features_needed["hidden_states"],
            return_dict=True,
        )
    
    results = {
        "sample_id": sample.id,
        "prompt_len": prompt_len,
        "response_len": response_len,
        "seq_len": seq_len,
        "label": sample.label,
    }
    
    # 提取 full_attention
    if features_needed["full_attention"] and outputs.attentions is not None:
        attn_config = AttentionExtractionConfig(
            store_on_cpu=True,
            half_precision=True,
        )
        extractor = AttentionExtractor(attn_config)
        attn_result = extractor.extract(outputs.attentions, prompt_len, response_len)
        results["full_attention"] = attn_result["full_attention"]
        results["n_layers"] = attn_result["full_attention"].shape[0]
        results["n_heads"] = attn_result["n_heads"]
        
        del outputs.attentions
    
    # 提取 hidden_states
    if features_needed["hidden_states"] and outputs.hidden_states is not None:
        hs_config = HiddenStatesExtractionConfig(
            store_on_cpu=True,
            half_precision=True,
        )
        extractor = HiddenStatesExtractor(hs_config)
        hs_result = extractor.extract(outputs.hidden_states, prompt_len, response_len)
        results["hidden_states"] = hs_result["hidden_states"]
        
        del outputs.hidden_states
    
    # 提取 token_probs
    if features_needed["token_probs"] and hasattr(outputs, 'logits') and outputs.logits is not None:
        probs_config = TokenProbsExtractionConfig(
            compute_entropy=True,
            compute_perplexity=True,
            response_only=True,
        )
        extractor = TokenProbsExtractor(probs_config)
        probs_result = extractor.extract(outputs.logits, input_ids, prompt_len, response_len)
        results["token_probs"] = probs_result.get("token_probs")
        results["token_entropy"] = probs_result.get("entropy")
        results["perplexity"] = probs_result.get("perplexity")
    
    return results


def save_features(output_dir: Path, features: dict, features_needed: dict):
    """保存特征到对应目录。"""
    sample_id = features["sample_id"]
    metadata = {
        "sample_id": sample_id,
        "prompt_len": features["prompt_len"],
        "response_len": features["response_len"],
        "seq_len": features["seq_len"],
        "label": features["label"],
    }
    
    # 保存 full_attention
    if features_needed["full_attention"] and "full_attention" in features:
        attn_dir = output_dir / "full_attention"
        attn_dir.mkdir(exist_ok=True)
        torch.save({
            **metadata,
            "full_attention": features["full_attention"],
            "n_layers": features.get("n_layers"),
            "n_heads": features.get("n_heads"),
        }, attn_dir / f"{sample_id}.pt")
    
    # 保存 hidden_states
    if features_needed["hidden_states"] and "hidden_states" in features:
        hs_dir = output_dir / "hidden_states"
        hs_dir.mkdir(exist_ok=True)
        torch.save({
            **metadata,
            "hidden_states": features["hidden_states"],
        }, hs_dir / f"{sample_id}.pt")
    
    # 保存 token_probs
    if features_needed["token_probs"] and "token_probs" in features:
        probs_dir = output_dir / "token_probs"
        probs_dir.mkdir(exist_ok=True)
        torch.save({
            **metadata,
            "token_probs": features["token_probs"],
            "token_entropy": features.get("token_entropy"),
            "perplexity": features.get("perplexity"),
        }, probs_dir / f"{sample_id}.pt")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """主函数。"""
    logger.info("=" * 60)
    logger.info("Stage 1: Base Feature Extraction")
    logger.info("=" * 60)
    
    # 确定需要的特征
    features_needed = determine_features_needed(cfg)
    logger.info(f"Features to extract: {features_needed}")
    
    # 构建输出目录
    output_dir = build_output_dir(cfg)
    logger.info(f"Output directory: {output_dir}")
    
    # 保存配置
    with open(output_dir / "extraction_config.json", "w") as f:
        json.dump({
            "features_needed": features_needed,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, f, indent=2, default=str)
    
    # 加载模型
    logger.info(f"Loading model: {cfg.model.name}")
    model = load_model(cfg.model)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    logger.info(f"Loading dataset: {cfg.dataset.name}")
    samples = load_dataset(cfg.dataset)
    logger.info(f"Total samples: {len(samples)}")
    
    # 提取特征
    stats = {"success": 0, "failed": 0, "skipped": 0}
    clear_cache_every = cfg.get("clear_cache_every", 10)
    
    for idx, sample in enumerate(tqdm(samples, desc="Extracting base features")):
        try:
            # 检查是否已存在
            existing = True
            for feature_type, needed in features_needed.items():
                if feature_type == "store_full_attention":
                    continue
                if needed:
                    feature_path = output_dir / feature_type / f"{sample.id}.pt"
                    if not feature_path.exists():
                        existing = False
                        break
            
            if existing:
                stats["skipped"] += 1
                continue
            
            # 提取特征
            features = extract_sample_features(model, sample, features_needed, device)
            
            # 保存
            save_features(output_dir, features, features_needed)
            
            stats["success"] += 1
            
            # 内存清理
            if (idx + 1) % clear_cache_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logger.error(f"Failed to extract features for sample {sample.id}: {e}")
            stats["failed"] += 1
    
    # 保存统计
    logger.info(f"Extraction complete: {stats}")
    with open(output_dir / "extraction_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
