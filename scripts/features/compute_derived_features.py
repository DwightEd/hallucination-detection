#!/usr/bin/env python3
"""compute_derived_features.py - Stage 2: 派生特征计算脚本。

从基础特征计算各方法需要的派生特征：
- attention_diags: 注意力对角线
- laplacian_diags: Laplacian 对角线
- attention_entropy: 注意力熵
- lookback_ratio: Lookback 比率（需要 prompt 注意力）
- mva_features: Multi-View Attention 特征

Usage:
    # 计算单个方法的派生特征
    python scripts/features/compute_derived_features.py \
        method=lapeigvals \
        base_features_dir=outputs/features/ragtruth/mistral_7b/seed_42/QA/base
    
    # 计算多个方法的派生特征
    python scripts/features/compute_derived_features.py \
        methods=[lapeigvals,lookback_lens,haloscope] \
        base_features_dir=outputs/features/ragtruth/mistral_7b/seed_42/QA/base
    
    # 使用完整配置
    python scripts/features/compute_derived_features.py \
        dataset.name=ragtruth \
        dataset.task_type=QA \
        model=mistral_7b \
        seed=42 \
        methods=[lapeigvals,lookback_lens]
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

from src.features.feature_registry import (
    get_method_requirements,
    describe_requirements,
    DerivedFeatureType,
    FeatureScope,
)
from src.features.derived import (
    compute_attention_diags,
    compute_laplacian_diags,
    compute_attention_entropy,
    compute_lookback_ratio,
    compute_mva_features,
)

logger = logging.getLogger(__name__)


def find_base_features_dir(cfg: DictConfig) -> Path:
    """查找基础特征目录。"""
    # 如果直接指定
    if "base_features_dir" in cfg:
        return Path(cfg.base_features_dir)
    
    # 从配置构建
    base_dir = Path(cfg.get("output_dir", "outputs/features"))
    dataset_name = cfg.dataset.name
    model_name = cfg.model.short_name if hasattr(cfg.model, "short_name") else cfg.model.name
    seed = cfg.get("seed", 42)
    task_type = cfg.dataset.get("task_type", "default")
    
    return base_dir / dataset_name / model_name / f"seed_{seed}" / task_type / "base"


def build_derived_dir(base_dir: Path, method: str) -> Path:
    """构建派生特征输出目录。"""
    # base_dir 是 .../base/
    # 输出到 .../derived/{method}/
    derived_dir = base_dir.parent / "derived" / method
    derived_dir.mkdir(parents=True, exist_ok=True)
    return derived_dir


def get_sample_ids(base_dir: Path) -> list:
    """获取所有样本 ID。"""
    sample_ids = set()
    
    # 从任一基础特征目录获取
    for subdir in ["full_attention", "hidden_states", "token_probs"]:
        feature_dir = base_dir / subdir
        if feature_dir.exists():
            for path in feature_dir.glob("*.pt"):
                sample_ids.add(path.stem)
            break
    
    return sorted(list(sample_ids))


def load_base_features(base_dir: Path, sample_id: str) -> dict:
    """加载基础特征。"""
    features = {}
    
    # 加载 full_attention
    attn_path = base_dir / "full_attention" / f"{sample_id}.pt"
    if attn_path.exists():
        data = torch.load(attn_path, map_location="cpu")
        features.update(data)
    
    # 加载 hidden_states
    hs_path = base_dir / "hidden_states" / f"{sample_id}.pt"
    if hs_path.exists():
        data = torch.load(hs_path, map_location="cpu")
        features["hidden_states"] = data["hidden_states"]
    
    # 加载 token_probs
    probs_path = base_dir / "token_probs" / f"{sample_id}.pt"
    if probs_path.exists():
        data = torch.load(probs_path, map_location="cpu")
        features["token_probs"] = data.get("token_probs")
        features["token_entropy"] = data.get("token_entropy")
    
    return features


def compute_method_features(method: str, base_features: dict) -> dict:
    """为特定方法计算派生特征。"""
    req = get_method_requirements(method)
    
    results = {
        "method": method,
        "sample_id": base_features.get("sample_id"),
        "prompt_len": base_features.get("prompt_len", 0),
        "response_len": base_features.get("response_len", 0),
        "label": base_features.get("label"),
    }
    
    full_attn = base_features.get("full_attention")
    prompt_len = results["prompt_len"]
    response_len = results["response_len"]
    
    # 计算各派生特征
    for df in req.derived_features:
        try:
            if df.feature_type == DerivedFeatureType.ATTENTION_DIAGS:
                if full_attn is not None:
                    results["attention_diags"] = compute_attention_diags(
                        full_attn,
                        prompt_len=prompt_len,
                        response_len=response_len,
                        response_only=(df.scope == FeatureScope.RESPONSE_ONLY),
                    )
            
            elif df.feature_type == DerivedFeatureType.LAPLACIAN_DIAGS:
                if full_attn is not None:
                    results["laplacian_diags"] = compute_laplacian_diags(
                        full_attn,
                        prompt_len=prompt_len,
                        response_len=response_len,
                        response_only=(df.scope == FeatureScope.RESPONSE_ONLY),
                    )
            
            elif df.feature_type == DerivedFeatureType.ATTENTION_ENTROPY:
                if full_attn is not None:
                    results["attention_entropy"] = compute_attention_entropy(
                        full_attn,
                        prompt_len=prompt_len,
                        response_len=response_len,
                        response_only=(df.scope == FeatureScope.RESPONSE_ONLY),
                    )
            
            elif df.feature_type == DerivedFeatureType.LOOKBACK_RATIO:
                if full_attn is not None and prompt_len > 0:
                    results["lookback_ratio"] = compute_lookback_ratio(
                        full_attn,
                        prompt_len=prompt_len,
                        response_len=response_len,
                    )
            
            elif df.feature_type == DerivedFeatureType.MVA_FEATURES:
                if full_attn is not None:
                    mva = compute_mva_features(
                        full_attn,
                        prompt_len=prompt_len,
                        response_len=response_len,
                    )
                    results["mva_features"] = mva
                    
        except Exception as e:
            logger.warning(f"Failed to compute {df.feature_type.value}: {e}")
    
    # 添加 hidden_states（如果方法需要）
    if req.needs_hidden_states() and "hidden_states" in base_features:
        results["hidden_states"] = base_features["hidden_states"]
    
    # 添加 token_probs（如果方法需要）
    if req.needs_token_probs() and "token_probs" in base_features:
        results["token_probs"] = base_features["token_probs"]
        results["token_entropy"] = base_features.get("token_entropy")
    
    return results


def save_derived_features(output_dir: Path, sample_id: str, features: dict):
    """保存派生特征。"""
    output_path = output_dir / f"{sample_id}.pt"
    torch.save(features, output_path)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """主函数。"""
    logger.info("=" * 60)
    logger.info("Stage 2: Derived Feature Computation")
    logger.info("=" * 60)
    
    # 确定方法列表
    if "methods" in cfg and cfg.methods:
        methods = list(cfg.methods) if hasattr(cfg.methods, "__iter__") else [cfg.methods]
    elif "method" in cfg:
        methods = [cfg.method]
    else:
        logger.error("No methods specified. Use 'method=xxx' or 'methods=[xxx,yyy]'")
        return
    
    logger.info(f"Methods to compute: {methods}")
    
    # 打印方法需求
    for method in methods:
        req = get_method_requirements(method)
        logger.info(describe_requirements(req))
    
    # 查找基础特征目录
    base_dir = find_base_features_dir(cfg)
    if not base_dir.exists():
        logger.error(f"Base features directory not found: {base_dir}")
        return
    
    logger.info(f"Base features directory: {base_dir}")
    
    # 获取样本 ID
    sample_ids = get_sample_ids(base_dir)
    logger.info(f"Total samples: {len(sample_ids)}")
    
    # 为每个方法计算派生特征
    for method in methods:
        logger.info(f"\n{'='*40}")
        logger.info(f"Computing derived features for: {method}")
        logger.info(f"{'='*40}")
        
        output_dir = build_derived_dir(base_dir, method)
        logger.info(f"Output directory: {output_dir}")
        
        stats = {"success": 0, "failed": 0, "skipped": 0}
        
        for sample_id in tqdm(sample_ids, desc=f"Computing {method}"):
            try:
                # 检查是否已存在
                output_path = output_dir / f"{sample_id}.pt"
                if output_path.exists() and not cfg.get("overwrite", False):
                    stats["skipped"] += 1
                    continue
                
                # 加载基础特征
                base_features = load_base_features(base_dir, sample_id)
                if not base_features:
                    logger.warning(f"No base features found for {sample_id}")
                    stats["failed"] += 1
                    continue
                
                # 计算派生特征
                derived = compute_method_features(method, base_features)
                
                # 保存
                save_derived_features(output_dir, sample_id, derived)
                
                stats["success"] += 1
                
            except Exception as e:
                logger.error(f"Failed to compute {method} for {sample_id}: {e}")
                stats["failed"] += 1
        
        # 保存统计
        logger.info(f"{method} complete: {stats}")
        with open(output_dir / "computation_stats.json", "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
