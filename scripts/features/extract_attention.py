#!/usr/bin/env python3
"""extract_attention.py - 提取 Attention 特征。

支持两种模式：
1. 存储完整 full_attention（用于需要复杂派生特征的方法）
2. 直接计算并存储派生特征（节省存储空间）

根据方法需求自动决定：
- 若方法需要 lookback_ratio 或 mva_features -> 存储 full_attention
- 若方法只需要 attention_diags 或 laplacian_diags -> 直接计算派生特征

Usage:
    # 自动决定存储模式（基于方法需求）
    python scripts/features/extract_attention.py \
        methods=[lapeigvals,lookback_lens] \
        dataset.name=ragtruth \
        model=mistral_7b
    
    # 强制存储 full_attention
    python scripts/features/extract_attention.py \
        storage_mode=full \
        dataset.name=ragtruth \
        model=mistral_7b
    
    # 只计算派生特征
    python scripts/features/extract_attention.py \
        storage_mode=derived \
        methods=[lapeigvals] \
        dataset.name=ragtruth \
        model=mistral_7b
"""

import os
import sys
from pathlib import Path

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
    should_store_full_attention,
    describe_requirements,
    DerivedFeatureType,
    FeatureScope,
)
from src.features.base.attention import (
    AttentionExtractor,
    AttentionExtractionConfig,
)
from src.features.derived.attention_diags import compute_attention_diags_direct
from src.features.derived.laplacian_diags import compute_laplacian_diags_direct
from src.features.derived.attention_entropy import compute_attention_entropy_direct
from src.features.derived.lookback_ratio import compute_lookback_ratio_direct
from src.features.derived.mva_features import compute_mva_features_direct

logger = logging.getLogger(__name__)


# =============================================================================
# 辅助函数
# =============================================================================

def build_output_dir(cfg: DictConfig) -> Path:
    """构建输出目录路径。"""
    base_dir = Path(cfg.get("output_dir", "outputs/features"))
    dataset_name = cfg.dataset.name
    model_name = cfg.model.get("short_name", cfg.model.name.split("/")[-1])
    seed = cfg.get("seed", 42)
    task_type = cfg.dataset.get("task_type", "default")
    
    return base_dir / dataset_name / model_name / f"seed_{seed}" / task_type


def determine_storage_mode(cfg: DictConfig) -> str:
    """确定存储模式：'full' 或 'derived'。"""
    # 1. 如果显式指定
    if "storage_mode" in cfg:
        return cfg.storage_mode
    
    # 2. 根据方法需求决定
    if "methods" in cfg and cfg.methods:
        methods = list(cfg.methods) if hasattr(cfg.methods, "__iter__") else [cfg.methods]
        requirements = compute_union_requirements(methods)
        
        if should_store_full_attention(requirements):
            return "full"
        else:
            return "derived"
    
    # 3. 默认存储 full
    return "full"


def get_layer_config(cfg: DictConfig, n_layers: int) -> list:
    """解析层配置。"""
    layer_cfg = cfg.get("layers", "all")
    
    if layer_cfg == "all":
        return list(range(n_layers))
    elif layer_cfg == "last":
        return [n_layers - 1]
    elif layer_cfg == "last_4":
        return list(range(max(0, n_layers - 4), n_layers))
    elif layer_cfg == "last_half":
        return list(range(n_layers // 2, n_layers))
    elif isinstance(layer_cfg, (list, tuple)):
        return [l if l >= 0 else n_layers + l for l in layer_cfg]
    else:
        return list(range(n_layers))


def needs_prompt_attention(cfg: DictConfig) -> bool:
    """检查是否需要 prompt 部分的 attention。"""
    if "methods" in cfg and cfg.methods:
        methods = list(cfg.methods) if hasattr(cfg.methods, "__iter__") else [cfg.methods]
        requirements = compute_union_requirements(methods)
        return requirements.uses_prompt_attention
    
    # 默认提取完整序列
    return True


# =============================================================================
# 特征提取
# =============================================================================

def extract_full_attention(
    model,
    sample,
    device: str,
    layer_indices: list = None,
    half_precision: bool = True,
) -> dict:
    """提取完整 attention 矩阵。"""
    # 分词
    prompt_ids = model.encode(sample.prompt, add_special_tokens=True)
    response_ids = model.encode(sample.response, add_special_tokens=False)
    
    prompt_len = prompt_ids.size(1)
    response_len = response_ids.size(1)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1).to(device)
    seq_len = input_ids.size(1)
    
    # 前向传播
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            output_attentions=True,
            return_dict=True,
        )
    
    attentions = outputs.attentions
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    
    # 选择层
    if layer_indices is None:
        layer_indices = list(range(n_layers))
    
    # 堆叠为 tensor
    attn_list = []
    for layer_idx in layer_indices:
        attn = attentions[layer_idx].squeeze(0)  # [n_heads, seq_len, seq_len]
        if half_precision:
            attn = attn.half()
        attn_list.append(attn.cpu())
    
    full_attention = torch.stack(attn_list, dim=0)  # [n_layers, n_heads, seq_len, seq_len]
    
    # 清理
    del outputs.attentions, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "full_attention": full_attention,
        "n_layers": len(layer_indices),
        "n_heads": n_heads,
        "seq_len": seq_len,
        "prompt_len": prompt_len,
        "response_len": response_len,
        "layer_indices": layer_indices,
    }


def extract_derived_attention_features(
    model,
    sample,
    device: str,
    required_derived: list,
    layer_indices: list = None,
    need_prompt: bool = False,
) -> dict:
    """直接从模型输出计算派生特征（不存储 full_attention）。"""
    # 分词
    prompt_ids = model.encode(sample.prompt, add_special_tokens=True)
    response_ids = model.encode(sample.response, add_special_tokens=False)
    
    prompt_len = prompt_ids.size(1)
    response_len = response_ids.size(1)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1).to(device)
    
    # 前向传播
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            output_attentions=True,
            return_dict=True,
        )
    
    attentions = outputs.attentions
    results = {
        "prompt_len": prompt_len,
        "response_len": response_len,
        "n_layers": len(attentions),
        "n_heads": attentions[0].shape[1],
    }
    
    # 计算各派生特征
    for derived_type in required_derived:
        if derived_type == "attention_diags":
            results["attention_diags"] = compute_attention_diags_direct(
                attentions,
                layers=layer_indices,
                prompt_len=prompt_len,
                response_len=response_len,
                response_only=not need_prompt,
            )
        
        elif derived_type == "laplacian_diags":
            results["laplacian_diags"] = compute_laplacian_diags_direct(
                attentions,
                layers=layer_indices,
                prompt_len=prompt_len,
                response_len=response_len,
                response_only=not need_prompt,
            )
        
        elif derived_type == "attention_entropy":
            results["attention_entropy"] = compute_attention_entropy_direct(
                attentions,
                layers=layer_indices,
                prompt_len=prompt_len,
                response_len=response_len,
                response_only=not need_prompt,
            )
        
        elif derived_type == "lookback_ratio":
            if prompt_len > 0:  # lookback 需要 prompt
                results["lookback_ratio"] = compute_lookback_ratio_direct(
                    attentions,
                    prompt_len=prompt_len,
                    response_len=response_len,
                    layers=layer_indices,
                )
        
        elif derived_type == "mva_features":
            mva = compute_mva_features_direct(
                attentions,
                prompt_len=prompt_len,
                response_len=response_len,
                layers=layer_indices,
            )
            results["mva_avg_in"] = mva["avg_in"]
            results["mva_div_in"] = mva["div_in"]
            results["mva_div_out"] = mva["div_out"]
    
    # 清理
    del outputs.attentions, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def get_required_derived_features(cfg: DictConfig) -> list:
    """获取需要计算的派生特征列表。"""
    if "methods" in cfg and cfg.methods:
        methods = list(cfg.methods) if hasattr(cfg.methods, "__iter__") else [cfg.methods]
        requirements = compute_union_requirements(methods)
        
        derived = []
        for df in requirements.derived_features:
            derived.append(df.feature_type.value)
        return list(set(derived))
    
    # 默认派生特征
    return ["attention_diags", "laplacian_diags"]


# =============================================================================
# 主函数
# =============================================================================

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """主函数。"""
    logger.info("=" * 60)
    logger.info("Attention Feature Extraction")
    logger.info("=" * 60)
    
    # 确定存储模式
    storage_mode = determine_storage_mode(cfg)
    logger.info(f"Storage mode: {storage_mode}")
    
    # 打印方法需求
    if "methods" in cfg and cfg.methods:
        methods = list(cfg.methods) if hasattr(cfg.methods, "__iter__") else [cfg.methods]
        requirements = compute_union_requirements(methods)
        logger.info(describe_requirements(requirements))
    
    # 构建输出目录
    output_dir = build_output_dir(cfg)
    
    if storage_mode == "full":
        save_dir = output_dir / "base" / "full_attention"
    else:
        save_dir = output_dir / "base" / "attention_derived"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {save_dir}")
    
    # 加载模型
    logger.info(f"Loading model: {cfg.model.name}")
    model = load_model(cfg.model)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取模型层数
    if hasattr(model.model.config, "num_hidden_layers"):
        n_layers = model.model.config.num_hidden_layers
    else:
        n_layers = 32  # 默认
    
    layer_indices = get_layer_config(cfg, n_layers)
    logger.info(f"Using layers: {layer_indices}")
    
    # 加载数据
    logger.info(f"Loading dataset: {cfg.dataset.name}")
    samples = load_dataset(cfg.dataset)
    logger.info(f"Total samples: {len(samples)}")
    
    # 获取需要的派生特征
    required_derived = get_required_derived_features(cfg)
    need_prompt = needs_prompt_attention(cfg)
    logger.info(f"Required derived features: {required_derived}")
    logger.info(f"Need prompt attention: {need_prompt}")
    
    # 保存配置
    config_info = {
        "storage_mode": storage_mode,
        "layer_indices": layer_indices,
        "required_derived": required_derived,
        "need_prompt_attention": need_prompt,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    with open(save_dir / "extraction_config.json", "w") as f:
        json.dump(config_info, f, indent=2, default=str)
    
    # 提取特征
    stats = {"success": 0, "failed": 0, "skipped": 0}
    clear_cache_every = cfg.get("clear_cache_every", 10)
    
    for idx, sample in enumerate(tqdm(samples, desc="Extracting attention")):
        try:
            # 检查是否已存在
            save_path = save_dir / f"{sample.id}.pt"
            if save_path.exists() and not cfg.get("overwrite", False):
                stats["skipped"] += 1
                continue
            
            # 提取特征
            if storage_mode == "full":
                features = extract_full_attention(
                    model, sample, device,
                    layer_indices=layer_indices,
                    half_precision=cfg.get("half_precision", True),
                )
            else:
                features = extract_derived_attention_features(
                    model, sample, device,
                    required_derived=required_derived,
                    layer_indices=layer_indices,
                    need_prompt=need_prompt,
                )
            
            # 添加元数据
            features["sample_id"] = sample.id
            features["label"] = sample.label
            
            # 保存
            torch.save(features, save_path)
            stats["success"] += 1
            
            # 内存清理
            if (idx + 1) % clear_cache_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logger.error(f"Failed for sample {sample.id}: {e}")
            import traceback
            traceback.print_exc()
            stats["failed"] += 1
    
    # 保存统计
    logger.info(f"Extraction complete: {stats}")
    with open(save_dir / "extraction_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
