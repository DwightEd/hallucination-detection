#!/usr/bin/env python3
"""extract_hidden_states.py - 提取 Hidden States 特征。

支持：
- 层选择（all, last, last_n, last_half 等）
- 位置选择（response_only, prompt_only, full）
- 池化策略（none, mean, max, last）
- 半精度存储

Usage:
    # 提取所有层的 hidden states
    python scripts/features/extract_hidden_states.py \
        dataset.name=ragtruth \
        model=mistral_7b

    # 只提取后半部分层（用于 haloscope）
    python scripts/features/extract_hidden_states.py \
        layers=last_half \
        dataset.name=ragtruth \
        model=mistral_7b

    # 提取最后 4 层并做池化
    python scripts/features/extract_hidden_states.py \
        layers=last_4 \
        pooling=mean \
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
from src.features.base.hidden_states import (
    HiddenStatesExtractor,
    HiddenStatesExtractionConfig,
    PoolingStrategy,
)

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


def get_layer_config(cfg: DictConfig, n_layers: int) -> list:
    """解析层配置。"""
    layer_cfg = cfg.get("layers", "all")
    
    if layer_cfg == "all":
        return list(range(n_layers + 1))  # 包含 embedding 层
    elif layer_cfg == "last":
        return [n_layers]
    elif layer_cfg == "last_4":
        return list(range(max(0, n_layers - 3), n_layers + 1))
    elif layer_cfg == "last_half":
        return list(range(n_layers // 2, n_layers + 1))
    elif isinstance(layer_cfg, (list, tuple)):
        return [l if l >= 0 else n_layers + 1 + l for l in layer_cfg]
    else:
        return list(range(n_layers + 1))


def get_pooling_strategy(cfg: DictConfig) -> PoolingStrategy:
    """解析池化策略。"""
    pooling = cfg.get("pooling", "none")
    
    mapping = {
        "none": PoolingStrategy.NONE,
        "mean": PoolingStrategy.MEAN,
        "max": PoolingStrategy.MAX,
        "last": PoolingStrategy.LAST,
        "first": PoolingStrategy.FIRST,
        "response_mean": PoolingStrategy.RESPONSE_MEAN,
    }
    
    return mapping.get(pooling.lower(), PoolingStrategy.NONE)


def get_scope(cfg: DictConfig) -> str:
    """获取提取范围。"""
    return cfg.get("scope", "response")  # full, prompt, response


# =============================================================================
# 特征提取
# =============================================================================

def extract_hidden_states(
    model,
    sample,
    device: str,
    layer_indices: list = None,
    pooling: PoolingStrategy = PoolingStrategy.NONE,
    scope: str = "response",
    half_precision: bool = True,
) -> dict:
    """提取 hidden states。"""
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
            output_hidden_states=True,
            return_dict=True,
        )
    
    hidden_states = outputs.hidden_states  # tuple of [batch, seq_len, hidden_dim]
    n_layers = len(hidden_states) - 1  # 不包含 embedding
    hidden_dim = hidden_states[0].shape[-1]
    
    # 选择层
    if layer_indices is None:
        layer_indices = list(range(n_layers + 1))
    
    # 确定范围
    if scope == "response":
        start_idx = prompt_len
        end_idx = seq_len
    elif scope == "prompt":
        start_idx = 0
        end_idx = prompt_len
    else:  # full
        start_idx = 0
        end_idx = seq_len
    
    # 提取并处理
    hs_list = []
    for layer_idx in layer_indices:
        hs = hidden_states[layer_idx].squeeze(0)  # [seq_len, hidden_dim]
        
        # 范围选择
        hs = hs[start_idx:end_idx]
        
        # 池化
        if pooling == PoolingStrategy.MEAN:
            hs = hs.mean(dim=0, keepdim=True)
        elif pooling == PoolingStrategy.MAX:
            hs = hs.max(dim=0, keepdim=True)[0]
        elif pooling == PoolingStrategy.LAST:
            hs = hs[-1:, :]
        elif pooling == PoolingStrategy.FIRST:
            hs = hs[:1, :]
        
        if half_precision:
            hs = hs.half()
        
        hs_list.append(hs.cpu())
    
    # 堆叠
    if pooling != PoolingStrategy.NONE:
        hidden_states_tensor = torch.cat(hs_list, dim=0)  # [n_layers, hidden_dim]
    else:
        hidden_states_tensor = torch.stack(hs_list, dim=0)  # [n_layers, seq_len, hidden_dim]
    
    # 清理
    del outputs.hidden_states, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "hidden_states": hidden_states_tensor,
        "n_layers": len(layer_indices),
        "hidden_dim": hidden_dim,
        "seq_len": end_idx - start_idx,
        "prompt_len": prompt_len,
        "response_len": response_len,
        "layer_indices": layer_indices,
        "pooling": pooling.name,
        "scope": scope,
    }


# =============================================================================
# 主函数
# =============================================================================

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """主函数。"""
    logger.info("=" * 60)
    logger.info("Hidden States Feature Extraction")
    logger.info("=" * 60)
    
    # 构建输出目录
    output_dir = build_output_dir(cfg)
    save_dir = output_dir / "base" / "hidden_states"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {save_dir}")
    
    # 加载模型
    logger.info(f"Loading model: {cfg.model.name}")
    model = load_model(cfg.model)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取模型配置
    if hasattr(model.model.config, "num_hidden_layers"):
        n_layers = model.model.config.num_hidden_layers
    else:
        n_layers = 32
    
    if hasattr(model.model.config, "hidden_size"):
        hidden_dim = model.model.config.hidden_size
    else:
        hidden_dim = 4096
    
    layer_indices = get_layer_config(cfg, n_layers)
    pooling = get_pooling_strategy(cfg)
    scope = get_scope(cfg)
    
    logger.info(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    logger.info(f"Using layers: {layer_indices}")
    logger.info(f"Pooling: {pooling.name}")
    logger.info(f"Scope: {scope}")
    
    # 加载数据
    logger.info(f"Loading dataset: {cfg.dataset.name}")
    samples = load_dataset(cfg.dataset)
    logger.info(f"Total samples: {len(samples)}")
    
    # 保存配置
    config_info = {
        "layer_indices": layer_indices,
        "pooling": pooling.name,
        "scope": scope,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    with open(save_dir / "extraction_config.json", "w") as f:
        json.dump(config_info, f, indent=2, default=str)
    
    # 提取特征
    stats = {"success": 0, "failed": 0, "skipped": 0}
    clear_cache_every = cfg.get("clear_cache_every", 10)
    
    for idx, sample in enumerate(tqdm(samples, desc="Extracting hidden states")):
        try:
            # 检查是否已存在
            save_path = save_dir / f"{sample.id}.pt"
            if save_path.exists() and not cfg.get("overwrite", False):
                stats["skipped"] += 1
                continue
            
            # 提取特征
            features = extract_hidden_states(
                model, sample, device,
                layer_indices=layer_indices,
                pooling=pooling,
                scope=scope,
                half_precision=cfg.get("half_precision", True),
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
