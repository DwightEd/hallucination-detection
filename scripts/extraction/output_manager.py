"""Output management module for feature extraction.

重构版本：使用统一的 PathManager 管理路径。

负责：
- 构建输出目录路径
- 特征字典转换
- 结果整合和元数据保存
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from src.core import Sample, ExtractedFeatures, PathManager

logger = logging.getLogger(__name__)


def build_output_dir(cfg: DictConfig, split_name: str = "train") -> Path:
    """构建输出目录路径。
    
    使用 PathManager 确保与其他脚本路径一致。
    
    Args:
        cfg: 配置对象
        split_name: "train" 或 "test"
        
    Returns:
        输出目录路径
    """
    pm = PathManager.from_config(cfg)
    return pm.get_features_dir(split=split_name)


def extract_features_dict(features: ExtractedFeatures) -> Dict[str, Any]:
    """将 ExtractedFeatures 转换为可序列化的字典格式。
    
    Args:
        features: ExtractedFeatures 对象
        
    Returns:
        特征字典
    """
    result = {
        "sample_id": features.sample_id,
        "prompt_len": features.prompt_len,
        "response_len": features.response_len,
        "label": features.label,
        "layers": features.layers,
    }
    
    # 标准特征属性
    for attr in ['attn_diags', 'laplacian_diags', 'attn_entropy', 
                 'hidden_states', 'token_probs', 'token_entropy']:
        value = getattr(features, attr, None)
        if value is not None:
            result[attr] = value
    
    # Full attention 特殊处理
    if features.full_attention is not None:
        result['full_attentions'] = features.full_attention
    
    # Token-level hallucination labels
    if features.hallucination_labels is not None:
        result['hallucination_labels'] = features.hallucination_labels
    if features.hallucination_token_spans is not None:
        result['hallucination_token_spans'] = features.hallucination_token_spans
    
    return result


def save_sample_answer(sample: Sample, features: ExtractedFeatures) -> Dict[str, Any]:
    """创建样本答案条目。
    
    Args:
        sample: Sample 对象
        features: ExtractedFeatures 对象
        
    Returns:
        答案字典
    """
    return {
        "id": sample.id,
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
        "task_type": sample.task_type.value if sample.task_type else None,
        "split": sample.split.value if sample.split else None,
        "source_model": sample.metadata.get("source_model"),
        "labels": sample.metadata.get("hallucination_spans", []),
        "prompt_len": features.prompt_len,
        "response_len": features.response_len,
    }


def compute_stable_config_hash(cfg: DictConfig, split_name: str) -> Dict[str, Any]:
    """计算稳定的配置哈希字典，用于断点续传。
    
    注意：这个哈希只用于日志记录和参考，实际的断点续传
    基于文件存在性（ignore_config_change=True时）。
    
    只包含真正影响原始特征提取的参数：
    - dataset_name: 数据集
    - model_name: 模型
    - split_name: 数据集分割
    - features_mode: 特征提取模式
    - max_length: 最大序列长度
    
    不包含的参数（这些只影响后续处理，不影响原始特征）：
    - methods: 检测方法
    - seed: 随机种子
    - allow_full_attention: 是否保存完整注意力矩阵
    
    Args:
        cfg: 配置对象
        split_name: split 名称
        
    Returns:
        配置哈希字典
    """
    pm = PathManager.from_config(cfg)
    
    return {
        "dataset_name": pm.dataset_name,
        "task_type": pm.task_suffix,
        "model_name": pm.model_short_name,
        "features_mode": cfg.features.get("mode", "teacher_forcing"),
        "max_length": cfg.features.get("max_length", 4096),
        "split_name": split_name,
    }


def finalize_outputs(
    output_dir: Path,
    checkpoint_manager,
    samples: List[Sample],
    cfg: Optional[DictConfig] = None
) -> None:
    """整合单个特征文件为合并格式。
    
    Args:
        output_dir: 输出目录
        checkpoint_manager: CheckpointManager 实例
        samples: 样本列表
        cfg: 配置对象（可选）
    """
    features_dir = output_dir / "features"
    metadata_path = output_dir / "metadata.json"
    labels_path = output_dir / "labels.pt"
    
    # 检查是否已经合并过
    if metadata_path.exists() and labels_path.exists():
        try:
            with open(metadata_path, "r") as f:
                existing_metadata = json.load(f)
            
            existing_n_processed = existing_metadata.get("n_processed", 0)
            expected_n_samples = len(samples)
            
            has_consolidated_features = (
                features_dir.exists() and 
                (any(features_dir.glob("*.pt")) or any(features_dir.glob("*_index.json")))
            )
            
            if existing_n_processed == expected_n_samples and has_consolidated_features:
                logger.info(f"Features already consolidated ({existing_n_processed} samples). Skipping.")
                logger.info(f"  To force re-consolidation, delete: {metadata_path}")
                return
        except Exception as e:
            logger.warning(f"Failed to check existing metadata: {e}, will re-consolidate")
    
    # 执行合并
    logger.info("Consolidating features (memory-optimized streaming mode)...")
    
    features_dir.mkdir(exist_ok=True)
    
    consolidated = checkpoint_manager.consolidate_features_streaming(features_dir)
    
    labels = torch.tensor([s.label if s.label is not None else -1 for s in samples])
    torch.save(labels, labels_path)
    logger.info(f"Saved labels: {len(labels)} samples")
    
    progress = checkpoint_manager.get_progress()
    metadata = {
        "n_samples": len(samples),
        "n_processed": progress["processed"],
        "n_failed": progress["failed"],
        "sample_ids": consolidated.get("sample_ids", []),
        "saved_features": consolidated.get("saved_features", []),
        "config": OmegaConf.to_container(cfg, resolve=True) if cfg else {},
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Consolidated features saved to {features_dir}")
    
    # 可选：清理单个特征文件
    cleanup_individual = cfg.get("cleanup_individual_features", False) if cfg else False
    if cleanup_individual:
        logger.info("Cleaning up individual feature files...")
        for pt_file in (output_dir / "features_individual").glob("*.pt"):
            try:
                pt_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove {pt_file}: {e}")


def save_answers(
    output_dir: Path,
    answers: List[Dict[str, Any]],
    is_resume: bool = False
) -> None:
    """保存答案文件，支持断点续传合并。
    
    Args:
        output_dir: 输出目录
        answers: 新答案列表
        is_resume: 是否为续传模式
    """
    answers_path = output_dir / "answers.json"
    
    existing_answers = []
    if is_resume and answers_path.exists():
        try:
            with open(answers_path) as f:
                existing_answers = json.load(f)
        except Exception:
            pass
    
    # 用新的覆盖旧的
    new_ids = {a["id"] for a in answers}
    existing_answers = [a for a in existing_answers if a["id"] not in new_ids]
    
    all_answers = existing_answers + answers
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(all_answers, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(all_answers)} answers to {answers_path}")


# =============================================================================
# 向后兼容的辅助函数
# =============================================================================

def get_task_suffix(cfg: DictConfig) -> str:
    """获取任务后缀（向后兼容）。
    
    建议使用 PathManager 替代此函数。
    """
    pm = PathManager.from_config(cfg)
    return pm.task_suffix


def get_model_short_name(cfg: DictConfig) -> str:
    """获取模型短名称（向后兼容）。
    
    建议使用 PathManager 替代此函数。
    """
    pm = PathManager.from_config(cfg)
    return pm.model_short_name
