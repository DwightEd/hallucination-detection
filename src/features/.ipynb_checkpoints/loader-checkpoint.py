"""Unified Feature Loader - 统一的特征加载模块。

整合了 train_probe.py 和 evaluate.py 中重复的特征加载逻辑，
并集成 DerivedFeatureManager 以支持按需计算衍生特征。

核心功能：
1. 统一加载 features/, features_individual/, answers.json 等文件
2. 自动计算缺失的衍生特征（如 laplacian_diags）
3. 支持懒加载大型特征（hidden_states, full_attention）
4. 提供一致的 ExtractedFeatures 列表

Usage:
    from src.features.loader import FeatureLoader
    
    loader = FeatureLoader(features_dir)
    features_list, samples = loader.load()
    
    # 或者使用便捷函数
    from src.features.loader import load_features_for_method
    features_list, samples = load_features_for_method(
        features_dir, 
        method_name="lapeigvals",
        load_token_labels=True
    )
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass

import torch
import numpy as np

from src.core import (
    Sample, ExtractedFeatures, TaskType, SplitType,
    sanitize_sample_id,
)

# 从统一注册中心导入特征需求（保持向后兼容）
from src.features.registry import (
    DERIVED_FEATURE_DEPS,
    get_method_feature_set,
)

logger = logging.getLogger(__name__)

# 向后兼容：从 registry 动态构建 METHOD_REQUIRED_FEATURES
# 注意：这是从详细的 FeatureRequirements 转换为简化的 Set[str]
from src.features.registry import METHOD_FEATURE_REQUIREMENTS as _FULL_REQUIREMENTS
METHOD_REQUIRED_FEATURES = {
    name: req.to_feature_set() 
    for name, req in _FULL_REQUIREMENTS.items()
}


@dataclass
class LoadedFeatureSet:
    """加载的特征集合"""
    features_list: List[ExtractedFeatures]
    samples: List[Sample]
    available_features: Set[str]
    missing_features: Set[str]


class FeatureLoader:
    """统一的特征加载器。
    
    负责：
    1. 从 features/ 和 features_individual/ 加载特征
    2. 加载 metadata.json 和 answers.json
    3. 按需计算缺失的衍生特征
    4. 构建 ExtractedFeatures 对象列表
    """
    
    def __init__(
        self,
        features_dir: Path,
        compute_derived: bool = True,
        cache_derived: bool = True,
    ):
        """
        Args:
            features_dir: 特征目录路径
            compute_derived: 是否自动计算缺失的衍生特征
            cache_derived: 是否缓存计算的衍生特征
        """
        self.features_dir = Path(features_dir)
        self.compute_derived = compute_derived
        self.cache_derived = cache_derived
        
        # 子目录
        self.consolidated_dir = self.features_dir / "features"
        self.individual_dir = self.features_dir / "features_individual"
        
        # 缓存
        self._loaded_features: Dict[str, Dict[str, torch.Tensor]] = {}
        self._feature_indexes: Dict[str, Dict[str, Any]] = {}
        self._derived_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    
    def load(
        self,
        method_name: Optional[str] = None,
        load_token_labels: bool = False,
    ) -> Tuple[List[ExtractedFeatures], List[Sample]]:
        """加载特征和样本。
        
        Args:
            method_name: 方法名称（用于确定需要哪些特征）
            load_token_labels: 是否加载 token 级别标签
            
        Returns:
            (features_list, samples)
        """
        # 1. 加载元数据和答案
        metadata = self._load_metadata()
        samples = self._load_samples()
        
        # 2. 加载合并的特征文件
        self._load_consolidated_features()
        
        # 3. 加载大型特征索引
        self._load_feature_indexes()
        
        # 4. 确定需要的特征
        required_features = set()
        if method_name:
            required_features = METHOD_REQUIRED_FEATURES.get(method_name, set())
        
        # 5. 计算缺失的衍生特征
        if self.compute_derived:
            self._compute_missing_derived_features(required_features)
        
        # 6. 构建 ExtractedFeatures 列表
        sample_ids = metadata.get("sample_ids", [s.id for s in samples])
        features_list = self._build_features_list(
            sample_ids=sample_ids,
            samples=samples,
            load_token_labels=load_token_labels,
        )
        
        return features_list, samples
    
    def _load_metadata(self) -> Dict[str, Any]:
        """加载 metadata.json"""
        metadata_path = self.features_dir / "metadata.json"
        if not metadata_path.exists():
            logger.warning(f"metadata.json not found in {self.features_dir}")
            return {}
        
        with open(metadata_path) as f:
            return json.load(f)
    
    def _load_samples(self) -> List[Sample]:
        """从 answers.json 加载样本"""
        answers_path = self.features_dir / "answers.json"
        if not answers_path.exists():
            logger.warning(f"answers.json not found in {self.features_dir}")
            return []
        
        samples = []
        with open(answers_path) as f:
            answers = json.load(f)
        
        for ans in answers:
            task_type = None
            if ans.get("task_type"):
                try:
                    task_type = TaskType(ans["task_type"])
                except (ValueError, KeyError):
                    pass
            
            split = None
            if ans.get("split"):
                try:
                    split = SplitType(ans["split"])
                except (ValueError, KeyError):
                    pass
            
            samples.append(Sample(
                id=ans["id"],
                prompt=ans.get("prompt", ""),
                response=ans.get("response", ""),
                label=ans.get("label", 0),
                task_type=task_type,
                split=split,
                metadata={
                    "source_model": ans.get("source_model"),
                    "prompt_len": ans.get("prompt_len", 0),
                    "response_len": ans.get("response_len", 0),
                    "hallucination_spans": ans.get("labels", []),
                }
            ))
        
        return samples
    
    def _load_consolidated_features(self):
        """加载合并的特征文件"""
        if not self.consolidated_dir.exists():
            self.consolidated_dir = self.features_dir
        
        feature_files = {
            "attn_diags": "attn_diags.pt",
            "laplacian_diags": "laplacian_diags.pt",
            "attn_entropy": "attn_entropy.pt",
            "token_probs": "token_probs.pt",
            "token_entropy": "token_entropy.pt",
            "hallucination_labels": "hallucination_labels.pt",
            "hallucination_token_spans": "hallucination_token_spans.pt",
        }
        
        for key, filename in feature_files.items():
            filepath = self.consolidated_dir / filename
            if filepath.exists():
                try:
                    self._loaded_features[key] = torch.load(filepath, weights_only=False)
                    logger.debug(f"Loaded {key}: {len(self._loaded_features[key])} samples")
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
    
    def _load_feature_indexes(self):
        """加载大型特征索引"""
        index_files = {
            "hidden_states": "hidden_states_index.json",
            "full_attentions": "full_attentions_index.json",
        }
        
        for key, filename in index_files.items():
            filepath = self.consolidated_dir / filename
            if filepath.exists():
                try:
                    with open(filepath) as f:
                        self._feature_indexes[key] = json.load(f)
                    logger.debug(f"Loaded {key} index: {self._feature_indexes[key].get('sample_count', 0)} samples")
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
    
    def _compute_missing_derived_features(self, required_features: Set[str]):
        """计算缺失的衍生特征（带时间记录）"""
        import time
        
        total_start = time.time()
        computed = []
        
        # 检查 laplacian_diags
        if "laplacian_diags" in required_features:
            if "laplacian_diags" not in self._loaded_features:
                if "attn_diags" in self._loaded_features:
                    start = time.time()
                    logger.info("Computing laplacian_diags from attn_diags...")
                    self._compute_laplacian_diags()
                    elapsed = time.time() - start
                    computed.append(("laplacian_diags", elapsed))
        
        # 检查 token_entropy
        if "token_entropy" in required_features:
            if "token_entropy" not in self._loaded_features:
                if "token_probs" in self._loaded_features:
                    start = time.time()
                    logger.info("Computing token_entropy from token_probs...")
                    self._compute_token_entropy()
                    elapsed = time.time() - start
                    computed.append(("token_entropy", elapsed))
        
        # 记录总时间
        if computed:
            total_time = time.time() - total_start
            logger.info(f"Derived feature computation complete in {total_time:.2f}s")
            for name, t in computed:
                logger.info(f"  - {name}: {t:.2f}s")
    
    def _compute_laplacian_diags(self):
        """从 attn_diags 计算 laplacian_diags
        
        对于归一化的 attention 矩阵：
        - 行和 D_ii ≈ 1（每行和为1）
        - Laplacian L_ii = D_ii - A_ii = 1 - attn_diag
        """
        attn_diags_data = self._loaded_features.get("attn_diags", {})
        laplacian_diags_data = {}
        
        for sid, attn_diag in attn_diags_data.items():
            if isinstance(attn_diag, torch.Tensor):
                laplacian_diags_data[sid] = 1.0 - attn_diag
        
        if laplacian_diags_data:
            self._loaded_features["laplacian_diags"] = laplacian_diags_data
            self._derived_cache["laplacian_diags"] = laplacian_diags_data
            logger.info(f"Computed laplacian_diags for {len(laplacian_diags_data)} samples")
            
            # 可选：保存到磁盘
            if self.cache_derived:
                self._save_derived_feature("laplacian_diags", laplacian_diags_data)
    
    def _compute_token_entropy(self):
        """从 token_probs 计算 token_entropy"""
        token_probs_data = self._loaded_features.get("token_probs", {})
        token_entropy_data = {}
        
        for sid, probs in token_probs_data.items():
            if isinstance(probs, torch.Tensor):
                eps = 1e-10
                probs_clamped = probs.clamp(min=eps)
                token_entropy_data[sid] = -probs_clamped * torch.log(probs_clamped)
        
        if token_entropy_data:
            self._loaded_features["token_entropy"] = token_entropy_data
            self._derived_cache["token_entropy"] = token_entropy_data
            logger.info(f"Computed token_entropy for {len(token_entropy_data)} samples")
    
    def _save_derived_feature(self, feature_name: str, data: Dict[str, torch.Tensor]):
        """保存衍生特征到磁盘"""
        try:
            output_path = self.consolidated_dir / f"{feature_name}.pt"
            torch.save(data, output_path)
            logger.info(f"Saved derived feature {feature_name} to {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save derived feature {feature_name}: {e}")
    
    def _get_feature_for_sample(
        self,
        sample_id: str,
        feature_key: str,
    ) -> Optional[torch.Tensor]:
        """获取单个样本的特征值"""
        data = self._loaded_features.get(feature_key, {})
        
        if not isinstance(data, dict):
            return None
        
        # 尝试多种键匹配方式
        if sample_id in data:
            return data[sample_id]
        if str(sample_id) in data:
            return data[str(sample_id)]
        
        # 尝试数值索引
        try:
            idx = int(sample_id)
            if idx in data:
                return data[idx]
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _build_features_list(
        self,
        sample_ids: List[str],
        samples: List[Sample],
        load_token_labels: bool = False,
    ) -> List[ExtractedFeatures]:
        """构建 ExtractedFeatures 列表"""
        
        # 加载标签
        labels_path = self.features_dir / "labels.pt"
        if labels_path.exists():
            labels = torch.load(labels_path, weights_only=False)
        else:
            labels = torch.tensor([s.label or 0 for s in samples])
        
        features_list = []
        first_sample_debug = True
        
        for i, sample_id in enumerate(sample_ids):
            # 获取各类特征
            sample_features = {}
            for key in ["attn_diags", "laplacian_diags", "attn_entropy", 
                        "token_probs", "token_entropy"]:
                val = self._get_feature_for_sample(sample_id, key)
                if val is not None:
                    sample_features[key] = val
            
            # 调试信息（仅第一个样本）
            if first_sample_debug:
                first_sample_debug = False
                logger.info(f"=== First sample feature check ===")
                logger.info(f"  sample_id: {sample_id}")
                logger.info(f"  Features matched: {list(sample_features.keys())}")
                if "hidden_states" in self._feature_indexes:
                    logger.info(f"  hidden_states: lazy-load available")
                if not sample_features:
                    logger.warning(f"  WARNING: No features matched for first sample!")
            
            # 构建懒加载路径
            feature_paths = {}
            for feature_key in ["hidden_states", "full_attentions"]:
                if feature_key in self._feature_indexes:
                    index_data = self._feature_indexes[feature_key]
                    if "index" in index_data and sample_id in index_data["index"]:
                        feature_paths[feature_key] = index_data["index"][sample_id]
            
            # 获取样本信息
            sample = samples[i] if i < len(samples) else None
            
            # Token-level 标签
            hallucination_labels = None
            hallucination_token_spans = None
            
            if load_token_labels and self.individual_dir.exists():
                # 从 individual 文件加载
                safe_id = sanitize_sample_id(sample_id)
                individual_file = self.individual_dir / f"{safe_id}.pt"
                
                if individual_file.exists():
                    try:
                        data = torch.load(individual_file, map_location="cpu", weights_only=False)
                        feat_data = data.get("features", {})
                        hallucination_labels = feat_data.get("hallucination_labels")
                        hallucination_token_spans = feat_data.get("hallucination_token_spans")
                    except Exception:
                        pass
            
            # 构建元数据
            sample_metadata = {
                "task_type_str": sample.task_type.value if sample and sample.task_type else "unknown",
                "_feature_paths": feature_paths,
            }
            
            # 创建 ExtractedFeatures
            features_list.append(ExtractedFeatures(
                sample_id=sample_id,
                prompt_len=sample.metadata.get("prompt_len", 0) if sample else 0,
                response_len=sample.metadata.get("response_len", 0) if sample else 0,
                label=int(labels[i]) if i < len(labels) else 0,
                attn_diags=sample_features.get("attn_diags"),
                laplacian_diags=sample_features.get("laplacian_diags"),
                attn_entropy=sample_features.get("attn_entropy"),
                hidden_states=None,  # 懒加载
                token_probs=sample_features.get("token_probs"),
                token_entropy=sample_features.get("token_entropy"),
                full_attention=None,  # 懒加载
                hallucination_labels=hallucination_labels,
                hallucination_token_spans=hallucination_token_spans,
                metadata=sample_metadata,
            ))
        
        # 统计信息
        n_with_labels = sum(1 for f in features_list if f.hallucination_labels is not None)
        n_hallucinated = sum(1 for f in features_list if f.label == 1)
        logger.info(f"Loaded {len(features_list)} samples")
        if load_token_labels:
            logger.info(f"Token-level labels: {n_with_labels}/{n_hallucinated} hallucinated samples")
        
        return features_list
    
    def get_available_features(self) -> Set[str]:
        """获取可用的特征类型"""
        return set(self._loaded_features.keys())


# =============================================================================
# 便捷函数
# =============================================================================

def load_features_for_method(
    features_dir: Path,
    method_name: str,
    load_token_labels: bool = False,
) -> Tuple[List[ExtractedFeatures], List[Sample]]:
    """为特定方法加载特征。
    
    自动计算方法所需的衍生特征。
    
    Args:
        features_dir: 特征目录
        method_name: 方法名称
        load_token_labels: 是否加载 token 级别标签
        
    Returns:
        (features_list, samples)
    """
    loader = FeatureLoader(features_dir)
    return loader.load(method_name=method_name, load_token_labels=load_token_labels)


def load_features(
    features_dir: Path,
    load_token_labels: bool = False,
) -> Tuple[List[ExtractedFeatures], List[Sample]]:
    """加载特征（通用版本）。
    
    Args:
        features_dir: 特征目录
        load_token_labels: 是否加载 token 级别标签
        
    Returns:
        (features_list, samples)
    """
    loader = FeatureLoader(features_dir)
    return loader.load(load_token_labels=load_token_labels)


def split_features_by_split(
    features_list: List[ExtractedFeatures],
    samples: List[Sample],
) -> Tuple[List[ExtractedFeatures], List[int], List[ExtractedFeatures], List[int]]:
    """按 split 字段分割数据。
    
    Returns:
        (train_features, train_labels, test_features, test_labels)
    """
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    
    for feat, sample in zip(features_list, samples):
        label = feat.label if feat.label is not None else (sample.label or 0)
        
        if sample.split == SplitType.TRAIN:
            train_features.append(feat)
            train_labels.append(label)
        else:
            test_features.append(feat)
            test_labels.append(label)
    
    return train_features, train_labels, test_features, test_labels


# =============================================================================
# 衍生特征计算辅助函数
# =============================================================================

def compute_laplacian_from_diags(
    attn_diags: torch.Tensor, 
    attn_row_sums: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """从对角线计算 Laplacian 对角线。
    
    对于归一化的 attention 矩阵：
    - 行和 D_ii ≈ 1（每行和为1）
    - Laplacian L_ii = D_ii - A_ii = 1 - attn_diag
    
    Args:
        attn_diags: attention 对角线值
        attn_row_sums: 可选的行和，如果为 None 则假设为 1
        
    Returns:
        Laplacian 对角线值
    """
    if attn_row_sums is not None:
        return attn_row_sums - attn_diags
    return 1.0 - attn_diags


def compute_entropy_from_attention(attention: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """从完整 attention 矩阵计算熵。
    
    H = -sum(p * log(p))
    
    Args:
        attention: attention 矩阵
        eps: 数值稳定性的小常数
        
    Returns:
        attention 熵
    """
    attention = attention.clamp(min=eps)
    return -torch.sum(attention * torch.log(attention), dim=-1)