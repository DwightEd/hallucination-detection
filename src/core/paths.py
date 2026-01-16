"""统一的路径管理模块。

集中管理所有路径构建逻辑，确保整个项目使用一致的目录结构。

目录结构说明：
==============

特征目录 (features_dir):
    {features_dir}/{dataset}/{model}/seed_{seed}/{task_suffix}/
    
    子目录：
    - features_individual/  : 单个样本的特征文件 (.pt)
    - features/            : 合并后的特征文件
    - metadata.json        : 元数据
    - answers.json         : 样本答案和标签
    - labels.pt            : 标签张量

模型目录 (models_dir):
    {models_dir}/{dataset}/{model}/seed_{seed}/{task_suffix}/{method}/{level}/
    
    其中 level = "sample" 或 "token"
    
    包含：
    - model.pkl           : 保存的模型
    - train_metrics.json  : 训练指标
    - eval_results.json   : 评估结果
    - config.yaml         : 配置备份

结果目录 (results_dir):
    {results_dir}/{dataset}/{model}/seed_{seed}/train_{train_task}_eval_{eval_task}/{method}/
    
    用于跨任务评估结果

Usage:
    from src.core.paths import PathManager
    
    # 从配置创建
    pm = PathManager.from_config(cfg)
    
    # 获取特征目录
    features_dir = pm.get_features_dir()
    features_dir_test = pm.get_features_dir(split="test")
    
    # 获取模型目录
    model_dir = pm.get_model_dir(method="lapeigvals", level="sample")
    
    # 获取特定文件路径
    model_path = pm.get_model_path(method="lapeigvals", level="sample")
"""
from __future__ import annotations
import re
import logging
from pathlib import Path
from typing import Optional, List, Any, Union
from dataclasses import dataclass

from omegaconf import DictConfig, ListConfig

logger = logging.getLogger(__name__)


def parse_task_types(task_types: Any) -> Optional[List[str]]:
    """解析 task_types 配置为列表。
    
    支持多种输入格式：
    - None -> None
    - [] -> None
    - "qa" -> ["qa"]
    - ["qa", "summary"] -> ["qa", "summary"]
    - "[qa, summary]" -> ["qa", "summary"]
    
    Args:
        task_types: 任务类型配置（可能是字符串、列表或 None）
        
    Returns:
        任务类型列表或 None
    """
    if task_types is None:
        return None
    
    task_str = str(task_types).strip()
    
    # 空值处理
    if task_str.lower() in ('null', 'none', '[]', ''):
        return None
    
    # 列表类型
    if isinstance(task_types, (list, ListConfig)):
        if len(task_types) == 0:
            return None
        return [str(t).strip().strip("'\"") for t in task_types if str(t).strip()]
    
    # 字符串形式的列表 "[qa, summary]"
    if task_str.startswith('[') and task_str.endswith(']'):
        inner = task_str[1:-1].strip()
        if not inner:
            return None
        parts = re.split(r'[,\s]+', inner)
        return [p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")]
    
    # 单个值
    return [task_str.strip("'\"")]


@dataclass
class PathConfig:
    """路径配置数据类。
    
    存储构建路径所需的所有参数。
    """
    # 基础目录
    base_dir: Path
    features_dir: Path
    models_dir: Path
    results_dir: Path
    
    # 数据集和模型
    dataset_name: str
    model_name: str
    model_short_name: str
    
    # 任务和种子
    task_suffix: str
    seed: int
    
    @classmethod
    def from_config(cls, cfg: DictConfig) -> "PathConfig":
        """从 Hydra 配置创建 PathConfig。
        
        Args:
            cfg: Hydra 配置对象
            
        Returns:
            PathConfig 实例
        """
        # 基础目录
        base_dir = Path(cfg.get("base_dir", "."))
        features_dir = Path(cfg.get("features_dir", "outputs/features"))
        models_dir = Path(cfg.get("models_dir", "outputs/models"))
        results_dir = Path(cfg.get("results_dir", "outputs/results"))
        
        # 数据集名称
        dataset_name = cfg.dataset.name
        
        # 模型名称
        model_name = cfg.model.name
        if hasattr(cfg.model, 'short_name') and cfg.model.short_name:
            model_short_name = cfg.model.short_name
        else:
            model_short_name = model_name.split("/")[-1]
        
        # 任务后缀
        task_suffix = cls._compute_task_suffix(cfg)
        
        # 种子
        seed = cfg.get("seed", 42)
        
        return cls(
            base_dir=base_dir,
            features_dir=features_dir,
            models_dir=models_dir,
            results_dir=results_dir,
            dataset_name=dataset_name,
            model_name=model_name,
            model_short_name=model_short_name,
            task_suffix=task_suffix,
            seed=seed,
        )
    
    @staticmethod
    def _compute_task_suffix(cfg: DictConfig) -> str:
        """计算任务后缀。
        
        优先级：
        1. dataset.task_type (单个任务)
        2. dataset.task_types (多个任务)
        3. "all" (默认)
        """
        # 尝试 task_type
        task_type = cfg.dataset.get('task_type', None)
        if task_type:
            parsed = parse_task_types(task_type)
            if parsed:
                return "_".join(parsed)
        
        # 尝试 task_types
        task_types = cfg.dataset.get('task_types', None)
        parsed = parse_task_types(task_types)
        if parsed:
            return "_".join(parsed)
        
        return "all"


class PathManager:
    """统一的路径管理器。
    
    提供一致的路径构建方法，确保整个项目使用相同的目录结构。
    
    Attributes:
        config: PathConfig 配置对象
    """
    
    def __init__(self, config: PathConfig):
        """
        Args:
            config: PathConfig 配置对象
        """
        self.config = config
    
    @classmethod
    def from_config(cls, cfg: DictConfig) -> "PathManager":
        """从 Hydra 配置创建 PathManager。
        
        Args:
            cfg: Hydra 配置对象
            
        Returns:
            PathManager 实例
        """
        path_config = PathConfig.from_config(cfg)
        return cls(path_config)
    
    # =========================================================================
    # 特征目录
    # =========================================================================
    
    def get_features_dir(self, split: str = "train") -> Path:
        """获取特征目录路径。
        
        Args:
            split: "train" 或 "test"
            
        Returns:
            特征目录路径
            格式: {features_dir}/{dataset}/{model}/seed_{seed}/{task_suffix}[_test]
        """
        task_suffix = self.config.task_suffix
        if split == "test":
            task_suffix = f"{task_suffix}_test"
        
        return (
            self.config.features_dir 
            / self.config.dataset_name 
            / self.config.model_short_name 
            / f"seed_{self.config.seed}" 
            / task_suffix
        )
    
    def get_features_individual_dir(self, split: str = "train") -> Path:
        """获取单个特征文件目录。
        
        Args:
            split: "train" 或 "test"
            
        Returns:
            features_individual 目录路径
        """
        return self.get_features_dir(split) / "features_individual"
    
    def get_consolidated_features_dir(self, split: str = "train") -> Path:
        """获取合并特征目录。
        
        Args:
            split: "train" 或 "test"
            
        Returns:
            features 目录路径
        """
        return self.get_features_dir(split) / "features"
    
    # =========================================================================
    # 模型目录
    # =========================================================================
    
    def get_model_dir(
        self, 
        method: Optional[str] = None, 
        level: str = "sample"
    ) -> Path:
        """获取模型输出目录。
        
        Args:
            method: 方法名称（如果为 None，则不包含方法子目录）
            level: 分类级别 ("sample" 或 "token")
            
        Returns:
            模型目录路径
            格式: {models_dir}/{dataset}/{model}/seed_{seed}/{task}/{method}/{level}
        """
        path = (
            self.config.models_dir 
            / self.config.dataset_name 
            / self.config.model_short_name 
            / f"seed_{self.config.seed}" 
            / self.config.task_suffix
        )
        
        if method:
            path = path / method / level
        
        return path
    
    def get_model_path(
        self, 
        method: str, 
        level: str = "sample"
    ) -> Path:
        """获取模型文件路径。
        
        Args:
            method: 方法名称
            level: 分类级别
            
        Returns:
            model.pkl 文件路径
        """
        return self.get_model_dir(method, level) / "model.pkl"
    
    def get_train_metrics_path(
        self, 
        method: str, 
        level: str = "sample"
    ) -> Path:
        """获取训练指标文件路径。
        
        Args:
            method: 方法名称
            level: 分类级别
            
        Returns:
            train_metrics.json 文件路径
        """
        return self.get_model_dir(method, level) / "train_metrics.json"
    
    def get_eval_results_path(
        self, 
        method: str, 
        level: str = "sample"
    ) -> Path:
        """获取评估结果文件路径。
        
        Args:
            method: 方法名称
            level: 分类级别
            
        Returns:
            eval_results.json 文件路径
        """
        return self.get_model_dir(method, level) / "eval_results.json"
    
    # =========================================================================
    # 结果目录（用于跨任务评估）
    # =========================================================================
    
    def get_results_dir(
        self,
        method: str,
        train_task: Optional[str] = None,
        eval_task: Optional[str] = None,
    ) -> Path:
        """获取结果目录路径（用于跨任务评估）。
        
        Args:
            method: 方法名称
            train_task: 训练任务（默认使用当前任务）
            eval_task: 评估任务（默认使用当前任务）
            
        Returns:
            结果目录路径
            格式: {results_dir}/{dataset}/{model}/seed_{seed}/train_{train}_eval_{eval}/{method}
        """
        train_task = train_task or self.config.task_suffix
        eval_task = eval_task or self.config.task_suffix
        
        return (
            self.config.results_dir 
            / self.config.dataset_name 
            / self.config.model_short_name 
            / f"seed_{self.config.seed}" 
            / f"train_{train_task}_eval_{eval_task}"
            / method
        )
    
    # =========================================================================
    # 便捷方法
    # =========================================================================
    
    def get_metadata_path(self, split: str = "train") -> Path:
        """获取 metadata.json 路径。"""
        return self.get_features_dir(split) / "metadata.json"
    
    def get_answers_path(self, split: str = "train") -> Path:
        """获取 answers.json 路径。"""
        return self.get_features_dir(split) / "answers.json"
    
    def get_labels_path(self, split: str = "train") -> Path:
        """获取 labels.pt 路径。"""
        return self.get_features_dir(split) / "labels.pt"
    
    def get_config_backup_path(self, split: str = "train") -> Path:
        """获取配置备份路径。"""
        return self.get_features_dir(split) / "config.yaml"
    
    # =========================================================================
    # 属性访问
    # =========================================================================
    
    @property
    def dataset_name(self) -> str:
        return self.config.dataset_name
    
    @property
    def model_short_name(self) -> str:
        return self.config.model_short_name
    
    @property
    def task_suffix(self) -> str:
        return self.config.task_suffix
    
    @property
    def seed(self) -> int:
        return self.config.seed
    
    def __repr__(self) -> str:
        return (
            f"PathManager("
            f"dataset={self.config.dataset_name}, "
            f"model={self.config.model_short_name}, "
            f"seed={self.config.seed}, "
            f"task={self.config.task_suffix})"
        )


# =============================================================================
# 便捷函数（向后兼容）
# =============================================================================

def get_task_suffix(cfg: DictConfig) -> str:
    """获取任务后缀（向后兼容函数）。
    
    建议使用 PathManager 替代此函数。
    """
    return PathConfig._compute_task_suffix(cfg)


def get_model_short_name(cfg: DictConfig) -> str:
    """获取模型短名称（向后兼容函数）。
    
    建议使用 PathManager 替代此函数。
    """
    if hasattr(cfg.model, 'short_name') and cfg.model.short_name:
        return cfg.model.short_name
    return cfg.model.name.split("/")[-1]


def get_features_dir(cfg: DictConfig, split: str = "train") -> Path:
    """获取特征目录（向后兼容函数）。
    
    建议使用 PathManager 替代此函数。
    """
    pm = PathManager.from_config(cfg)
    return pm.get_features_dir(split)


def get_output_dir(cfg: DictConfig, method: str, level: str = "sample") -> Path:
    """获取输出目录（向后兼容函数）。
    
    建议使用 PathManager 替代此函数。
    """
    pm = PathManager.from_config(cfg)
    return pm.get_model_dir(method, level)
