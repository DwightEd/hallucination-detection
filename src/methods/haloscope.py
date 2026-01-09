"""
HaloScope - Harnessing Unlabeled LLM Generations for Hallucination Detection

基于隐藏状态SVD的无监督幻觉检测方法
论文: https://arxiv.org/abs/2409.17504 (NeurIPS'24)

核心思想：
1. 提取LLM隐藏状态
2. 对隐藏状态进行SVD分解
3. 使用主成分方向上的投影作为幻觉分数
4. 无需标注数据即可检测幻觉
"""

import logging
import pickle
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class HaloScopeDetector:
    """
    HaloScope检测器
    
    无监督幻觉检测方法，基于隐藏状态的SVD分析
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化检测器
        
        Args:
            config: 方法配置
        """
        self.config = config
        self.detection_cfg = config.get('detection', {})
        self.feature_cfg = config.get('feature_config', {})
        
        # SVD相关参数
        self.weighted_svd = self.detection_cfg.get('weighted_svd', True)
        self.feat_loc_svd = self.detection_cfg.get('feat_loc_svd', 3)
        self.layer_selection = self.detection_cfg.get('layer_selection', 'last_half')
        self.specific_layers = self.detection_cfg.get('specific_layers', [])
        
        # SVD配置
        svd_cfg = config.get('svd_config', {})
        self.n_components = svd_cfg.get('n_components', 10)
        self.center = svd_cfg.get('center', True)
        self.normalize = svd_cfg.get('normalize', False)
        
        # 分数配置
        self.score_aggregation = self.detection_cfg.get('score_aggregation', 'mean')
        self.percentile_threshold = self.detection_cfg.get('percentile_threshold', 90)
        self.fixed_threshold = self.detection_cfg.get('fixed_threshold', None)
        self.score_direction = self.detection_cfg.get('score_direction', 'higher_is_hallucination')
        
        # 用于存储分布估计
        self.reference_scores = None
        self.mean_hidden = None
        self.std_hidden = None
        self.svd_components = None
        self.singular_values = None
    
    def _select_layers(self, num_layers: int) -> List[int]:
        """
        选择要使用的层
        
        Args:
            num_layers: 总层数
            
        Returns:
            层索引列表
        """
        if self.layer_selection == 'all':
            return list(range(num_layers))
        elif self.layer_selection == 'last_half':
            return list(range(num_layers // 2, num_layers))
        elif self.layer_selection == 'specific':
            return self.specific_layers
        else:
            logger.warning(f"Unknown layer_selection: {self.layer_selection}, using all layers")
            return list(range(num_layers))
    
    def _extract_hidden_states(self, data: Dict[str, Any]) -> np.ndarray:
        """
        从数据中提取隐藏状态
        
        Args:
            data: 包含hidden_states的数据字典
            
        Returns:
            [num_selected_layers, seq_len, hidden_dim] 或聚合后的形状
        """
        hidden_states = data.get('hidden_states')
        
        if hidden_states is None:
            raise ValueError("hidden_states not found in data")
        
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.cpu().float().numpy()
        
        # Handle NaN and Inf values
        if np.any(~np.isfinite(hidden_states)):
            hidden_states = np.nan_to_num(hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # hidden_states形状可能是:
        # [num_layers, seq_len, hidden_dim] 或
        # [num_layers+1, seq_len, hidden_dim] (包含embedding层)
        
        if len(hidden_states.shape) == 2:
            # 单层: [seq_len, hidden_dim]
            hidden_states = hidden_states[np.newaxis, ...]
        
        num_layers = hidden_states.shape[0]
        
        # 选择层
        selected_layers = self._select_layers(num_layers)
        selected_hidden = hidden_states[selected_layers]
        
        # 如果只使用response部分
        if self.feature_cfg.get('response_only', True):
            # 假设数据中有response_start标记
            response_start = data.get('response_start', 0)
            if response_start > 0:
                selected_hidden = selected_hidden[:, response_start:, :]
        
        # Token聚合
        if self.feature_cfg.get('aggregate_tokens', True):
            agg_method = self.feature_cfg.get('token_aggregation', 'mean')
            if agg_method == 'mean':
                selected_hidden = selected_hidden.mean(axis=1)  # [num_layers, hidden_dim]
            elif agg_method == 'max':
                selected_hidden = selected_hidden.max(axis=1)
            elif agg_method == 'last':
                selected_hidden = selected_hidden[:, -1, :]
        
        return selected_hidden
    
    def _compute_svd_score(self, hidden: np.ndarray) -> float:
        """
        计算SVD分数
        
        Args:
            hidden: 隐藏状态 [num_layers, hidden_dim] 或 [hidden_dim]
            
        Returns:
            幻觉分数
        """
        if len(hidden.shape) == 1:
            hidden = hidden[np.newaxis, :]
        
        # Handle NaN/Inf
        if np.any(~np.isfinite(hidden)):
            hidden = np.nan_to_num(hidden, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 标准化
        if self.center and self.mean_hidden is not None:
            hidden = hidden - self.mean_hidden
        if self.normalize and self.std_hidden is not None:
            hidden = hidden / (self.std_hidden + 1e-8)
        
        # 展平
        hidden_flat = hidden.flatten()
        
        if self.svd_components is None:
            # 如果没有预计算的SVD，直接使用hidden的norm
            norm = np.linalg.norm(hidden_flat)
            return float(norm) if np.isfinite(norm) else 0.0
        
        # 投影到主成分
        projections = np.dot(hidden_flat, self.svd_components.T)
        
        # Handle NaN in projections
        if np.any(~np.isfinite(projections)):
            projections = np.nan_to_num(projections, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 计算分数
        if self.weighted_svd and self.singular_values is not None:
            # 按奇异值加权
            weights = self.singular_values[:len(projections)]
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                weights = np.ones_like(weights) / len(weights)
            score = np.sum(np.abs(projections) * weights)
        else:
            score = np.mean(np.abs(projections))
        
        return float(score) if np.isfinite(score) else 0.0
    
    def fit(self, unlabeled_data: List[Dict[str, Any]]):
        """
        使用unlabeled数据估计分布（可选）
        
        Args:
            unlabeled_data: 无标签数据列表
        """
        logger.info(f"Fitting HaloScope on {len(unlabeled_data)} samples")
        
        # 提取所有隐藏状态
        all_hidden = []
        for item in unlabeled_data:
            try:
                hidden = self._extract_hidden_states(item)
                all_hidden.append(hidden.flatten())
            except Exception as e:
                logger.warning(f"Failed to extract hidden states: {e}")
        
        if not all_hidden:
            logger.warning("No hidden states extracted, using default parameters")
            return
        
        all_hidden = np.array(all_hidden)
        
        # Handle NaN/Inf in all_hidden
        if np.any(~np.isfinite(all_hidden)):
            logger.warning("Found NaN/Inf in hidden states, replacing with zeros")
            all_hidden = np.nan_to_num(all_hidden, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 计算统计量
        self.mean_hidden = all_hidden.mean(axis=0)
        self.std_hidden = all_hidden.std(axis=0)
        
        # Avoid division by zero in std
        self.std_hidden = np.where(self.std_hidden < 1e-8, 1.0, self.std_hidden)
        
        # SVD
        centered = all_hidden - self.mean_hidden
        if self.normalize:
            centered = centered / (self.std_hidden + 1e-8)
        
        try:
            # Use truncated SVD for better numerical stability
            from sklearn.decomposition import TruncatedSVD
            n_components = min(self.n_components, centered.shape[0] - 1, centered.shape[1])
            if n_components > 0:
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                svd.fit(centered)
                self.svd_components = svd.components_
                self.singular_values = svd.singular_values_
                logger.info(f"SVD completed, top singular values: {self.singular_values[:5]}")
            else:
                logger.warning("Not enough samples for SVD, using fallback")
                self.svd_components = None
                self.singular_values = None
        except Exception as e:
            logger.warning(f"SVD failed: {e}, using fallback mode")
            self.svd_components = None
            self.singular_values = None
        
        # 计算参考分数分布
        self.reference_scores = []
        for hidden_flat in all_hidden:
            score = self._compute_svd_score(hidden_flat.reshape(-1))
            self.reference_scores.append(score)
        
        self.reference_scores = np.array(self.reference_scores)
        
        # Handle NaN in reference scores
        valid_scores = self.reference_scores[np.isfinite(self.reference_scores)]
        if len(valid_scores) > 0:
            logger.info(f"Reference scores: mean={valid_scores.mean():.4f}, std={valid_scores.std():.4f}")
        else:
            logger.warning("All reference scores are NaN, using uniform distribution")
            self.reference_scores = np.linspace(0, 1, len(all_hidden))
    
    def predict_score(self, data: Dict[str, Any]) -> float:
        """
        预测单个样本的幻觉分数
        
        Args:
            data: 包含hidden_states的数据
            
        Returns:
            幻觉分数 (0-1之间，越高越可能是幻觉)
        """
        hidden = self._extract_hidden_states(data)
        score = self._compute_svd_score(hidden)
        
        # 归一化到0-1
        if self.reference_scores is not None and len(self.reference_scores) > 0:
            # 使用参考分布进行归一化
            percentile = (self.reference_scores < score).mean()
            return float(percentile)
        
        return float(score)
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        预测单个样本
        
        Args:
            data: 输入数据
            
        Returns:
            {'label': int, 'proba': float, 'score': float}
        """
        score = self.predict_score(data)
        
        # 确定阈值
        if self.fixed_threshold is not None:
            threshold = self.fixed_threshold
        elif self.reference_scores is not None:
            threshold = np.percentile(self.reference_scores, self.percentile_threshold)
            threshold = (self.reference_scores < threshold).mean()  # 转换为百分位
        else:
            threshold = 0.5
        
        # 判断标签
        if self.score_direction == 'higher_is_hallucination':
            label = int(score > threshold)
        else:
            label = int(score < threshold)
        
        return {
            'label': label,
            'proba': score if self.score_direction == 'higher_is_hallucination' else 1 - score,
            'score': score
        }
    
    def predict_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            data_list: 数据列表
            
        Returns:
            预测结果列表
        """
        return [self.predict(data) for data in data_list]
    
    def save(self, path: Path):
        """保存检测器状态"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        state = {
            'config': self.config,
            'mean_hidden': self.mean_hidden,
            'std_hidden': self.std_hidden,
            'svd_components': self.svd_components,
            'singular_values': self.singular_values,
            'reference_scores': self.reference_scores
        }
        
        np.savez(path / 'haloscope_state.npz', **{
            k: v for k, v in state.items() if v is not None and k != 'config'
        })
        
        import json
        with open(path / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"HaloScope state saved to {path}")
    
    def load(self, path: Path):
        """加载检测器状态"""
        path = Path(path)
        
        import json
        with open(path / 'config.json', 'r') as f:
            self.config = json.load(f)
        
        # 重新初始化配置
        self.__init__(self.config)
        
        # 加载状态
        state_file = path / 'haloscope_state.npz'
        if state_file.exists():
            state = np.load(state_file, allow_pickle=True)
            self.mean_hidden = state.get('mean_hidden')
            self.std_hidden = state.get('std_hidden')
            self.svd_components = state.get('svd_components')
            self.singular_values = state.get('singular_values')
            self.reference_scores = state.get('reference_scores')
        
        logger.info(f"HaloScope state loaded from {path}")


def compute_haloscope_score(
    hidden_states: np.ndarray,
    layer_selection: str = 'last_half',
    weighted_svd: bool = True
) -> float:
    """
    便捷函数：直接计算HaloScope分数
    
    Args:
        hidden_states: [num_layers, seq_len, hidden_dim]
        layer_selection: 层选择策略
        weighted_svd: 是否使用加权SVD
        
    Returns:
        幻觉分数
    """
    config = {
        'detection': {
            'layer_selection': layer_selection,
            'weighted_svd': weighted_svd,
            'feat_loc_svd': 3
        },
        'feature_config': {
            'aggregate_tokens': True,
            'token_aggregation': 'mean'
        }
    }
    
    detector = HaloScopeDetector(config)
    data = {'hidden_states': hidden_states}
    
    return detector.predict_score(data)


# =============================================================================
# HaloScopeMethod - BaseMethod 兼容的包装类
# =============================================================================

from src.core import ExtractedFeatures, MethodConfig, METHODS
from .base import BaseMethod


@METHODS.register("haloscope", aliases=["halo", "haloscope_svd"])
class HaloScopeMethod(BaseMethod):
    """
    HaloScope方法的BaseMethod兼容包装类
    
    使用隐藏状态的SVD分析检测幻觉
    """
    
    def __init__(self, config: Optional[MethodConfig] = None):
        """初始化方法"""
        super().__init__(config)
        
        # 创建底层检测器配置
        detector_config = {
            'detection': {
                'layer_selection': 'last_half',
                'weighted_svd': True,
                'feat_loc_svd': 3,
            },
            'feature_config': {
                'aggregate_tokens': True,
                'token_aggregation': 'mean',
                'response_only': True
            },
            'svd_config': {
                'n_components': 10,
                'center': True,
                'normalize': False
            }
        }
        
        # 如果config中有params，覆盖默认值
        if config and config.params:
            for key, value in config.params.items():
                if key in detector_config:
                    if isinstance(value, dict):
                        detector_config[key].update(value)
                    else:
                        detector_config[key] = value
        
        self.detector = HaloScopeDetector(detector_config)
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """从ExtractedFeatures提取方法特定的特征向量
        
        Args:
            features: 提取的特征
            
        Returns:
            特征向量 [feature_dim]
        """
        hidden_states = features.hidden_states
        
        if hidden_states is None:
            raise ValueError(f"Sample {features.sample_id} has no hidden_states")
        
        # 转换为numpy
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.cpu().float().numpy()
        
        # Handle NaN/Inf in hidden states
        if np.any(~np.isfinite(hidden_states)):
            hidden_states = np.nan_to_num(hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 获取处理后的隐藏状态
        data = {
            'hidden_states': hidden_states,
            'response_start': features.prompt_len if hasattr(features, 'prompt_len') else 0
        }
        
        # 使用检测器的内部方法处理隐藏状态
        processed = self.detector._extract_hidden_states(data)
        
        # 展平作为特征，确保无 NaN
        result = processed.flatten()
        if np.any(~np.isfinite(result)):
            result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return result
    
    def fit(self, features_list: List[ExtractedFeatures], labels: Optional[List[int]] = None, cv: bool = True) -> Dict[str, float]:
        """训练方法
        
        重写父类方法，先用所有数据进行SVD拟合，再训练分类器
        """
        # 提取隐藏状态用于SVD拟合
        all_data = []
        for feat in features_list:
            if feat.hidden_states is not None:
                hs = feat.hidden_states
                if isinstance(hs, torch.Tensor):
                    hs = hs.cpu().float().numpy()
                all_data.append({
                    'hidden_states': hs,
                    'response_start': feat.prompt_len if hasattr(feat, 'prompt_len') else 0
                })
        
        # 先拟合SVD（无监督）
        if all_data:
            self.detector.fit(all_data)
        
        # 然后调用父类的fit进行监督学习
        return super().fit(features_list, labels, cv)
    
    def save(self, path: Path) -> None:
        """保存方法"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": self.config,
            "classifier": self.classifier,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
            "feature_dim": self._feature_dim,
            "detector": {
                "mean_hidden": self.detector.mean_hidden,
                "std_hidden": self.detector.std_hidden,
                "svd_components": self.detector.svd_components,
                "singular_values": self.detector.singular_values,
                "reference_scores": self.detector.reference_scores,
            }
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved HaloScope method to {path}")
    
    def load(self, path: Path) -> None:
        """加载方法"""
        path = Path(path)
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.config = state["config"]
        self.classifier = state["classifier"]
        self.scaler = state["scaler"]
        self.is_fitted = state["is_fitted"]
        self._feature_dim = state["feature_dim"]
        
        if "detector" in state:
            self.detector.mean_hidden = state["detector"]["mean_hidden"]
            self.detector.std_hidden = state["detector"]["std_hidden"]
            self.detector.svd_components = state["detector"]["svd_components"]
            self.detector.singular_values = state["detector"]["singular_values"]
            self.detector.reference_scores = state["detector"].get("reference_scores")
        
        logger.info(f"Loaded HaloScope method from {path}")