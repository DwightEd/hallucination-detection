"""
ACT - LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations

基于隐藏状态探测的幻觉检测方法
论文: https://arxiv.org/abs/2410.02707
代码: https://github.com/technion-cs-nlp/LLMsKnow

核心思想：
1. LLM在处理正确答案和幻觉答案时，隐藏状态表现不同
2. 从特定层（如中间层13-15）提取隐藏状态
3. 使用exact_answer最后一个token位置的隐藏状态
4. 训练逻辑回归探针进行幻觉检测

关键特点：
- 需要exact_answer字段来定位答案最后token
- 支持多种探测位置：mlp, attention, residual等
- 使用特定层的隐藏状态（论文推荐中间层效果最好）

数据要求：
- 输入数据需要包含'exact_answer'字段（在metadata中）
- exact_answer是模型生成的精确答案文本
"""

import logging
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch

from src.core import ExtractedFeatures, MethodConfig, METHODS
from .base import BaseMethod

logger = logging.getLogger(__name__)


class ACTDetector:
    """
    ACT (Activation Contrast for Truthfulness) 检测器
    
    从隐藏状态中提取特征进行幻觉检测
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化检测器
        
        Args:
            config: 方法配置
        """
        self.config = config
        
        # 层选择配置
        self.layers = config.get('layers', [24, 28, 32])  # 默认使用24, 28, 32层
        self.layer_selection = config.get('layer_selection', 'specific')  # specific, last, middle
        
        # Token位置配置
        self.token_position = config.get('token_position', 'exact_answer_last')
        # 选项: exact_answer_last, response_last, response_mean
        
        # 探测位置配置 (在原始论文中，可以选择mlp/attention/residual)
        # 由于框架限制，我们主要使用hidden_states
        self.probe_location = config.get('probe_location', 'hidden_states')
        
        # 特征处理配置
        self.normalize = config.get('normalize', True)
        self.center = config.get('center', True)
        
        # 统计量
        self.mean_hidden = None
        self.std_hidden = None
    
    def _select_layers(self, num_layers: int) -> List[int]:
        """
        选择要使用的层
        
        Args:
            num_layers: 模型总层数
            
        Returns:
            层索引列表
        """
        if self.layer_selection == 'specific':
            # 过滤掉超出范围的层
            return [l for l in self.layers if 0 <= l < num_layers]
        elif self.layer_selection == 'last':
            return [num_layers - 1]
        elif self.layer_selection == 'middle':
            mid = num_layers // 2
            return [mid - 1, mid, mid + 1]
        elif self.layer_selection == 'last_quarter':
            start = num_layers * 3 // 4
            return list(range(start, num_layers))
        else:
            return self.layers
    
    def _find_exact_answer_position(
        self,
        features: ExtractedFeatures,
        tokenizer=None,
    ) -> int:
        """
        找到exact_answer最后一个token的位置
        
        Args:
            features: 提取的特征
            tokenizer: 分词器（可选）
            
        Returns:
            token位置索引（从序列开始计算的绝对位置）
        """
        # 检查metadata中是否有exact_answer
        exact_answer = features.metadata.get('exact_answer')
        
        if exact_answer is not None and tokenizer is not None:
            # 如果有exact_answer，尝试定位其最后一个token
            # 这需要知道完整的输入序列和分词器
            # 在实际使用中，这个位置可能已经预计算好了
            exact_answer_token_idx = features.metadata.get('exact_answer_last_token_idx')
            if exact_answer_token_idx is not None:
                return exact_answer_token_idx
        
        # 检查是否有预计算的exact_answer位置
        if 'exact_answer_position' in features.metadata:
            return features.metadata['exact_answer_position']
        
        # 默认：使用response的最后一个token位置
        return features.prompt_len + features.response_len - 1
    
    def extract_features(
        self,
        features: ExtractedFeatures,
        tokenizer=None,
    ) -> np.ndarray:
        """
        从ExtractedFeatures中提取ACT特征
        
        Args:
            features: 提取的特征
            tokenizer: 分词器（可选，用于定位exact_answer）
            
        Returns:
            特征向量 [feature_dim]
        """
        hidden_states = features.hidden_states
        
        if hidden_states is None:
            # 尝试延迟加载
            hidden_states = features.get_hidden_states()
        
        if hidden_states is None:
            raise ValueError(f"Sample {features.sample_id} has no hidden_states")
        
        # 转换为numpy
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.cpu().float().numpy()
        
        # Handle NaN/Inf
        if np.any(~np.isfinite(hidden_states)):
            hidden_states = np.nan_to_num(hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # hidden_states形状: [num_layers, seq_len, hidden_dim] 或 [num_layers, hidden_dim]（已聚合）
        if len(hidden_states.shape) == 2:
            # 已经聚合，可能是 [num_layers, hidden_dim]
            num_layers = hidden_states.shape[0]
            selected_layers = self._select_layers(num_layers)
            
            # 选择特定层
            selected_hidden = hidden_states[selected_layers]  # [num_selected_layers, hidden_dim]
        else:
            # [num_layers, seq_len, hidden_dim]
            num_layers = hidden_states.shape[0]
            selected_layers = self._select_layers(num_layers)
            
            # 根据token_position选择位置
            if self.token_position == 'exact_answer_last':
                token_idx = self._find_exact_answer_position(features)
                # 确保索引在范围内
                token_idx = min(token_idx, hidden_states.shape[1] - 1)
                token_idx = max(token_idx, 0)
                selected_hidden = hidden_states[selected_layers, token_idx, :]  # [num_selected_layers, hidden_dim]
            
            elif self.token_position == 'response_last':
                # 使用response的最后一个token
                token_idx = min(features.prompt_len + features.response_len - 1, 
                               hidden_states.shape[1] - 1)
                selected_hidden = hidden_states[selected_layers, token_idx, :]
            
            elif self.token_position == 'response_mean':
                # 对response部分取平均
                start_idx = features.prompt_len
                end_idx = min(features.prompt_len + features.response_len, hidden_states.shape[1])
                if end_idx > start_idx:
                    selected_hidden = hidden_states[selected_layers, start_idx:end_idx, :].mean(axis=1)
                else:
                    selected_hidden = hidden_states[selected_layers, -1, :]
            
            else:
                # 默认使用最后一个token
                selected_hidden = hidden_states[selected_layers, -1, :]
        
        # 展平特征
        feat_vec = selected_hidden.flatten()
        
        # 标准化
        if self.center and self.mean_hidden is not None:
            feat_vec = feat_vec - self.mean_hidden
        if self.normalize and self.std_hidden is not None:
            feat_vec = feat_vec / (self.std_hidden + 1e-8)
        
        # Handle NaN/Inf in final features
        if np.any(~np.isfinite(feat_vec)):
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feat_vec.astype(np.float32)
    
    def fit(self, features_list: List[ExtractedFeatures]):
        """
        使用数据拟合统计量（用于标准化）
        
        Args:
            features_list: 特征列表
        """
        all_features = []
        
        for feat in features_list:
            try:
                vec = self.extract_features(feat)
                all_features.append(vec)
            except Exception as e:
                logger.warning(f"Failed to extract features for {feat.sample_id}: {e}")
        
        if not all_features:
            logger.warning("No features extracted for fitting")
            return
        
        all_features = np.array(all_features)
        
        # 计算统计量
        self.mean_hidden = all_features.mean(axis=0)
        self.std_hidden = all_features.std(axis=0)
        
        # 避免除零
        self.std_hidden = np.where(self.std_hidden < 1e-8, 1.0, self.std_hidden)
        
        logger.info(f"ACT detector fitted on {len(all_features)} samples")


@METHODS.register("act", aliases=["llmsknow", "act_probe", "intrinsic_probe"])
class ACTMethod(BaseMethod):
    """
    ACT方法的BaseMethod兼容包装类
    
    基于论文 "LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations"
    
    使用特定层的隐藏状态在exact_answer最后token位置进行探测
    
    参数:
        layers: 使用的层索引列表，默认 [24, 28, 32]
        token_position: token位置选择策略
            - exact_answer_last: exact_answer的最后一个token (推荐)
            - response_last: response的最后一个token
            - response_mean: response所有token的平均
        normalize: 是否标准化特征
        center: 是否中心化特征
    
    数据要求:
        - hidden_states: 必需
        - metadata中的exact_answer: 推荐（用于定位精确答案）
    """
    
    def __init__(self, config: Optional[MethodConfig] = None):
        """初始化方法"""
        super().__init__(config)
        
        # 从config.params获取参数
        params = self.config.params if self.config else {}
        
        # 创建底层检测器配置
        detector_config = {
            'layers': params.get('layers', [24, 28, 32]),
            'layer_selection': params.get('layer_selection', 'specific'),
            'token_position': params.get('token_position', 'exact_answer_last'),
            'probe_location': params.get('probe_location', 'hidden_states'),
            'normalize': params.get('normalize', True),
            'center': params.get('center', True),
        }
        
        self.detector = ACTDetector(detector_config)
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """从ExtractedFeatures提取方法特定的特征向量
        
        Args:
            features: 提取的特征
            
        Returns:
            特征向量 [feature_dim]
        """
        return self.detector.extract_features(features)
    
    def extract_token_features(self, features: ExtractedFeatures, token_idx: int) -> Optional[np.ndarray]:
        """提取单个token的特征
        
        对于ACT方法，我们为每个token提取其位置的隐藏状态
        
        Args:
            features: 提取的特征
            token_idx: token索引（相对于response开始）
            
        Returns:
            token特征向量
        """
        hidden_states = features.hidden_states
        
        if hidden_states is None:
            hidden_states = features.get_hidden_states()
        
        if hidden_states is None:
            return None
        
        # 转换为numpy
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.cpu().float().numpy()
        
        if len(hidden_states.shape) != 3:
            return None
        
        num_layers = hidden_states.shape[0]
        selected_layers = self.detector._select_layers(num_layers)
        
        # 计算绝对位置
        abs_idx = features.prompt_len + token_idx
        
        if abs_idx >= hidden_states.shape[1]:
            return None
        
        # 提取特定位置的隐藏状态
        token_hidden = hidden_states[selected_layers, abs_idx, :]  # [num_layers, hidden_dim]
        
        feat_vec = token_hidden.flatten().astype(np.float32)
        
        # 标准化
        if self.detector.center and self.detector.mean_hidden is not None:
            feat_vec = feat_vec - self.detector.mean_hidden
        if self.detector.normalize and self.detector.std_hidden is not None:
            feat_vec = feat_vec / (self.detector.std_hidden + 1e-8)
        
        # Handle NaN/Inf
        if np.any(~np.isfinite(feat_vec)):
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feat_vec
    
    def fit(
        self,
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
        cv: bool = True,
    ) -> Dict[str, Any]:
        """训练方法
        
        重写父类方法，先拟合检测器（用于标准化），再训练分类器
        """
        # 先拟合检测器
        self.detector.fit(features_list)
        
        # 然后调用父类的fit进行监督学习
        return super().fit(features_list, labels, cv)
    
    def save(self, path: Path) -> None:
        """保存方法"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": self.config,
            # Sample-level
            "classifier": self.classifier,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
            "feature_dim": self._feature_dim,
            # Token-level
            "token_classifier": self.token_classifier,
            "token_scaler": self.token_scaler,
            "is_token_fitted": self.is_token_fitted,
            "token_feature_dim": self._token_feature_dim,
            # Detector state
            "detector": {
                "config": self.detector.config,
                "mean_hidden": self.detector.mean_hidden,
                "std_hidden": self.detector.std_hidden,
            }
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved ACT method to {path}")
    
    def load(self, path: Path) -> None:
        """加载方法"""
        path = Path(path)
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.config = state["config"]
        
        # Sample-level
        self.classifier = state.get("classifier")
        self.scaler = state.get("scaler")
        self.is_fitted = state.get("is_fitted", False)
        self._feature_dim = state.get("feature_dim")
        
        # Token-level
        self.token_classifier = state.get("token_classifier")
        self.token_scaler = state.get("token_scaler")
        self.is_token_fitted = state.get("is_token_fitted", False)
        self._token_feature_dim = state.get("token_feature_dim")
        
        # Detector state
        if "detector" in state:
            detector_state = state["detector"]
            self.detector = ACTDetector(detector_state.get("config", {}))
            self.detector.mean_hidden = detector_state.get("mean_hidden")
            self.detector.std_hidden = detector_state.get("std_hidden")
        
        logger.info(f"Loaded ACT method from {path}")


@METHODS.register("act_layers", aliases=["act_multi_layer"])
class ACTMultiLayerMethod(ACTMethod):
    """
    ACT多层探测方法
    
    同时使用多个层的隐藏状态进行探测
    特别适合探索不同层的幻觉检测能力
    """
    
    def __init__(self, config: Optional[MethodConfig] = None):
        """初始化"""
        # 确保使用多层
        if config and config.params:
            if 'layers' not in config.params:
                config.params['layers'] = [24, 28, 32]
        
        super().__init__(config)


@METHODS.register("act_exact_answer", aliases=["act_ea"])
class ACTExactAnswerMethod(ACTMethod):
    """
    ACT精确答案探测方法
    
    专门使用exact_answer最后token位置的隐藏状态
    这是论文中推荐的最佳配置
    
    注意: 需要数据中包含exact_answer字段
    """
    
    def __init__(self, config: Optional[MethodConfig] = None):
        """初始化"""
        # 确保使用exact_answer_last
        if config and config.params:
            config.params['token_position'] = 'exact_answer_last'
        elif config:
            config.params = {'token_position': 'exact_answer_last'}
        
        super().__init__(config)
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """提取特征，检查exact_answer字段"""
        # 检查是否有exact_answer
        if 'exact_answer' not in features.metadata:
            logger.warning(
                f"Sample {features.sample_id} missing 'exact_answer' in metadata. "
                "Falling back to response_last token position."
            )
        
        return super().extract_method_features(features)
