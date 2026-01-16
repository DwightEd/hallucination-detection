"""Semantic Entropy Probes (SEPs) for Hallucination Detection.

基于论文: "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs"
论文链接: https://arxiv.org/abs/2406.15927
代码参考: https://github.com/OATML/semantic-entropy-probes

核心思想：
1. Semantic Entropy (SE) 通过对多次生成结果进行语义聚类来估计不确定性
2. SEPs 训练 linear probes 从单次生成的 hidden states 直接预测 SE
3. 这样可以避免多次采样的计算开销，同时保持检测效果

与 Token Entropy 的区别：
- Token Entropy: 直接使用 token 预测概率的熵
- Semantic Entropy: 在语义空间中计算熵（需要 NLI 模型进行语义聚类）
- SEPs: 从 hidden states 预测 semantic entropy（本实现）

SEPs 的优势：
1. 计算效率高：只需单次前向传播，不需要多次采样
2. 泛化能力强：比直接预测准确率的 probes 泛化更好
3. 不需要标注：可以使用自动计算的 SE 作为监督信号

Hidden State 位置选择：
- TBG (Token-Before-Generating): 输入 prompt 最后一个 token 的 hidden state
- SLT (Second-Last Token): 生成回复倒数第二个 token 的 hidden state
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
import logging
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.core import ExtractedFeatures, MethodConfig, METHODS
from .base import BaseMethod

logger = logging.getLogger(__name__)


def extract_hidden_state_at_position(
    hidden_states: torch.Tensor,
    position: str,
    prompt_len: int,
    response_len: int,
) -> np.ndarray:
    """从指定位置提取 hidden state。
    
    Args:
        hidden_states: Hidden states tensor, 可能的形状:
            - [n_layers, seq_len, hidden_dim]
            - [n_layers, hidden_dim] (已经 pooled)
            - [seq_len, hidden_dim] (单层)
        position: 提取位置
            - "tbg": Token-Before-Generating (prompt 最后一个 token)
            - "slt": Second-Last Token (response 倒数第二个 token)
            - "last": 最后一个 token
            - "mean": 平均所有 response tokens
        prompt_len: Prompt 长度
        response_len: Response 长度
        
    Returns:
        Hidden state feature vector [feature_dim]
    """
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.float().cpu().numpy()
    
    # 处理不同形状
    if len(hidden_states.shape) == 2:
        # [seq_len, hidden_dim] 或 [n_layers, hidden_dim]
        if hidden_states.shape[0] < 100:  # 假设层数 < 100
            # 已经是 [n_layers, hidden_dim]，取最后几层的平均
            n_layers = hidden_states.shape[0]
            last_layers = hidden_states[max(0, n_layers - 4):]  # 最后4层
            return last_layers.mean(axis=0)
        else:
            # [seq_len, hidden_dim]
            hidden_states = hidden_states[np.newaxis, ...]  # -> [1, seq_len, hidden_dim]
    
    if len(hidden_states.shape) == 3:
        n_layers, seq_len, hidden_dim = hidden_states.shape
        
        # 选择要使用的层（使用最后几层）
        n_use_layers = min(4, n_layers)
        layer_indices = list(range(n_layers - n_use_layers, n_layers))
        
        if position == "tbg":
            # Token-Before-Generating: prompt 最后一个 token
            pos = max(0, prompt_len - 1)
            pos = min(pos, seq_len - 1)
            selected = hidden_states[layer_indices, pos, :]  # [n_use_layers, hidden_dim]
            
        elif position == "slt":
            # Second-Last Token: response 倒数第二个
            total_len = prompt_len + response_len
            pos = max(0, min(total_len - 2, seq_len - 1))
            selected = hidden_states[layer_indices, pos, :]
            
        elif position == "last":
            # 最后一个 token
            total_len = min(prompt_len + response_len, seq_len)
            pos = max(0, total_len - 1)
            selected = hidden_states[layer_indices, pos, :]
            
        elif position == "mean":
            # Response tokens 的平均
            resp_start = min(prompt_len, seq_len)
            resp_end = min(prompt_len + response_len, seq_len)
            if resp_end <= resp_start:
                resp_start = 0
                resp_end = seq_len
            selected = hidden_states[layer_indices, resp_start:resp_end, :].mean(axis=1)
            
        else:
            raise ValueError(f"Unknown position: {position}")
        
        # 将多层特征拼接或平均
        return selected.flatten()  # [n_use_layers * hidden_dim]
    
    raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")


def compute_sep_features(
    hidden_states: torch.Tensor,
    prompt_len: int,
    response_len: int,
    positions: List[str] = None,
    use_layer_stats: bool = True,
) -> np.ndarray:
    """计算 SEP 特征。
    
    Args:
        hidden_states: Hidden states tensor
        prompt_len: Prompt 长度
        response_len: Response 长度
        positions: 要提取的位置列表，默认 ["tbg", "slt"]
        use_layer_stats: 是否添加层间统计特征
        
    Returns:
        Feature vector for SEP
    """
    if positions is None:
        positions = ["tbg", "slt"]
    
    features = []
    
    # 从不同位置提取 hidden states
    for pos in positions:
        try:
            feat = extract_hidden_state_at_position(
                hidden_states, pos, prompt_len, response_len
            )
            features.append(feat)
        except Exception as e:
            logger.warning(f"Failed to extract hidden state at {pos}: {e}")
    
    if len(features) == 0:
        raise ValueError("No valid hidden states extracted")
    
    # 如果需要层间统计特征
    if use_layer_stats and isinstance(hidden_states, (torch.Tensor, np.ndarray)):
        hs = hidden_states
        if isinstance(hs, torch.Tensor):
            hs = hs.float().cpu().numpy()
        
        if len(hs.shape) == 3:
            n_layers = hs.shape[0]
            # 计算层间的 norm 变化
            layer_norms = [np.linalg.norm(hs[l]) for l in range(n_layers)]
            if len(layer_norms) > 1:
                features.append(np.array([
                    np.mean(layer_norms),
                    np.std(layer_norms),
                    np.max(layer_norms) - np.min(layer_norms),  # range
                    layer_norms[-1] - layer_norms[0],  # trend
                ], dtype=np.float32))
    
    return np.concatenate(features)


@METHODS.register("semantic_entropy_probes", aliases=["sep", "seps"])
class SemanticEntropyProbesMethod(BaseMethod):
    """Semantic Entropy Probes (SEPs) 幻觉检测方法。
    
    基于论文: "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs"
    
    核心思想：
    - 训练 linear probes 从 hidden states 预测 semantic entropy
    - 使用 logistic regression 作为分类器
    - 从特定位置（TBG, SLT）提取 hidden states
    
    与 Token Entropy 的区别：
    - Token Entropy 直接使用 token 概率的熵
    - SEPs 从 hidden states 学习预测语义级别的不确定性
    
    参数配置（在 config.params 中）：
    - positions: List[str] - 提取 hidden states 的位置
        - "tbg": Token-Before-Generating (推荐)
        - "slt": Second-Last Token (推荐)
        - "last": 最后一个 token
        - "mean": Response 平均
    - use_layer_stats: bool - 是否使用层间统计特征
    - layer_selection: str - 层选择策略
        - "last_4": 最后4层 (默认)
        - "all": 所有层
    """
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        
        params = self.config.params or {}
        
        # Hidden state 提取位置
        self.positions = params.get("positions", ["tbg", "slt"])
        
        # 是否使用层间统计特征
        self.use_layer_stats = params.get("use_layer_stats", True)
        
        # 层选择
        self.layer_selection = params.get("layer_selection", "last_4")
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """从 ExtractedFeatures 提取 SEP 特征。
        
        Args:
            features: 提取的特征
            
        Returns:
            SEP 特征向量
        """
        # 使用懒加载获取 hidden_states
        hidden_states = features.hidden_states
        if hidden_states is None:
            hidden_states = features.get_hidden_states()
        
        if hidden_states is None:
            raise ValueError("SemanticEntropyProbes requires hidden_states. "
                           "Make sure to set hidden_states: true in feature config.")
        
        return compute_sep_features(
            hidden_states=hidden_states,
            prompt_len=features.prompt_len,
            response_len=features.response_len,
            positions=self.positions,
            use_layer_stats=self.use_layer_stats,
        )
    
    def extract_token_features(self, features: ExtractedFeatures, token_idx: int) -> Optional[np.ndarray]:
        """Extract hidden state features for a single token.
        
        对于SEP，每个token的特征是该位置的hidden state。
        
        Args:
            features: Extracted features from model
            token_idx: Token index in the response (0-indexed from response start)
            
        Returns:
            Feature vector for this token
        """
        # 使用懒加载获取 hidden_states
        hidden_states = features.hidden_states
        if hidden_states is None:
            hidden_states = features.get_hidden_states()
        
        if hidden_states is None:
            return None
        
        # 转换为numpy
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.cpu().float().numpy()
        
        # hidden_states shape: [n_layers, seq_len, hidden_dim]
        if len(hidden_states.shape) != 3:
            return None
        
        n_layers, seq_len, hidden_dim = hidden_states.shape
        abs_idx = features.prompt_len + token_idx
        
        if abs_idx >= seq_len:
            return None
        
        # 层选择
        if self.layer_selection == "last_4":
            start_layer = max(0, n_layers - 4)
        else:
            start_layer = 0
        
        # 提取该位置的hidden states
        token_hidden = hidden_states[start_layer:, abs_idx, :]  # [selected_layers, hidden_dim]
        
        feat_list = [token_hidden.flatten()]
        
        # 添加层间统计特征
        if self.use_layer_stats:
            feat_list.append(token_hidden.mean(axis=0))  # 层平均
            feat_list.append(token_hidden.std(axis=0))   # 层标准差
        
        result = np.concatenate(feat_list).astype(np.float32)
        
        # Handle NaN/Inf
        if np.any(~np.isfinite(result)):
            result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return result


@METHODS.register("hidden_state_probe", aliases=["hs_probe", "activation_probe"])
class HiddenStateProbeMethod(BaseMethod):
    """基于 Hidden State 的通用 Probe 方法。
    
    这是一个更通用的 probe 方法，可以配置不同的特征提取策略。
    SEPs 是其特例，专注于预测 semantic entropy。
    
    参数配置（在 config.params 中）：
    - pooling: str - hidden states 的池化方式
        - "last": 最后一个 token
        - "mean": 平均池化
        - "max": 最大池化
        - "cls": 第一个 token（类似 BERT）
    - layer_selection: str - 层选择
        - "last": 最后一层
        - "last_4": 最后4层
        - "all": 所有层
        - "specific": 使用 specific_layers 指定
    - specific_layers: List[int] - 当 layer_selection="specific" 时使用
    """
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        
        params = self.config.params or {}
        self.pooling = params.get("pooling", "mean")
        self.layer_selection = params.get("layer_selection", "last_4")
        self.specific_layers = params.get("specific_layers", [-4, -3, -2, -1])
    
    def _select_layers(self, hidden_states: np.ndarray) -> np.ndarray:
        """选择要使用的层。"""
        n_layers = hidden_states.shape[0]
        
        if self.layer_selection == "last":
            return hidden_states[-1:]
        elif self.layer_selection == "last_4":
            return hidden_states[-4:]
        elif self.layer_selection == "all":
            return hidden_states
        elif self.layer_selection == "specific":
            indices = [l if l >= 0 else n_layers + l for l in self.specific_layers]
            indices = [i for i in indices if 0 <= i < n_layers]
            return hidden_states[indices]
        else:
            return hidden_states[-4:]
    
    def _pool_sequence(self, hidden_states: np.ndarray, prompt_len: int, response_len: int) -> np.ndarray:
        """对序列维度进行池化。
        
        Args:
            hidden_states: [n_layers, seq_len, hidden_dim]
        """
        n_layers, seq_len, hidden_dim = hidden_states.shape
        
        # 只关注 response 部分
        resp_start = min(prompt_len, seq_len)
        resp_end = min(prompt_len + response_len, seq_len)
        if resp_end <= resp_start:
            resp_start = 0
            resp_end = seq_len
        
        response_hs = hidden_states[:, resp_start:resp_end, :]  # [n_layers, resp_len, hidden_dim]
        
        if self.pooling == "last":
            return response_hs[:, -1, :]  # [n_layers, hidden_dim]
        elif self.pooling == "mean":
            return response_hs.mean(axis=1)
        elif self.pooling == "max":
            return response_hs.max(axis=1)
        elif self.pooling == "cls":
            return hidden_states[:, 0, :]  # 第一个 token
        else:
            return response_hs.mean(axis=1)
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """提取 hidden state 特征。"""
        # 使用懒加载获取 hidden_states
        hs = features.hidden_states
        if hs is None:
            hs = features.get_hidden_states()
        
        if hs is None:
            raise ValueError("HiddenStateProbe requires hidden_states")
        
        if isinstance(hs, torch.Tensor):
            hs = hs.float().cpu().numpy()
        
        # 处理不同形状
        if len(hs.shape) == 2:
            # [seq_len, hidden_dim] 或 [n_layers, hidden_dim]
            if hs.shape[0] < 100:  # 假设是 [n_layers, hidden_dim]
                return hs.flatten()
            else:
                # [seq_len, hidden_dim] - 单层
                hs = hs[np.newaxis, ...]
        
        # [n_layers, seq_len, hidden_dim]
        hs = self._select_layers(hs)
        pooled = self._pool_sequence(hs, features.prompt_len, features.response_len)
        
        return pooled.flatten()