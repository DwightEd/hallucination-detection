"""Semantic Entropy Probes (SEPs) for Hallucination Detection.

基于论文: "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs"
论文链接: https://arxiv.org/abs/2406.15927
代码参考: https://github.com/OATML/semantic-entropy-probes

核心思想：
1. Semantic Entropy (SE) 通过对多次生成结果进行语义聚类来估计不确定性
2. SEPs 训练 linear probes 从单次生成的 hidden states 直接预测 SE
3. 这样可以避免多次采样的计算开销，同时保持检测效果

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

from src.core import ExtractedFeatures, MethodConfig, METHODS, Prediction
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
        hidden_states: [n_layers, seq_len, hidden_dim] 或其他形状
        position: 提取位置 (tbg, slt, last, mean)
        prompt_len: Prompt 长度
        response_len: Response 长度
        
    Returns:
        Hidden state feature vector
    """
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.float().cpu().numpy()
    
    if len(hidden_states.shape) == 2:
        if hidden_states.shape[0] < 100:
            n_layers = hidden_states.shape[0]
            last_layers = hidden_states[max(0, n_layers - 4):]
            return last_layers.mean(axis=0)
        else:
            hidden_states = hidden_states[np.newaxis, ...]
    
    if len(hidden_states.shape) == 3:
        n_layers, seq_len, hidden_dim = hidden_states.shape
        
        n_use_layers = min(4, n_layers)
        layer_indices = list(range(n_layers - n_use_layers, n_layers))
        
        if position == "tbg":
            pos = max(0, prompt_len - 1)
            pos = min(pos, seq_len - 1)
            selected = hidden_states[layer_indices, pos, :]
            
        elif position == "slt":
            total_len = prompt_len + response_len
            pos = max(0, min(total_len - 2, seq_len - 1))
            selected = hidden_states[layer_indices, pos, :]
            
        elif position == "last":
            total_len = min(prompt_len + response_len, seq_len)
            pos = max(0, total_len - 1)
            selected = hidden_states[layer_indices, pos, :]
            
        elif position == "mean":
            resp_start = min(prompt_len, seq_len)
            resp_end = min(prompt_len + response_len, seq_len)
            if resp_end <= resp_start:
                resp_start = 0
                resp_end = seq_len
            selected = hidden_states[layer_indices, resp_start:resp_end, :].mean(axis=1)
            
        else:
            raise ValueError(f"Unknown position: {position}")
        
        return selected.flatten()
    
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
    
    if use_layer_stats and isinstance(hidden_states, (torch.Tensor, np.ndarray)):
        hs = hidden_states
        if isinstance(hs, torch.Tensor):
            hs = hs.float().cpu().numpy()
        
        if len(hs.shape) == 3:
            n_layers = hs.shape[0]
            layer_norms = [np.linalg.norm(hs[l]) for l in range(n_layers)]
            if len(layer_norms) > 1:
                features.append(np.array([
                    np.mean(layer_norms),
                    np.std(layer_norms),
                    np.max(layer_norms) - np.min(layer_norms),
                    layer_norms[-1] - layer_norms[0],
                ], dtype=np.float32))
    
    return np.concatenate(features)


@METHODS.register("semantic_entropy_probes", aliases=["sep", "seps"])
class SemanticEntropyProbesMethod(BaseMethod):
    """Semantic Entropy Probes (SEPs) 幻觉检测方法。"""
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        
        params = self.config.params or {}
        self.positions = params.get("positions", ["tbg", "slt"])
        self.use_layer_stats = params.get("use_layer_stats", True)
        self.layer_selection = params.get("layer_selection", "last_4")
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """提取 SEP 特征。"""
        hidden_states = features.hidden_states
        if hidden_states is None:
            hidden_states = features.get_hidden_states()
        
        if hidden_states is None:
            raise ValueError("SemanticEntropyProbes requires hidden_states")
        
        return compute_sep_features(
            hidden_states=hidden_states,
            prompt_len=features.prompt_len,
            response_len=features.response_len,
            positions=self.positions,
            use_layer_stats=self.use_layer_stats,
        )
    
    def extract_token_features(
        self,
        features: ExtractedFeatures,
        token_idx: int
    ) -> Optional[np.ndarray]:
        """为单个 token 提取 hidden state 特征。"""
        hidden_states = features.hidden_states
        if hidden_states is None:
            hidden_states = features.get_hidden_states()
        
        if hidden_states is None:
            return None
        
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.cpu().float().numpy()
        
        if len(hidden_states.shape) != 3:
            return None
        
        n_layers, seq_len, hidden_dim = hidden_states.shape
        abs_idx = features.prompt_len + token_idx
        
        if abs_idx >= seq_len:
            return None
        
        if self.layer_selection == "last_4":
            start_layer = max(0, n_layers - 4)
        else:
            start_layer = 0
        
        token_hidden = hidden_states[start_layer:, abs_idx, :]
        
        feat_list = [token_hidden.flatten()]
        
        if self.use_layer_stats:
            feat_list.append(token_hidden.mean(axis=0))
            feat_list.append(token_hidden.std(axis=0))
        
        result = np.concatenate(feat_list).astype(np.float32)
        
        if np.any(~np.isfinite(result)):
            result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return result


@METHODS.register("hidden_state_probe", aliases=["hs_probe", "activation_probe"])
class HiddenStateProbeMethod(BaseMethod):
    """基于 Hidden State 的通用 Probe 方法。"""
    
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
    
    def _pool_sequence(
        self,
        hidden_states: np.ndarray,
        prompt_len: int,
        response_len: int
    ) -> np.ndarray:
        """对序列维度进行池化。"""
        n_layers, seq_len, hidden_dim = hidden_states.shape
        
        resp_start = min(prompt_len, seq_len)
        resp_end = min(prompt_len + response_len, seq_len)
        
        if resp_end <= resp_start:
            resp_start = 0
            resp_end = seq_len
        
        resp_hidden = hidden_states[:, resp_start:resp_end, :]
        
        if self.pooling == "last":
            return resp_hidden[:, -1, :]
        elif self.pooling == "mean":
            return resp_hidden.mean(axis=1)
        elif self.pooling == "max":
            return resp_hidden.max(axis=1)
        elif self.pooling == "cls":
            return hidden_states[:, 0, :]
        else:
            return resp_hidden.mean(axis=1)
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """提取特征。"""
        hidden_states = features.hidden_states
        if hidden_states is None:
            hidden_states = features.get_hidden_states()
        
        if hidden_states is None:
            raise ValueError("HiddenStateProbe requires hidden_states")
        
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.cpu().float().numpy()
        
        if len(hidden_states.shape) == 2:
            return hidden_states.flatten()
        
        selected = self._select_layers(hidden_states)
        pooled = self._pool_sequence(selected, features.prompt_len, features.response_len)
        
        feat_vec = pooled.flatten()
        
        if np.any(~np.isfinite(feat_vec)):
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feat_vec.astype(np.float32)