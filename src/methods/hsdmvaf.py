"""
HSDMVAF - Hallucinated Span Detection with Multi-View Attention Features

基于多视角注意力特征的幻觉检测方法
使用 Transformer 编码器 + CRF 进行 token 级别的幻觉检测

参考论文: "Hallucinated Span Detection with Multi-View Attention Features"
论文链接: https://aclanthology.org/2025.starsem-1.31/
代码参考: https://github.com/Ogamon958/mva_hal_

核心思想：
论文从注意力矩阵中提取三种互补的特征：
1. Average Incoming Attention (avg_in): 每个 token 收到的平均注意力权重
   - 表示该 token 对其他 token 的影响程度
   - 高值表示该 token 是"关键"token
   
2. Diversity of Incoming Attention (div_in): 入向注意力的多样性
   - 使用熵或标准差衡量
   - 表示注意力是否均匀分布在该 token 上
   
3. Diversity of Outgoing Attention (div_out): 出向注意力的多样性
   - 该 token 在生成时关注其他 token 的分散程度
   - 高值表示该 token 是基于广泛上下文生成的

这些特征输入到 Transformer Encoder + CRF 进行序列标注。
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.core import ExtractedFeatures, MethodConfig, METHODS
from .base import BaseMethod

logger = logging.getLogger(__name__)


# =============================================================================
# Multi-View Attention Feature Extraction (论文核心)
# =============================================================================

def compute_multi_view_attention_features(
    attention: torch.Tensor,
    prompt_len: int,
    response_len: int,
    normalize: bool = True,
) -> torch.Tensor:
    """计算 Multi-View Attention Features。
    
    根据论文 "Hallucinated Span Detection with Multi-View Attention Features"
    从注意力矩阵中提取三种互补特征。
    
    Args:
        attention: 注意力矩阵 [n_layers, n_heads, seq_len, seq_len]
                  或 [n_heads, seq_len, seq_len] (单层)
        prompt_len: Prompt 长度
        response_len: Response 长度
        normalize: 是否标准化特征
        
    Returns:
        Multi-view features [resp_len, feature_dim]
        feature_dim = n_layers * n_heads * 3 (avg_in, div_in, div_out)
    """
    if isinstance(attention, np.ndarray):
        attention = torch.from_numpy(attention)
    
    attention = attention.float()
    
    # 处理不同形状
    if len(attention.shape) == 3:
        # [n_heads, seq_len, seq_len] -> [1, n_heads, seq_len, seq_len]
        attention = attention.unsqueeze(0)
    
    n_layers, n_heads, seq_len, _ = attention.shape
    
    # 确定 response 范围
    resp_start = min(prompt_len, seq_len)
    resp_end = min(prompt_len + response_len, seq_len)
    
    if resp_end <= resp_start:
        resp_start = 0
        resp_end = seq_len
    
    actual_resp_len = resp_end - resp_start
    
    # 为每个 response token 计算特征
    features_list = []
    
    for layer in range(n_layers):
        for head in range(n_heads):
            attn = attention[layer, head]  # [seq_len, seq_len]
            
            # 只关注 response tokens
            # attn[i, j] = token i 对 token j 的注意力权重
            
            # 1. Average Incoming Attention (avg_in)
            # 每个 response token 收到的平均注意力
            # = 所有 token 对该 token 的注意力权重的平均
            # 注意：由于是 causal attention，只有后续 token 能关注前面的 token
            avg_in = []
            for t in range(resp_start, resp_end):
                # 从所有后续 token 收到的注意力
                if t < seq_len - 1:
                    incoming = attn[t+1:, t]  # 后续 token 对 token t 的注意力
                    avg_in.append(incoming.mean().item() if len(incoming) > 0 else 0.0)
                else:
                    avg_in.append(0.0)
            avg_in = torch.tensor(avg_in, dtype=torch.float32)
            
            # 2. Diversity of Incoming Attention (div_in)
            # 入向注意力的多样性（使用熵）
            div_in = []
            for t in range(resp_start, resp_end):
                if t < seq_len - 1:
                    incoming = attn[t+1:, t]
                    if len(incoming) > 0 and incoming.sum() > 1e-10:
                        # 计算熵
                        p = incoming / (incoming.sum() + 1e-10)
                        p = torch.clamp(p, min=1e-10)
                        entropy = -torch.sum(p * torch.log(p))
                        div_in.append(entropy.item())
                    else:
                        div_in.append(0.0)
                else:
                    div_in.append(0.0)
            div_in = torch.tensor(div_in, dtype=torch.float32)
            
            # 3. Diversity of Outgoing Attention (div_out)
            # 出向注意力的多样性
            # 该 token 在生成时关注其他 token 的分散程度
            div_out = []
            for t in range(resp_start, resp_end):
                outgoing = attn[t, :t+1]  # token t 对之前所有 token 的注意力
                if len(outgoing) > 0 and outgoing.sum() > 1e-10:
                    # 计算熵
                    p = outgoing / (outgoing.sum() + 1e-10)
                    p = torch.clamp(p, min=1e-10)
                    entropy = -torch.sum(p * torch.log(p))
                    div_out.append(entropy.item())
                else:
                    div_out.append(0.0)
            div_out = torch.tensor(div_out, dtype=torch.float32)
            
            # 拼接这三个特征
            layer_head_features = torch.stack([avg_in, div_in, div_out], dim=1)  # [resp_len, 3]
            features_list.append(layer_head_features)
    
    # 拼接所有层和头的特征
    # [resp_len, n_layers * n_heads * 3]
    all_features = torch.cat(features_list, dim=1)
    
    # 标准化
    if normalize:
        mean = all_features.mean(dim=0, keepdim=True)
        std = all_features.std(dim=0, keepdim=True) + 1e-8
        all_features = (all_features - mean) / std
    
    return all_features


def compute_mva_features_from_diags(
    attn_diags: torch.Tensor,
    attn_entropy: Optional[torch.Tensor],
    prompt_len: int,
    response_len: int,
) -> torch.Tensor:
    """从对角线特征近似计算 MVA 特征。
    
    当没有完整注意力矩阵时，使用对角线特征进行近似。
    
    Args:
        attn_diags: 注意力对角线 [n_layers, n_heads, seq_len]
        attn_entropy: 注意力熵 [n_layers, n_heads, seq_len] (可选)
        prompt_len: Prompt 长度
        response_len: Response 长度
        
    Returns:
        近似的 MVA features [resp_len, feature_dim]
    """
    if isinstance(attn_diags, np.ndarray):
        attn_diags = torch.from_numpy(attn_diags)
    
    attn_diags = attn_diags.float()
    
    n_layers, n_heads, seq_len = attn_diags.shape
    
    # 确定 response 范围
    resp_start = min(prompt_len, seq_len)
    resp_end = min(prompt_len + response_len, seq_len)
    
    if resp_end <= resp_start:
        resp_start = 0
        resp_end = seq_len
    
    # 提取 response 部分的对角线
    resp_diags = attn_diags[:, :, resp_start:resp_end]  # [n_layers, n_heads, resp_len]
    
    features_list = []
    
    # 使用对角线值作为 self-attention 强度的代理
    # 这不是完美的 MVA 特征，但可以作为近似
    for layer in range(n_layers):
        for head in range(n_heads):
            diag = resp_diags[layer, head]  # [resp_len]
            
            # 特征 1: 对角线值（自注意力强度）
            feat1 = diag
            
            # 特征 2: 对角线值的局部变化（近似 diversity）
            if len(diag) > 1:
                diff = torch.zeros_like(diag)
                diff[1:] = torch.abs(diag[1:] - diag[:-1])
                feat2 = diff
            else:
                feat2 = torch.zeros_like(diag)
            
            # 特征 3: 累积统计（近似 incoming attention）
            feat3 = torch.cumsum(diag, dim=0) / (torch.arange(len(diag), dtype=torch.float32) + 1)
            
            layer_head_features = torch.stack([feat1, feat2, feat3], dim=1)  # [resp_len, 3]
            features_list.append(layer_head_features)
    
    # 添加 attention entropy（如果有）
    if attn_entropy is not None:
        if isinstance(attn_entropy, np.ndarray):
            attn_entropy = torch.from_numpy(attn_entropy)
        
        resp_entropy = attn_entropy[:, :, resp_start:resp_end].float()
        
        for layer in range(n_layers):
            for head in range(n_heads):
                entropy = resp_entropy[layer, head]  # [resp_len]
                # 添加熵作为额外特征
                features_list.append(entropy.unsqueeze(1))  # [resp_len, 1]
    
    all_features = torch.cat(features_list, dim=1)
    
    # 标准化
    mean = all_features.mean(dim=0, keepdim=True)
    std = all_features.std(dim=0, keepdim=True) + 1e-8
    all_features = (all_features - mean) / std
    
    return all_features


# =============================================================================
# Transformer + CRF 模型
# =============================================================================

class MultiViewAttentionEncoder(nn.Module):
    """多视角注意力特征编码器。"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        
        if mask is not None:
            src_key_padding_mask = ~mask.bool()
        else:
            src_key_padding_mask = None
        
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        
        return x


class CRFLayer(nn.Module):
    """条件随机场层。"""
    
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
    
    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_algorithm(emissions, mask)
        
        return (forward_score - gold_score).mean()
    
    def _score_sentence(self, emissions, tags, mask):
        batch_size, seq_len = tags.shape
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        
        for i in range(1, seq_len):
            score += self.transitions[tags[:, i-1], tags[:, i]] * mask[:, i].float()
            score += emissions[:, i].gather(1, tags[:, i:i+1]).squeeze(1) * mask[:, i].float()
        
        last_tag_indices = mask.sum(1).long() - 1
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        
        return score
    
    def _forward_algorithm(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, i:i+1].bool(), next_score, score)
        
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)
    
    def decode(self, emissions, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []
        
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[:, i:i+1].bool(), next_score, score)
            history.append(indices)
        
        score += self.end_transitions
        best_tags_list = []
        _, best_last_tag = score.max(dim=1)
        
        for b in range(batch_size):
            best_tags = [best_last_tag[b].item()]
            seq_length = int(mask[b].sum().item())
            
            for hist in reversed(history[:seq_length-1]):
                best_tags.append(hist[b, best_tags[-1]].item())
            
            best_tags.reverse()
            best_tags_list.append(best_tags)
        
        return best_tags_list


class HSDMVAFModel(nn.Module):
    """HSDMVAF 模型：Transformer Encoder + CRF。"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.input_dim = config.get('input_dim', 128)
        
        enc_config = config.get('encoder', {})
        hidden_dim = enc_config.get('hidden_dim', 256)
        
        self.encoder = MultiViewAttentionEncoder(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_heads=enc_config.get('num_heads', 8),
            num_layers=enc_config.get('num_layers', 4),
            feedforward_dim=enc_config.get('feedforward_dim', 512),
            dropout=enc_config.get('dropout', 0.1)
        )
        
        self.use_crf = config.get('use_crf', True)
        self.classifier = nn.Linear(hidden_dim, 2)
        
        if self.use_crf:
            self.crf = CRFLayer(num_tags=2)
    
    def forward(self, features, labels=None, mask=None):
        hidden = self.encoder(features, mask)
        logits = self.classifier(hidden)
        
        output = {'logits': logits}
        
        if labels is not None:
            if self.use_crf:
                if mask is None:
                    mask = labels != -100
                valid_labels = labels.clone()
                valid_labels[labels == -100] = 0
                output['loss'] = self.crf(logits, valid_labels, mask)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                output['loss'] = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        if self.use_crf:
            output['predictions'] = self.crf.decode(logits, mask)
        else:
            output['predictions'] = logits.argmax(dim=-1)
        
        return output
    
    def predict_proba(self, features, mask=None):
        with torch.no_grad():
            hidden = self.encoder(features, mask)
            logits = self.classifier(hidden)
            probs = F.softmax(logits, dim=-1)
        return probs[:, :, 1]


# =============================================================================
# BaseMethod 兼容的 HSDMVAF 方法
# =============================================================================

@METHODS.register("hsdmvaf", aliases=["mva", "multi_view_attention"])
class HSDMVAFMethod(BaseMethod):
    """HSDMVAF 幻觉检测方法（样本级别）。
    
    基于论文: "Hallucinated Span Detection with Multi-View Attention Features"
    
    这个实现将 token 级别的预测聚合为样本级别的分数，
    以便与其他方法进行公平比较。
    
    特征提取：
    1. 如果有 full_attention: 使用完整的 MVA 特征
    2. 否则使用 attn_diags + attn_entropy 近似
    
    参数配置：
    - pooling: str - 样本级别聚合方式 ("max", "mean", "any")
    - use_full_attention: bool - 是否使用完整注意力矩阵
    """
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        
        params = self.config.params or {}
        self.pooling = params.get("pooling", "max")
        self.use_full_attention = params.get("use_full_attention", True)
        self.max_layers = params.get("max_layers", 4)  # 限制层数避免特征过大
        self.max_heads = params.get("max_heads", 8)
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """提取 MVA 特征并聚合为样本级别。"""
        
        # 尝试使用完整注意力矩阵 (使用懒加载)
        full_attention = features.get_full_attention() if self.use_full_attention else None
        
        if full_attention is not None:
            mva_features = compute_multi_view_attention_features(
                attention=full_attention,
                prompt_len=features.prompt_len,
                response_len=features.response_len,
            )
            # Release large feature after use
            features.release_large_features()
        elif features.attn_diags is not None:
            # 使用对角线近似
            mva_features = compute_mva_features_from_diags(
                attn_diags=features.attn_diags,
                attn_entropy=features.attn_entropy,
                prompt_len=features.prompt_len,
                response_len=features.response_len,
            )
        else:
            raise ValueError("HSDMVAF requires full_attention or attn_diags")
        
        # 转换为 numpy
        if isinstance(mva_features, torch.Tensor):
            mva_features = mva_features.float().cpu().numpy()
        
        # Handle NaN/Inf
        if np.any(~np.isfinite(mva_features)):
            mva_features = np.nan_to_num(mva_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 聚合为样本级别特征
        # [resp_len, feature_dim] -> [aggregated_dim]
        
        if len(mva_features) == 0:
            return np.zeros(mva_features.shape[1] * 4 if len(mva_features.shape) > 1 else 4)
        
        # 使用多种统计量
        sample_features = []
        
        # 均值
        sample_features.append(mva_features.mean(axis=0))
        
        # 最大值
        sample_features.append(mva_features.max(axis=0))
        
        # 标准差
        sample_features.append(mva_features.std(axis=0))
        
        # 最后一个 token（通常最重要）
        sample_features.append(mva_features[-1])
        
        result = np.concatenate(sample_features).astype(np.float32)
        
        # Final NaN check
        if np.any(~np.isfinite(result)):
            result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return result


# =============================================================================
# 独立的 HSDMVAF 检测器（支持 token 级别检测）
# =============================================================================

class HSDMVAFDetector:
    """HSDMVAF 检测器，支持 token 级别的幻觉检测。"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _extract_mva_features(self, data: Dict[str, Any]) -> torch.Tensor:
        """从数据中提取 MVA 特征。"""
        prompt_len = data.get('prompt_len', 0)
        response_len = data.get('response_len', 100)
        
        if 'full_attention' in data and data['full_attention'] is not None:
            return compute_multi_view_attention_features(
                data['full_attention'], prompt_len, response_len
            )
        elif 'attn_diags' in data and data['attn_diags'] is not None:
            return compute_mva_features_from_diags(
                data['attn_diags'],
                data.get('attn_entropy'),
                prompt_len, response_len
            )
        else:
            raise ValueError("需要 full_attention 或 attn_diags")
    
    def fit(self, train_data: List[Dict], val_data: List[Dict]):
        """训练模型。"""
        # 提取特征
        train_features = []
        train_labels = []
        
        for item in train_data:
            try:
                feat = self._extract_mva_features(item)
                train_features.append(feat)
                
                label = item.get('hallucination_labels', item.get('label', 0))
                if isinstance(label, int):
                    label = [label] * len(feat)
                train_labels.append(torch.tensor(label))
            except Exception as e:
                logger.warning(f"跳过样本: {e}")
        
        if not train_features:
            raise ValueError("没有有效的训练数据")
        
        # Padding
        max_len = max(f.shape[0] for f in train_features)
        padded_features = []
        padded_labels = []
        
        for feat, label in zip(train_features, train_labels):
            pad_len = max_len - feat.shape[0]
            if pad_len > 0:
                feat = F.pad(feat, (0, 0, 0, pad_len))
                label = F.pad(label, (0, pad_len), value=-100)
            padded_features.append(feat)
            padded_labels.append(label)
        
        X = torch.stack(padded_features)
        y = torch.stack(padded_labels)
        
        # 初始化模型
        self.config['input_dim'] = X.shape[-1]
        self.model = HSDMVAFModel(self.config).to(self.device)
        
        # 训练
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        for epoch in range(self.config.get('epochs', 50)):
            self.model.train()
            X_dev = X.to(self.device)
            y_dev = y.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(X_dev, y_dev)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    def predict(self, data: Dict) -> Dict[str, Any]:
        """预测单个样本。"""
        self.model.eval()
        
        feat = self._extract_mva_features(data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = self.model.predict_proba(feat)
        
        probs_np = probs[0].float().cpu().numpy()
        
        pooling = self.config.get('pooling', 'max')
        if pooling == 'max':
            sample_prob = float(probs_np.max())
        elif pooling == 'mean':
            sample_prob = float(probs_np.mean())
        else:
            sample_prob = float((probs_np > 0.5).any())
        
        return {
            'label': int(sample_prob > 0.5),
            'proba': sample_prob,
            'token_probs': probs_np.tolist()
        }
    
    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config
        }, path / 'model.pt')
    
    def load(self, path: Path):
        path = Path(path)
        checkpoint = torch.load(path / 'model.pt', map_location=self.device)
        self.config = checkpoint['config']
        self.model = HSDMVAFModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])