"""Token Probability Feature Extractor - Token 概率提取。

提取模型预测的 token 概率，支持：
- Top-k 概率
- 熵计算
- 困惑度计算

Usage:
    from src.features.base.token_probs import TokenProbsExtractor
    
    extractor = TokenProbsExtractor(
        compute_entropy=True,
        compute_perplexity=True,
    )
    
    probs = extractor.extract(logits, input_ids)
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TokenProbsExtractionConfig:
    """Token 概率提取配置。"""
    compute_entropy: bool = True           # 计算熵
    compute_perplexity: bool = True        # 计算困惑度
    top_k: int = 5                          # 保存 top-k 概率
    response_only: bool = True             # 只计算 response 部分
    store_on_cpu: bool = True
    half_precision: bool = False           # 概率通常用 float32


class TokenProbsExtractor:
    """Token 概率提取器。"""
    
    def __init__(self, config: Optional[TokenProbsExtractionConfig] = None):
        self.config = config or TokenProbsExtractionConfig()
    
    def extract(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        prompt_len: int = 0,
        response_len: int = 0,
    ) -> Dict[str, Any]:
        """从 logits 中提取 token 概率特征。
        
        Args:
            logits: 模型输出的 logits [batch, seq_len, vocab_size]
            input_ids: 输入 token ids [batch, seq_len]
            prompt_len: Prompt 长度
            response_len: Response 长度
            
        Returns:
            {
                "token_probs": 每个 token 的概率,
                "entropy": 每个位置的熵,
                "perplexity": 困惑度,
                "top_k_probs": top-k 概率,
                "top_k_indices": top-k token indices,
            }
        """
        if logits is None:
            raise ValueError("No logits available")
        
        # 移除 batch 维度
        if logits.dim() == 3:
            logits = logits.squeeze(0)  # [seq_len, vocab_size]
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)  # [seq_len]
        
        seq_len = logits.shape[0]
        
        # 确定要处理的范围
        if self.config.response_only and prompt_len > 0:
            start_idx = prompt_len
            end_idx = min(prompt_len + response_len, seq_len)
        else:
            start_idx = 0
            end_idx = seq_len
        
        # 对于 token 概率，我们需要对齐：
        # logits[i] 预测 input_ids[i+1]
        # 所以 response 部分的概率应该从 logits[prompt_len-1:end_idx-1] 预测 input_ids[prompt_len:end_idx]
        
        results = {}
        
        # 计算 softmax 概率
        with torch.no_grad():
            # 只计算需要的部分以节省内存
            relevant_logits = logits[start_idx:end_idx]
            probs = F.softmax(relevant_logits, dim=-1)
            
            # 获取实际 token 的概率
            # logits[i] 预测 token[i+1]
            if start_idx < seq_len - 1:
                target_ids = input_ids[start_idx + 1:end_idx + 1]
                if target_ids.shape[0] > relevant_logits.shape[0]:
                    target_ids = target_ids[:relevant_logits.shape[0]]
                elif target_ids.shape[0] < relevant_logits.shape[0]:
                    relevant_logits = relevant_logits[:target_ids.shape[0]]
                    probs = probs[:target_ids.shape[0]]
                
                token_probs = probs.gather(
                    dim=-1,
                    index=target_ids.unsqueeze(-1)
                ).squeeze(-1)
                
                results["token_probs"] = self._to_output(token_probs)
            
            # 计算熵
            if self.config.compute_entropy:
                entropy = self._compute_entropy(probs)
                results["entropy"] = self._to_output(entropy)
            
            # 计算困惑度
            if self.config.compute_perplexity and "token_probs" in results:
                log_probs = torch.log(token_probs + 1e-10)
                avg_log_prob = log_probs.mean()
                perplexity = torch.exp(-avg_log_prob)
                results["perplexity"] = perplexity.item()
            
            # Top-k 概率
            if self.config.top_k > 0:
                top_k_probs, top_k_indices = torch.topk(
                    probs, k=min(self.config.top_k, probs.shape[-1]), dim=-1
                )
                results["top_k_probs"] = self._to_output(top_k_probs)
                results["top_k_indices"] = self._to_output(top_k_indices)
        
        results["prompt_len"] = prompt_len
        results["response_len"] = response_len
        results["computed_range"] = (start_idx, end_idx)
        
        return results
    
    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """计算概率分布的熵。
        
        Args:
            probs: [seq_len, vocab_size] 的概率分布
            
        Returns:
            [seq_len] 的熵
        """
        # H = -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy
    
    def _to_output(self, tensor: torch.Tensor) -> torch.Tensor:
        """转换 tensor 到输出格式。"""
        if self.config.half_precision:
            tensor = tensor.half()
        if self.config.store_on_cpu:
            tensor = tensor.cpu()
        return tensor


def extract_token_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_len: int = 0,
    response_len: int = 0,
    response_only: bool = True,
) -> torch.Tensor:
    """提取 token 概率的便捷函数。
    
    Args:
        logits: 模型输出的 logits
        input_ids: 输入 token ids
        prompt_len: Prompt 长度
        response_len: Response 长度
        response_only: 是否只计算 response 部分
        
    Returns:
        Tensor [response_len] 或 [seq_len]
    """
    config = TokenProbsExtractionConfig(
        response_only=response_only,
        compute_entropy=False,
        compute_perplexity=False,
    )
    extractor = TokenProbsExtractor(config)
    result = extractor.extract(logits, input_ids, prompt_len, response_len)
    return result.get("token_probs", torch.tensor([]))


def compute_token_entropy(
    logits: torch.Tensor,
    prompt_len: int = 0,
    response_len: int = 0,
    response_only: bool = True,
) -> torch.Tensor:
    """计算 token 熵的便捷函数。
    
    Args:
        logits: 模型输出的 logits
        prompt_len: Prompt 长度
        response_len: Response 长度
        response_only: 是否只计算 response 部分
        
    Returns:
        Tensor [response_len] 或 [seq_len]
    """
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    
    seq_len = logits.shape[0]
    
    if response_only and prompt_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len)
    else:
        start_idx = 0
        end_idx = seq_len
    
    with torch.no_grad():
        relevant_logits = logits[start_idx:end_idx]
        probs = F.softmax(relevant_logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)
    
    return entropy.cpu()
