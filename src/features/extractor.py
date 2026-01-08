"""Feature extraction with memory safety controls.

特征提取模块，支持：
- 内存安全控制（full attention 默认禁用）
- 逐层处理和即时清理
- 多种特征类型提取
- 内存预估功能

Usage:
    from src.features.extractor import FeatureExtractor, create_extractor
    
    extractor = create_extractor(model, config)
    features = extractor.extract(sample)
    
    # 检查内存需求
    mem_estimate = extractor.get_memory_estimate(seq_len=2048)
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging
import torch

from src.core import (
    Sample, ExtractedFeatures, FeaturesConfig, ExtractionMode,
    parse_layers, Progress, FeatureError,
)
from src.models import LoadedModel
from src.utils import (
    get_model_device,
    clear_gpu_memory,
    extract_attention_diagonal,
    compute_laplacian_diagonal,
    compute_attention_entropy,
    pool_hidden_states,
    compute_token_probs,
)
from .hallucination_spans import (
    extract_hallucination_info_from_sample,
    calculate_hallucination_token_spans,
    get_token_hallucination_labels,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 内存估算常量
# =============================================================================

# 每个token的大致内存占用（float16，单位：bytes）
BYTES_PER_ATTENTION_DIAG = 2        # [n_heads] per token per layer
BYTES_PER_LAPLACIAN_DIAG = 2        # [n_heads] per token per layer
BYTES_PER_ATTENTION_ENTROPY = 2    # [n_heads] per token per layer
BYTES_PER_HIDDEN_STATE = 2          # [hidden_size] per token per layer
BYTES_PER_TOKEN_PROB = 4            # float32 per token
BYTES_PER_FULL_ATTENTION = 2        # [seq_len * n_heads] per token per layer


# =============================================================================
# 特征提取器
# =============================================================================

class FeatureExtractor:
    """特征提取器，支持内存安全的特征提取。
    
    关键特性：
    - `store_full_attention` 默认为 False，必须显式启用
    - 逐层处理注意力，处理完立即清理GPU内存
    - 支持负数索引（如 -4, -3, -2, -1 表示最后4层）
    - 提供内存预估功能
    
    Attributes:
        model: 已加载的模型实例
        config: 特征提取配置
        attn_layers: 要提取注意力的层索引
        hidden_layers: 要提取隐藏状态的层索引
    """
    
    def __init__(
        self,
        model: LoadedModel,
        config: FeaturesConfig,
        store_full_attention: bool = False,
    ):
        """初始化特征提取器。
        
        Args:
            model: 已加载的模型实例
            config: 特征提取配置
            store_full_attention: 是否存储完整注意力矩阵
                ⚠️ 警告：启用此选项会消耗大量内存！
                对于 seq_len=2048, n_layers=32, n_heads=32:
                内存需求约 ~17GB
        """
        self.model = model
        self.config = config
        
        # ⚠️ 内存安全控制：full_attention 必须显式启用
        self._store_full_attention = store_full_attention
        if self._store_full_attention:
            logger.warning(
                "⚠️ Full attention storage ENABLED - this requires significant memory! "
                "Consider using store_full_attention=False unless absolutely necessary."
            )
        
        n_layers = model.num_layers
        
        # 解析层索引（支持负数）
        self.attn_layers = self._parse_layers_with_negative(
            config.attention_layers, n_layers
        ) if config.attention_enabled else []
        
        self.hidden_layers = self._parse_layers_with_negative(
            config.hidden_states_layers, n_layers
        ) if config.hidden_states_enabled else []
        
        logger.info(
            f"Extractor initialized: "
            f"attn_layers={len(self.attn_layers)}, "
            f"hidden_layers={len(self.hidden_layers)}, "
            f"store_full_attention={self._store_full_attention}"
        )
    
    def _parse_layers_with_negative(
        self,
        layer_spec: str,
        n_layers: int
    ) -> List[int]:
        """解析层规格，支持负数索引。
        
        Args:
            layer_spec: 层规格字符串，如 "all", "last_4", "-4,-3,-2,-1"
            n_layers: 模型总层数
            
        Returns:
            层索引列表（正数）
        """
        # 首先使用标准解析
        try:
            layers = parse_layers(layer_spec, n_layers)
        except:
            layers = []
        
        # 如果包含负数，转换为正数索引
        if layer_spec and any(c == '-' for c in str(layer_spec)):
            try:
                parts = str(layer_spec).split(',')
                layers = []
                for p in parts:
                    p = p.strip()
                    if p.lstrip('-').isdigit():
                        idx = int(p)
                        if idx < 0:
                            idx = n_layers + idx
                        if 0 <= idx < n_layers:
                            layers.append(idx)
            except:
                pass
        
        return sorted(set(layers))
    
    def get_memory_estimate(
        self,
        seq_len: int = 2048,
        batch_size: int = 1,
    ) -> Dict[str, float]:
        """估算特征提取的内存需求。
        
        Args:
            seq_len: 序列长度
            batch_size: 批次大小
            
        Returns:
            各特征类型的内存估算（单位：MB）
        """
        n_heads = self.model.num_heads
        hidden_size = self.model.hidden_size
        n_attn_layers = len(self.attn_layers)
        n_hidden_layers = len(self.hidden_layers)
        
        estimates = {}
        
        # 注意力对角线
        if self.config.attention_enabled:
            attn_diag_bytes = batch_size * seq_len * n_heads * n_attn_layers * BYTES_PER_ATTENTION_DIAG
            estimates["attention_diags_mb"] = attn_diag_bytes / 1024**2
            
            lap_diag_bytes = batch_size * seq_len * n_heads * n_attn_layers * BYTES_PER_LAPLACIAN_DIAG
            estimates["laplacian_diags_mb"] = lap_diag_bytes / 1024**2
            
            entropy_bytes = batch_size * seq_len * n_heads * n_attn_layers * BYTES_PER_ATTENTION_ENTROPY
            estimates["attention_entropy_mb"] = entropy_bytes / 1024**2
        
        # 完整注意力
        if self._store_full_attention:
            full_attn_bytes = batch_size * seq_len * seq_len * n_heads * n_attn_layers * BYTES_PER_FULL_ATTENTION
            estimates["full_attention_mb"] = full_attn_bytes / 1024**2
        
        # 隐藏状态
        if self.config.hidden_states_enabled:
            hs_bytes = batch_size * seq_len * hidden_size * n_hidden_layers * BYTES_PER_HIDDEN_STATE
            estimates["hidden_states_mb"] = hs_bytes / 1024**2
        
        # Token概率
        if self.config.token_probs_enabled:
            prob_bytes = batch_size * seq_len * BYTES_PER_TOKEN_PROB
            estimates["token_probs_mb"] = prob_bytes / 1024**2
        
        # 总计
        estimates["total_mb"] = sum(estimates.values())
        estimates["total_gb"] = estimates["total_mb"] / 1024
        
        return estimates
    
    def extract(self, sample: Sample) -> ExtractedFeatures:
        """从单个样本提取特征。
        
        Args:
            sample: 输入样本
            
        Returns:
            提取的特征
        """
        if self.config.mode == "teacher_forcing":
            return self._extract_teacher_forcing(sample)
        elif self.config.mode == "generation":
            return self._extract_generation(sample)
        else:
            raise FeatureError(f"Unknown extraction mode: {self.config.mode}")
    
    @torch.inference_mode()
    def _extract_teacher_forcing(self, sample: Sample) -> ExtractedFeatures:
        """使用 teacher forcing 模式提取特征。
        
        使用真实响应计算特征，逐层处理以优化内存。
        """
        device = get_model_device(self.model.model)
        
        # =====================================================================
        # 分词
        # =====================================================================
        prompt_ids = self.model.encode(sample.prompt, add_special_tokens=True)
        response_ids = self.model.encode(sample.response, add_special_tokens=False)
        
        prompt_len = prompt_ids.size(1)
        response_len = response_ids.size(1)
        
        # 拼接并截断
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        if input_ids.size(1) > self.config.max_length:
            input_ids = input_ids[:, :self.config.max_length]
            response_len = max(0, self.config.max_length - prompt_len)
        
        seq_len = input_ids.size(1)
        input_ids = input_ids.to(device)
        
        # =====================================================================
        # 前向传播
        # =====================================================================
        outputs = self.model.model(
            input_ids=input_ids,
            output_attentions=self.config.attention_enabled,
            output_hidden_states=self.config.hidden_states_enabled,
            return_dict=True,
        )
        
        # =====================================================================
        # 初始化特征容器
        # =====================================================================
        features = ExtractedFeatures(
            sample_id=sample.id,
            prompt_len=prompt_len,
            response_len=response_len,
            label=sample.label,
            layers=self.attn_layers or self.hidden_layers,
            model_name=self.model.config.name,
            mode=ExtractionMode.TEACHER_FORCING,
        )
        
        # =====================================================================
        # 逐层提取注意力特征（内存优化）
        # =====================================================================
        if self.config.attention_enabled and outputs.attentions is not None:
            attn_diags = []
            lap_diags = []
            attn_entropy = []
            full_attention_list = [] if self._store_full_attention else None
            
            for layer_idx in self.attn_layers:
                if layer_idx < len(outputs.attentions):
                    # 获取当前层注意力
                    attn = outputs.attentions[layer_idx]
                    
                    # 提取对角线特征
                    attn_diags.append(extract_attention_diagonal(attn).squeeze(0).cpu())
                    lap_diags.append(compute_laplacian_diagonal(attn).squeeze(0).cpu())
                    attn_entropy.append(compute_attention_entropy(attn).squeeze(0).cpu())
                    
                    # 完整注意力（仅在显式启用时存储）
                    if self._store_full_attention:
                        full_attention_list.append(attn.squeeze(0).cpu())
                    
                    # ⚠️ 立即清理GPU张量
                    del attn
            
            # 堆叠特征
            if attn_diags:
                features.attn_diags = torch.stack(attn_diags, dim=0)
                features.laplacian_diags = torch.stack(lap_diags, dim=0)
                features.attn_entropy = torch.stack(attn_entropy, dim=0)
                
                if self._store_full_attention and full_attention_list:
                    features.full_attention = torch.stack(full_attention_list, dim=0)
            
            # 清理临时列表
            del attn_diags, lap_diags, attn_entropy
            if full_attention_list is not None:
                del full_attention_list
        
        # =====================================================================
        # 逐层提取隐藏状态（内存优化）
        # =====================================================================
        if self.config.hidden_states_enabled and outputs.hidden_states is not None:
            hs_list = []
            
            for layer_idx in self.hidden_layers:
                if layer_idx < len(outputs.hidden_states):
                    # 获取当前层隐藏状态
                    hs = outputs.hidden_states[layer_idx]
                    
                    # 池化
                    pooled = pool_hidden_states(hs, self.config.hidden_states_pooling)
                    hs_list.append(pooled.squeeze(0).cpu())
                    
                    # ⚠️ 立即清理GPU张量
                    del hs, pooled
            
            if hs_list:
                features.hidden_states = torch.stack(hs_list, dim=0)
            
            del hs_list
        
        # =====================================================================
        # Token概率
        # =====================================================================
        if self.config.token_probs_enabled and response_len > 0:
            logits = outputs.logits
            response_start = prompt_len
            
            if logits.size(1) > response_start:
                features.token_probs = compute_token_probs(
                    logits[:, response_start-1:, :],
                    input_ids[:, response_start-1:],
                ).squeeze(0).cpu()
                
                if features.token_probs.numel() > 0:
                    features.perplexity = torch.exp(
                        -torch.log(features.token_probs.clamp(min=1e-10)).mean()
                    ).item()
        
        # =====================================================================
        # 清理
        # =====================================================================
        del outputs, input_ids
        
        # 强制清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # =====================================================================
        # Token级幻觉标签
        # =====================================================================
        try:
            span_labels, has_spans = extract_hallucination_info_from_sample(sample.metadata)
            if has_spans and sample.label == 1:
                token_spans = calculate_hallucination_token_spans(
                    labels=span_labels,
                    prompt_text=sample.prompt,
                    response_text=sample.response,
                    tokenizer=self.model.tokenizer,
                    end_inclusive=False,
                )
                features.hallucination_labels = get_token_hallucination_labels(seq_len, token_spans)
                features.hallucination_token_spans = token_spans
        except Exception as e:
            logger.debug(f"Could not calculate hallucination labels for {sample.id}: {e}")
        
        return features
    
    @torch.inference_mode()
    def _extract_generation(self, sample: Sample) -> ExtractedFeatures:
        """使用生成模式提取特征。
        
        先生成响应，然后使用 teacher forcing 提取特征。
        """
        device = get_model_device(self.model.model)
        
        prompt_ids = self.model.encode(sample.prompt, add_special_tokens=True).to(device)
        prompt_len = prompt_ids.size(1)
        
        gen_outputs = self.model.generate(
            input_ids=prompt_ids,
            max_new_tokens=self.config.max_length - prompt_len,
            output_attentions=False,
            output_hidden_states=False,
        )
        
        generated_ids = gen_outputs["generated_ids"]
        response_ids = generated_ids[:, prompt_len:]
        generated_response = self.model.decode(response_ids[0])
        
        del gen_outputs, prompt_ids, generated_ids, response_ids
        
        sample_with_gen = Sample(
            id=sample.id,
            prompt=sample.prompt,
            response=generated_response,
            reference=sample.reference,
            label=sample.label,
            task_type=sample.task_type,
            metadata={**sample.metadata, "generated": True},
        )
        
        features = self._extract_teacher_forcing(sample_with_gen)
        features.mode = ExtractionMode.GENERATION
        features.metadata["generated_response"] = generated_response
        
        return features
    
    def extract_batch(
        self,
        samples: List[Sample],
        show_progress: bool = True,
    ) -> List[ExtractedFeatures]:
        """批量提取特征。
        
        Args:
            samples: 输入样本列表
            show_progress: 是否显示进度条
            
        Returns:
            提取的特征列表
        """
        features_list = []
        
        if show_progress:
            pbar = Progress(len(samples), desc="Extracting features")
        
        for sample in samples:
            try:
                features = self.extract(sample)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract features for {sample.id}: {e}")
            
            # 每个样本后清理GPU内存
            clear_gpu_memory()
            
            if show_progress:
                pbar.update()
        
        return features_list


# =============================================================================
# 工厂函数
# =============================================================================

def create_extractor(
    model: LoadedModel,
    config: FeaturesConfig,
    store_full_attention: bool = False,
) -> FeatureExtractor:
    """创建特征提取器。
    
    Args:
        model: 已加载的模型
        config: 特征配置
        store_full_attention: 是否存储完整注意力矩阵
            ⚠️ 警告：这会消耗大量内存！
        
    Returns:
        FeatureExtractor 实例
    """
    return FeatureExtractor(
        model=model,
        config=config,
        store_full_attention=store_full_attention,
    )


def create_extractor_from_requirements(
    model: LoadedModel,
    config: FeaturesConfig,
    feature_requirements: Dict[str, bool],
    allow_full_attention: bool = False,
) -> FeatureExtractor:
    """根据特征需求创建提取器。
    
    与 FeatureManager 集成使用。
    
    Args:
        model: 已加载的模型
        config: 基础特征配置
        feature_requirements: 特征需求字典，如：
            {
                "attention_diags": True,
                "hidden_states": True,
                "full_attention": False,
            }
        allow_full_attention: 是否允许完整注意力提取
        
    Returns:
        配置好的 FeatureExtractor 实例
    """
    # 更新配置
    if "attention_diags" in feature_requirements:
        config.attention_enabled = feature_requirements.get("attention_diags", False)
    
    if "hidden_states" in feature_requirements:
        config.hidden_states_enabled = feature_requirements.get("hidden_states", False)
    
    if "token_probs" in feature_requirements:
        config.token_probs_enabled = feature_requirements.get("token_probs", False)
    
    # 决定是否存储完整注意力
    store_full = False
    if feature_requirements.get("full_attention", False):
        if allow_full_attention:
            store_full = True
            logger.warning("Full attention extraction ENABLED by feature requirements")
        else:
            logger.warning(
                "Full attention requested but allow_full_attention=False. "
                "Full attention will NOT be extracted."
            )
    
    return FeatureExtractor(
        model=model,
        config=config,
        store_full_attention=store_full,
    )