"""Feature extraction with memory safety controls and batch processing.

特征提取模块，支持：
- 内存安全控制（full attention 默认启用）
- 真正的批处理（多样本并行前向传播）
- 逐层处理和即时清理
- 多种特征类型提取
- 内存预估功能

Usage:
    from src.features.extractor import FeatureExtractor, create_extractor
    
    # 单样本提取
    extractor = create_extractor(model, config)
    features = extractor.extract(sample)
    
    # 批量提取（自动根据 batch_size 选择最优策略）
    features_list = extractor.extract_batch(samples, batch_size=16)
    
    # 检查内存需求
    mem_estimate = extractor.get_memory_estimate(seq_len=2048, batch_size=16)
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import torch
from torch.nn.utils.rnn import pad_sequence

from src.core import (
    Sample, ExtractedFeatures, FeaturesConfig, ExtractionMode,
    parse_layers, Progress, FeatureError,
)
from src.models import LoadedModel
from src.utils import (
    get_model_device,
    clear_gpu_memory,
    extract_attention_diagonal,
    compute_attention_row_sums,
    compute_laplacian_diagonal,
    compute_attention_entropy,
    pool_hidden_states,
    compute_token_probs,
    compute_token_entropy,
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

BYTES_PER_ATTENTION_DIAG = 2
BYTES_PER_LAPLACIAN_DIAG = 2
BYTES_PER_ATTENTION_ENTROPY = 2
BYTES_PER_HIDDEN_STATE = 2
BYTES_PER_TOKEN_PROB = 4
BYTES_PER_FULL_ATTENTION = 2


# =============================================================================
# 批处理辅助数据结构
# =============================================================================

@dataclass
class BatchSampleInfo:
    """批处理中每个样本的信息"""
    sample_id: str
    prompt_len: int
    response_len: int
    total_len: int
    label: int
    sample: Sample


# =============================================================================
# 特征提取器
# =============================================================================

class FeatureExtractor:
    """特征提取器，支持内存安全的特征提取和真正的批处理。
    
    关键特性：
    - `store_full_attention` 默认为 False，必须显式启用
    - 逐层处理注意力，处理完立即清理GPU内存
    - 支持负数索引（如 -4, -3, -2, -1 表示最后4层）
    - 提供内存预估功能
    - 支持真正的批处理（多样本并行前向传播）
    
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
        store_full_attention: bool = True,
    ):
        """初始化特征提取器。
        
        Args:
            model: 已加载的模型实例
            config: 特征提取配置
            store_full_attention: 是否存储完整注意力矩阵
        """
        self.model = model
        self.config = config
        self._store_full_attention = store_full_attention
        
        if self._store_full_attention:
            logger.warning(
                "⚠️ Full attention storage ENABLED - this requires significant memory!"
            )
        
        n_layers = model.num_layers
        
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
        """解析层规格，支持负数索引。"""
        try:
            layers = parse_layers(layer_spec, n_layers)
        except:
            layers = []
        
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
        """估算特征提取的内存需求。"""
        n_heads = self.model.num_heads
        hidden_size = self.model.hidden_size
        n_attn_layers = len(self.attn_layers)
        n_hidden_layers = len(self.hidden_layers)
        
        estimates = {}
        
        if self.config.attention_enabled:
            attn_diag_bytes = batch_size * seq_len * n_heads * n_attn_layers * BYTES_PER_ATTENTION_DIAG
            estimates["attention_diags_mb"] = attn_diag_bytes / 1024**2
            
            lap_diag_bytes = batch_size * seq_len * n_heads * n_attn_layers * BYTES_PER_LAPLACIAN_DIAG
            estimates["laplacian_diags_mb"] = lap_diag_bytes / 1024**2
            
            entropy_bytes = batch_size * seq_len * n_heads * n_attn_layers * BYTES_PER_ATTENTION_ENTROPY
            estimates["attention_entropy_mb"] = entropy_bytes / 1024**2
        
        if self._store_full_attention:
            full_attn_bytes = batch_size * seq_len * seq_len * n_heads * n_attn_layers * BYTES_PER_FULL_ATTENTION
            estimates["full_attention_mb"] = full_attn_bytes / 1024**2
        
        if self.config.hidden_states_enabled:
            hs_bytes = batch_size * seq_len * hidden_size * n_hidden_layers * BYTES_PER_HIDDEN_STATE
            estimates["hidden_states_mb"] = hs_bytes / 1024**2
        
        if self.config.token_probs_enabled:
            prob_bytes = batch_size * seq_len * BYTES_PER_TOKEN_PROB
            estimates["token_probs_mb"] = prob_bytes / 1024**2
        
        estimates["total_mb"] = sum(estimates.values())
        estimates["total_gb"] = estimates["total_mb"] / 1024
        
        return estimates
    
    # =========================================================================
    # 单样本提取
    # =========================================================================
    
    def extract(self, sample: Sample) -> ExtractedFeatures:
        """从单个样本提取特征。"""
        if self.config.mode == "teacher_forcing":
            return self._extract_teacher_forcing(sample)
        elif self.config.mode == "generation":
            return self._extract_generation(sample)
        else:
            raise FeatureError(f"Unknown extraction mode: {self.config.mode}")
    
    @torch.inference_mode()
    def _extract_teacher_forcing(self, sample: Sample) -> ExtractedFeatures:
        """使用 teacher forcing 模式提取特征。"""
        device = get_model_device(self.model.model)
        
        # 分词
        prompt_ids = self.model.encode(sample.prompt, add_special_tokens=True)
        response_ids = self.model.encode(sample.response, add_special_tokens=False)
        
        prompt_len = prompt_ids.size(1)
        response_len = response_ids.size(1)
        
        # 拼接并截断
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        original_total_len = input_ids.size(1)
        
        if input_ids.size(1) > self.config.max_length:
            input_ids = input_ids[:, :self.config.max_length]
            # ⚠️ 修复：同时调整 prompt_len 和 response_len
            if prompt_len > self.config.max_length:
                # 严重情况：prompt本身已超过max_length，response完全被截断
                logger.warning(
                    f"⚠️ Sample {sample.id}: prompt_len ({prompt_len}) > max_length ({self.config.max_length}). "
                    f"Response completely truncated! Consider increasing max_length (current: {self.config.max_length}). "
                    f"Total tokens: {original_total_len}"
                )
                prompt_len = self.config.max_length
                response_len = 0
            else:
                old_response_len = response_len
                response_len = max(0, self.config.max_length - prompt_len)
                if response_len < old_response_len:
                    logger.debug(
                        f"Sample {sample.id}: response truncated from {old_response_len} to {response_len} tokens"
                    )
        
        # ⚠️ 关键验证：确保response_len > 0，否则无法提取有效特征
        if response_len == 0:
            logger.warning(
                f"⚠️ Sample {sample.id} has response_len=0 after truncation. "
                f"Token features (token_probs, token_entropy) will be empty. "
                f"Prompt tokens: {prompt_len}, max_length: {self.config.max_length}"
            )
        
        seq_len = input_ids.size(1)
        input_ids = input_ids.to(device)
        
        # 前向传播
        outputs = self.model.model(
            input_ids=input_ids,
            output_attentions=self.config.attention_enabled,
            output_hidden_states=self.config.hidden_states_enabled,
            return_dict=True,
        )
        
        # 初始化特征容器
        features = ExtractedFeatures(
            sample_id=sample.id,
            prompt_len=prompt_len,
            response_len=response_len,
            label=sample.label,
            layers=self.attn_layers or self.hidden_layers,
            model_name=self.model.config.name,
            mode=ExtractionMode.TEACHER_FORCING,
        )
        
        # 提取注意力特征
        if self.config.attention_enabled and outputs.attentions is not None:
            attn_diags = []
            attn_row_sums = []
            lap_diags = []
            attn_entropy = []
            full_attention_list = [] if self._store_full_attention else None
            
            for layer_idx in self.attn_layers:
                if layer_idx < len(outputs.attentions):
                    attn = outputs.attentions[layer_idx]
                    
                    attn_diags.append(extract_attention_diagonal(attn).squeeze(0).cpu())
                    attn_row_sums.append(compute_attention_row_sums(attn).squeeze(0).cpu())
                    lap_diags.append(compute_laplacian_diagonal(attn).squeeze(0).cpu())
                    attn_entropy.append(compute_attention_entropy(attn).squeeze(0).cpu())
                    
                    if self._store_full_attention:
                        full_attention_list.append(attn.squeeze(0).cpu())
                    
                    del attn
            
            if attn_diags:
                features.attn_diags = torch.stack(attn_diags, dim=0)
                features.attn_row_sums = torch.stack(attn_row_sums, dim=0)
                features.laplacian_diags = torch.stack(lap_diags, dim=0)
                features.attn_entropy = torch.stack(attn_entropy, dim=0)
                
                if self._store_full_attention and full_attention_list:
                    features.full_attention = torch.stack(full_attention_list, dim=0)
            
            del attn_diags, attn_row_sums, lap_diags, attn_entropy
            if full_attention_list is not None:
                del full_attention_list
        
        # 提取隐藏状态
        if self.config.hidden_states_enabled and outputs.hidden_states is not None:
            hs_list = []
            
            for layer_idx in self.hidden_layers:
                if layer_idx < len(outputs.hidden_states):
                    hs = outputs.hidden_states[layer_idx]
                    pooled = pool_hidden_states(hs, self.config.hidden_states_pooling)
                    hs_list.append(pooled.squeeze(0).cpu())
                    del hs, pooled
            
            if hs_list:
                features.hidden_states = torch.stack(hs_list, dim=0)
            
            del hs_list
        
        # Token概率和Token熵
        if self.config.token_probs_enabled and response_len > 0:
            logits = outputs.logits
            response_start = prompt_len
            
            if logits.size(1) > response_start:
                features.token_probs = compute_token_probs(
                    logits[:, response_start-1:, :],
                    input_ids[:, response_start-1:],
                ).squeeze(0).cpu()
                
                features.token_entropy = compute_token_entropy(
                    logits[:, response_start:, :]
                ).squeeze(0).cpu()
                
                if features.token_probs.numel() > 0:
                    features.perplexity = torch.exp(
                        -torch.log(features.token_probs.clamp(min=1e-10)).mean()
                    ).item()
        
        # 清理
        del outputs, input_ids
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Token级幻觉标签
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
        """使用生成模式提取特征。"""
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
    
    # =========================================================================
    # 批量提取
    # =========================================================================
    
    def extract_batch(
        self,
        samples: List[Sample],
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> List[ExtractedFeatures]:
        """批量提取特征。
        
        根据 batch_size 自动选择最优策略：
        - batch_size=1: 逐样本处理（兼容模式）
        - batch_size>1: 多样本并行前向传播（高效模式）
        
        Args:
            samples: 输入样本列表
            batch_size: 批次大小，设为 >1 启用真正的批处理
            show_progress: 是否显示进度
            
        Returns:
            提取的特征列表
        """
        if batch_size <= 1:
            return self._extract_sequential(samples, show_progress)
        else:
            return self._extract_batch(samples, batch_size, show_progress)
    
    def extract_streaming(
        self,
        samples: List[Sample],
        output_dir: Path,
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """流式提取：边提取边保存，每个样本处理完立即释放内存。
        
        Args:
            samples: 输入样本列表
            output_dir: 保存目录
            batch_size: 批次大小（按传入参数）
            show_progress: 是否显示进度
            
        Returns:
            统计信息 {"success": int, "failed": int, "skipped": int}
        """
        import gc
        from utils.async_saver import MemoryEfficientSaver
        
        output_dir = Path(output_dir)
        features_dir = output_dir / "features_individual"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {"success": 0, "failed": 0, "skipped": 0}
        device = get_model_device(self.model.model)
        
        # 异步保存器
        saver = MemoryEfficientSaver(output_dir, max_workers=2)
        
        n_batches = (len(samples) + batch_size - 1) // batch_size
        
        if show_progress:
            logger.info(f"Streaming extraction: {len(samples)} samples, batch_size={batch_size}")
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]
            
            if show_progress and batch_idx % 10 == 0:
                logger.info(f"Processing batch {batch_idx+1}/{n_batches}")
            
            # 提取当前 batch
            try:
                if batch_size > 1:
                    batch_features = self._process_batch(batch_samples, device)
                else:
                    batch_features = [self.extract(batch_samples[0])]
            except Exception as e:
                logger.warning(f"Batch {batch_idx} failed: {e}, falling back to sequential")
                batch_features = []
                for sample in batch_samples:
                    try:
                        batch_features.append(self.extract(sample))
                    except Exception as e2:
                        logger.warning(f"Sample {sample.id} failed: {e2}")
                        stats["failed"] += 1
            
            # 每个样本立即保存并释放
            for features in batch_features:
                try:
                    save_path = features_dir / f"{features.sample_id}.pt"
                    if save_path.exists():
                        stats["skipped"] += 1
                    else:
                        save_data = self._features_to_dict(features)
                        saver.save_and_release(features.sample_id, save_data)
                        stats["success"] += 1
                except Exception as e:
                    logger.warning(f"Failed to save {features.sample_id}: {e}")
                    stats["failed"] += 1
                
                # 每个样本处理完立即释放
                del features
                gc.collect()
                torch.cuda.empty_cache()
            
            del batch_features
            clear_gpu_memory()
        
        # 等待异步保存完成
        logger.info("Waiting for async saves...")
        saver.finalize()
        saver.shutdown()
        
        logger.info(f"Done: {stats}")
        return stats
    
    def _features_to_dict(self, features: ExtractedFeatures) -> Dict[str, Any]:
        """将 ExtractedFeatures 转换为可保存的字典。"""
        result = {
            "sample_id": features.sample_id,
            "prompt_len": features.prompt_len,
            "response_len": features.response_len,
            "label": features.label,
            "layers": features.layers,
            "model_name": features.model_name,
            "mode": features.mode.value if hasattr(features.mode, 'value') else str(features.mode),
        }
        
        # 保存张量
        tensor_fields = [
            "attn_diags", "laplacian_diags", "attn_entropy", "full_attention",
            "hidden_states", "token_probs", "token_entropy"
        ]
        
        for field in tensor_fields:
            value = getattr(features, field, None)
            if value is not None:
                # 确保在 CPU 上
                if isinstance(value, torch.Tensor):
                    result[field] = value.detach().cpu()
                else:
                    result[field] = value
        
        if features.perplexity is not None:
            result["perplexity"] = features.perplexity
        
        if features.hallucination_labels is not None:
            result["hallucination_labels"] = features.hallucination_labels
        
        if features.hallucination_token_spans is not None:
            result["hallucination_token_spans"] = features.hallucination_token_spans
        
        return result

    def _extract_sequential(
        self,
        samples: List[Sample],
        show_progress: bool = True,
    ) -> List[ExtractedFeatures]:
        """逐样本提取（兼容模式）。"""
        features_list = []
        
        if show_progress:
            pbar = Progress(len(samples), desc="Extracting features")
        
        for sample in samples:
            try:
                features = self.extract(sample)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract features for {sample.id}: {e}")
            
            clear_gpu_memory()
            
            if show_progress:
                pbar.update()
        
        return features_list
    
    @torch.inference_mode()
    def _extract_batch(
        self,
        samples: List[Sample],
        batch_size: int,
        show_progress: bool = True,
    ) -> List[ExtractedFeatures]:
        """真正的批处理提取（高效模式）。
        
        多样本并行前向传播，充分利用 GPU 算力。
        """
        all_features = []
        device = get_model_device(self.model.model)
        
        n_batches = (len(samples) + batch_size - 1) // batch_size
        
        if show_progress:
            logger.info(f"Batch extraction: {len(samples)} samples, batch_size={batch_size}, {n_batches} batches")
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]
            
            if show_progress and batch_idx % 10 == 0:
                logger.info(f"Processing batch {batch_idx+1}/{n_batches}")
            
            try:
                batch_features = self._process_batch(batch_samples, device)
                all_features.extend(batch_features)
            except Exception as e:
                logger.warning(f"Batch {batch_idx} failed: {e}, falling back to sequential")
                for sample in batch_samples:
                    try:
                        features = self.extract(sample)
                        all_features.append(features)
                    except Exception as e2:
                        logger.warning(f"Sample {sample.id} failed: {e2}")
            
            clear_gpu_memory()
        
        return all_features
    
    def _tokenize_batch(
        self,
        samples: List[Sample],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[BatchSampleInfo]]:
        """批量分词并 pad。"""
        all_input_ids = []
        sample_infos = []
        
        pad_token_id = self.model.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.model.tokenizer.eos_token_id or 0
        
        for sample in samples:
            prompt_ids = self.model.encode(sample.prompt, add_special_tokens=True)
            response_ids = self.model.encode(sample.response, add_special_tokens=False)
            
            prompt_len = prompt_ids.size(1)
            response_len = response_ids.size(1)
            original_total_len = prompt_len + response_len
            
            input_ids = torch.cat([prompt_ids, response_ids], dim=1)
            
            if input_ids.size(1) > self.config.max_length:
                input_ids = input_ids[:, :self.config.max_length]
                # ⚠️ 修复：同时调整 prompt_len 和 response_len
                if prompt_len > self.config.max_length:
                    logger.warning(
                        f"⚠️ Sample {sample.id}: prompt_len ({prompt_len}) > max_length ({self.config.max_length}). "
                        f"Response completely truncated! Total tokens: {original_total_len}"
                    )
                    prompt_len = self.config.max_length
                    response_len = 0
                else:
                    response_len = max(0, self.config.max_length - prompt_len)
            
            # 验证response_len
            if response_len == 0:
                logger.warning(
                    f"⚠️ Sample {sample.id} has response_len=0 in batch processing."
                )
            
            all_input_ids.append(input_ids.squeeze(0))
            
            sample_infos.append(BatchSampleInfo(
                sample_id=sample.id,
                prompt_len=prompt_len,
                response_len=response_len,
                total_len=input_ids.size(1),
                label=sample.label,
                sample=sample,
            ))
        
        # Pad to same length
        input_ids_padded = pad_sequence(
            all_input_ids,
            batch_first=True,
            padding_value=pad_token_id
        )
        
        # Create attention mask
        attention_mask = torch.zeros_like(input_ids_padded)
        for i, info in enumerate(sample_infos):
            attention_mask[i, :info.total_len] = 1
        
        return input_ids_padded, attention_mask, sample_infos
    
    def _process_batch(
        self,
        samples: List[Sample],
        device: torch.device,
    ) -> List[ExtractedFeatures]:
        """处理一个批次。"""
        # 批量分词
        input_ids, attention_mask, sample_infos = self._tokenize_batch(samples)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # 批量前向传播
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.config.attention_enabled,
            output_hidden_states=self.config.hidden_states_enabled,
            return_dict=True,
        )
        
        # 为每个样本提取特征
        features_list = []
        for i, info in enumerate(sample_infos):
            features = self._extract_from_batch_outputs(
                outputs=outputs,
                input_ids=input_ids,
                batch_idx=i,
                info=info,
            )
            features_list.append(features)
        
        del outputs, input_ids, attention_mask
        
        return features_list
    
    def _extract_from_batch_outputs(
        self,
        outputs,
        input_ids: torch.Tensor,
        batch_idx: int,
        info: BatchSampleInfo,
    ) -> ExtractedFeatures:
        """从批处理输出中提取单个样本的特征。"""
        seq_len = info.total_len
        prompt_len = info.prompt_len
        response_len = info.response_len
        
        features = ExtractedFeatures(
            sample_id=info.sample_id,
            prompt_len=prompt_len,
            response_len=response_len,
            label=info.label,
            layers=self.attn_layers or self.hidden_layers,
            model_name=self.model.config.name,
            mode=ExtractionMode.TEACHER_FORCING,
        )
        
        # 提取注意力特征
        if self.config.attention_enabled and outputs.attentions is not None:
            attn_diags = []
            attn_row_sums = []
            lap_diags = []
            attn_entropy_list = []
            full_attention_list = [] if self._store_full_attention else None
            
            for layer_idx in self.attn_layers:
                if layer_idx < len(outputs.attentions):
                    # [1, n_heads, seq, seq]
                    attn = outputs.attentions[layer_idx][batch_idx:batch_idx+1, :, :seq_len, :seq_len]
                    
                    attn_diags.append(extract_attention_diagonal(attn).squeeze(0).cpu())
                    attn_row_sums.append(compute_attention_row_sums(attn).squeeze(0).cpu())
                    lap_diags.append(compute_laplacian_diagonal(attn).squeeze(0).cpu())
                    attn_entropy_list.append(compute_attention_entropy(attn).squeeze(0).cpu())
                    
                    if self._store_full_attention:
                        full_attention_list.append(attn.squeeze(0).cpu())
            
            if attn_diags:
                features.attn_diags = torch.stack(attn_diags, dim=0)
                features.attn_row_sums = torch.stack(attn_row_sums, dim=0)
                features.laplacian_diags = torch.stack(lap_diags, dim=0)
                features.attn_entropy = torch.stack(attn_entropy_list, dim=0)
                
                if self._store_full_attention and full_attention_list:
                    features.full_attention = torch.stack(full_attention_list, dim=0)
        
        # 提取隐藏状态
        if self.config.hidden_states_enabled and outputs.hidden_states is not None:
            hs_list = []
            
            for layer_idx in self.hidden_layers:
                if layer_idx < len(outputs.hidden_states):
                    hs = outputs.hidden_states[layer_idx][batch_idx:batch_idx+1, :seq_len, :]
                    pooled = pool_hidden_states(hs, self.config.hidden_states_pooling)
                    hs_list.append(pooled.squeeze(0).cpu())
            
            if hs_list:
                features.hidden_states = torch.stack(hs_list, dim=0)
        
        # Token概率和熵
        if self.config.token_probs_enabled and response_len > 0:
            logits = outputs.logits[batch_idx:batch_idx+1, :seq_len, :]
            sample_input_ids = input_ids[batch_idx:batch_idx+1, :seq_len]
            response_start = prompt_len
            
            if logits.size(1) > response_start:
                features.token_probs = compute_token_probs(
                    logits[:, response_start-1:, :],
                    sample_input_ids[:, response_start-1:],
                ).squeeze(0).cpu()
                
                features.token_entropy = compute_token_entropy(
                    logits[:, response_start:, :]
                ).squeeze(0).cpu()
                
                if features.token_probs.numel() > 0:
                    features.perplexity = torch.exp(
                        -torch.log(features.token_probs.clamp(min=1e-10)).mean()
                    ).item()
        
        # Token级幻觉标签
        try:
            span_labels, has_spans = extract_hallucination_info_from_sample(info.sample.metadata)
            if has_spans and info.label == 1:
                token_spans = calculate_hallucination_token_spans(
                    labels=span_labels,
                    prompt_text=info.sample.prompt,
                    response_text=info.sample.response,
                    tokenizer=self.model.tokenizer,
                    end_inclusive=False,
                )
                features.hallucination_labels = get_token_hallucination_labels(seq_len, token_spans)
                features.hallucination_token_spans = token_spans
        except Exception as e:
            logger.debug(f"Could not calculate hallucination labels for {info.sample_id}: {e}")
        
        return features


# =============================================================================
# 工厂函数
# =============================================================================

def create_extractor(
    model: LoadedModel,
    config: FeaturesConfig,
    store_full_attention: bool = False,
) -> FeatureExtractor:
    """创建特征提取器。"""
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
    """根据特征需求创建提取器。"""
    if "attention_diags" in feature_requirements:
        config.attention_enabled = feature_requirements.get("attention_diags", False)
    
    if "hidden_states" in feature_requirements:
        config.hidden_states_enabled = feature_requirements.get("hidden_states", False)
    
    if "token_probs" in feature_requirements or "token_entropy" in feature_requirements:
        config.token_probs_enabled = (
            feature_requirements.get("token_probs", False) or
            feature_requirements.get("token_entropy", False)
        )
    
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