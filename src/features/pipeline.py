"""Two-Stage Feature Extraction Pipeline - 两阶段特征提取管线。

Stage 1: 提取并保存基础特征（full_attention, hidden_states, token_probs）
Stage 2: 各方法从基础特征计算自己需要的派生特征

目录结构：
outputs/features/{dataset}/{model}/seed_{seed}/{task_type}/
├── base/
│   ├── full_attention/
│   ├── hidden_states/
│   └── token_probs/
└── derived/
    └── {method_name}/

Usage:
    from src.features.pipeline import FeatureExtractionPipeline
    
    pipeline = FeatureExtractionPipeline(
        methods=["lapeigvals", "lookback_lens"],
        output_dir="outputs/features/ragtruth/mistral_7b/seed_42/QA",
    )
    
    # Stage 1: 提取基础特征
    pipeline.extract_base_features(model, samples)
    
    # Stage 2: 计算派生特征
    pipeline.compute_derived_features()
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum, auto
import logging
import json
import torch
import numpy as np
from tqdm import tqdm

from .feature_registry import (
    get_method_requirements,
    compute_union_requirements,
    get_base_features_needed,
    should_store_full_attention,
    describe_requirements,
    MethodFeatureRequirements,
    BaseFeatureType,
    DerivedFeatureType,
    FeatureScope,
    LayerSelection,
)

# Base extractors
from .base.attention import (
    AttentionExtractor,
    AttentionExtractionConfig,
    extract_attention_for_layers,
)
from .base.hidden_states import (
    HiddenStatesExtractor,
    HiddenStatesExtractionConfig,
    extract_hidden_states_for_layers,
)
from .base.token_probs import (
    TokenProbsExtractor,
    TokenProbsExtractionConfig,
)

# Derived extractors - 从合并文件导入
from .derived import (
    # Attention-based
    compute_attention_diags,
    compute_attention_diags_direct,
    compute_laplacian_diags,
    compute_laplacian_diags_direct,
    compute_attention_entropy,
    compute_attention_entropy_direct,
    compute_lookback_ratio,
    compute_lookback_ratio_direct,
    compute_mva_features,
    compute_mva_features_direct,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 配置
# =============================================================================

@dataclass
class PipelineConfig:
    """管线配置。"""
    methods: List[str] = field(default_factory=list)
    output_dir: str = "outputs/features"
    
    # Stage 1 配置
    save_base_features: bool = True           # 是否保存基础特征
    force_full_attention: bool = False        # 强制存储完整注意力
    base_feature_format: str = "pt"           # pt, npz, safetensors
    
    # Stage 2 配置
    compute_derived_on_fly: bool = True       # 是否即时计算派生特征
    save_derived_features: bool = True        # 是否保存派生特征
    
    # 内存管理
    batch_size: int = 1                       # 批量大小
    clear_cache_every: int = 10               # 每 N 个样本清理一次缓存
    
    # 设备
    device: str = "cuda"


# =============================================================================
# 主管线
# =============================================================================

class FeatureExtractionPipeline:
    """两阶段特征提取管线。"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        
        # 计算需求并集
        self.requirements = compute_union_requirements(config.methods)
        self.base_features_needed = get_base_features_needed(self.requirements)
        self.store_full_attention = (
            config.force_full_attention or 
            should_store_full_attention(self.requirements)
        )
        
        # 创建目录
        self._setup_directories()
        
        # 打印需求摘要
        logger.info(describe_requirements(self.requirements))
        logger.info(f"Will store full_attention: {self.store_full_attention}")
    
    def _setup_directories(self):
        """创建输出目录结构。"""
        # Base feature directories
        for bf_type in self.base_features_needed.keys():
            (self.output_dir / "base" / bf_type.value).mkdir(parents=True, exist_ok=True)
        
        # Derived feature directories
        for method in self.config.methods:
            (self.output_dir / "derived" / method).mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        self._save_metadata()
    
    def _save_metadata(self):
        """保存管线元数据。"""
        metadata = {
            "methods": self.config.methods,
            "requirements": self.requirements.to_dict(),
            "store_full_attention": self.store_full_attention,
            "base_features_needed": [bf.value for bf in self.base_features_needed.keys()],
        }
        
        with open(self.output_dir / "pipeline_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    # =========================================================================
    # Stage 1: 基础特征提取
    # =========================================================================
    
    def extract_base_features(
        self,
        model,
        samples: List[Any],
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, int]:
        """Stage 1: 提取基础特征。
        
        Args:
            model: 模型包装器
            samples: 样本列表
            progress_callback: 进度回调
            
        Returns:
            统计信息
        """
        stats = {
            "total": len(samples),
            "success": 0,
            "failed": 0,
            "skipped": 0,
        }
        
        # 确定需要什么
        need_attention = BaseFeatureType.FULL_ATTENTION in self.base_features_needed
        need_hidden = BaseFeatureType.HIDDEN_STATES in self.base_features_needed
        need_probs = BaseFeatureType.TOKEN_PROBS in self.base_features_needed
        
        logger.info(f"Stage 1: Extracting base features for {len(samples)} samples")
        logger.info(f"  - full_attention: {need_attention}")
        logger.info(f"  - hidden_states: {need_hidden}")
        logger.info(f"  - token_probs: {need_probs}")
        
        for idx, sample in enumerate(tqdm(samples, desc="Extracting base features")):
            try:
                # 检查是否已存在
                if self._base_features_exist(sample.id):
                    stats["skipped"] += 1
                    continue
                
                # 提取特征
                base_features = self._extract_single_sample(
                    model, sample,
                    extract_attention=need_attention,
                    extract_hidden=need_hidden,
                    extract_probs=need_probs,
                )
                
                # 保存
                if self.config.save_base_features:
                    self._save_base_features(sample.id, base_features)
                
                stats["success"] += 1
                
                # 内存清理
                if (idx + 1) % self.config.clear_cache_every == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                if progress_callback:
                    progress_callback(idx + 1, len(samples))
                    
            except Exception as e:
                logger.error(f"Failed to extract features for sample {sample.id}: {e}")
                stats["failed"] += 1
        
        logger.info(f"Stage 1 complete: {stats}")
        return stats
    
    def _extract_single_sample(
        self,
        model,
        sample,
        extract_attention: bool,
        extract_hidden: bool,
        extract_probs: bool,
    ) -> Dict[str, Any]:
        """提取单个样本的基础特征。"""
        device = self.config.device
        
        # 分词
        prompt_ids = model.encode(sample.prompt, add_special_tokens=True)
        response_ids = model.encode(sample.response, add_special_tokens=False)
        
        prompt_len = prompt_ids.size(1)
        response_len = response_ids.size(1)
        
        # 拼接
        input_ids = torch.cat([prompt_ids, response_ids], dim=1).to(device)
        seq_len = input_ids.size(1)
        
        # 前向传播
        with torch.no_grad():
            outputs = model.model(
                input_ids=input_ids,
                output_attentions=extract_attention,
                output_hidden_states=extract_hidden,
                return_dict=True,
            )
        
        results = {
            "sample_id": sample.id,
            "prompt_len": prompt_len,
            "response_len": response_len,
            "seq_len": seq_len,
            "label": sample.label,
        }
        
        # 提取 full_attention
        if extract_attention and outputs.attentions is not None:
            if self.store_full_attention:
                # 存储完整注意力
                attn_config = AttentionExtractionConfig(
                    layers=self._get_layer_indices(model),
                    store_on_cpu=True,
                    half_precision=True,
                )
                extractor = AttentionExtractor(attn_config)
                attn_result = extractor.extract(outputs.attentions, prompt_len, response_len)
                results["full_attention"] = attn_result["full_attention"]
            else:
                # 直接计算派生特征（节省内存）
                derived = self._compute_derived_from_attentions(
                    outputs.attentions, prompt_len, response_len
                )
                results.update(derived)
            
            # 清理
            del outputs.attentions
        
        # 提取 hidden_states
        if extract_hidden and outputs.hidden_states is not None:
            hs_config = HiddenStatesExtractionConfig(
                layers=self._get_layer_indices(model),
                store_on_cpu=True,
                half_precision=True,
            )
            extractor = HiddenStatesExtractor(hs_config)
            hs_result = extractor.extract(outputs.hidden_states, prompt_len, response_len)
            results["hidden_states"] = hs_result["hidden_states"]
            
            del outputs.hidden_states
        
        # 提取 token_probs
        if extract_probs and hasattr(outputs, 'logits') and outputs.logits is not None:
            probs_config = TokenProbsExtractionConfig(
                compute_entropy=True,
                compute_perplexity=True,
                response_only=True,
            )
            extractor = TokenProbsExtractor(probs_config)
            probs_result = extractor.extract(
                outputs.logits, input_ids, prompt_len, response_len
            )
            results["token_probs"] = probs_result.get("token_probs")
            results["token_entropy"] = probs_result.get("entropy")
            results["perplexity"] = probs_result.get("perplexity")
        
        return results
    
    def _compute_derived_from_attentions(
        self,
        attentions: tuple,
        prompt_len: int,
        response_len: int,
    ) -> Dict[str, torch.Tensor]:
        """从原始 attentions 直接计算派生特征（不存储 full_attention）。"""
        results = {}
        
        # 根据需求计算派生特征
        for df in self.requirements.derived_features:
            if df.feature_type == DerivedFeatureType.ATTENTION_DIAGS:
                results["attention_diags"] = compute_attention_diags_direct(
                    attentions, 
                    layers=self._resolve_layers(df.layer_selection),
                    prompt_len=prompt_len,
                    response_len=response_len,
                    response_only=(df.scope == FeatureScope.RESPONSE_ONLY),
                )
            
            elif df.feature_type == DerivedFeatureType.LAPLACIAN_DIAGS:
                results["laplacian_diags"] = compute_laplacian_diags_direct(
                    attentions,
                    layers=self._resolve_layers(df.layer_selection),
                    prompt_len=prompt_len,
                    response_len=response_len,
                    response_only=(df.scope == FeatureScope.RESPONSE_ONLY),
                )
            
            elif df.feature_type == DerivedFeatureType.ATTENTION_ENTROPY:
                results["attention_entropy"] = compute_attention_entropy_direct(
                    attentions,
                    layers=self._resolve_layers(df.layer_selection),
                    prompt_len=prompt_len,
                    response_len=response_len,
                    response_only=(df.scope == FeatureScope.RESPONSE_ONLY),
                )
            
            elif df.feature_type == DerivedFeatureType.LOOKBACK_RATIO:
                results["lookback_ratio"] = compute_lookback_ratio_direct(
                    attentions,
                    prompt_len=prompt_len,
                    response_len=response_len,
                    layers=self._resolve_layers(df.layer_selection),
                )
            
            elif df.feature_type == DerivedFeatureType.MVA_FEATURES:
                mva_results = compute_mva_features_direct(
                    attentions,
                    prompt_len=prompt_len,
                    response_len=response_len,
                    layers=self._resolve_layers(df.layer_selection),
                )
                results["mva_avg_in"] = mva_results["avg_in"]
                results["mva_div_in"] = mva_results["div_in"]
                results["mva_div_out"] = mva_results["div_out"]
        
        return results
    
    def _resolve_layers(self, selection: LayerSelection) -> Optional[List[int]]:
        """将层选择策略转换为具体层索引。"""
        # TODO: 需要模型信息来确定层数
        # 这里返回 None 表示使用所有层
        if selection == LayerSelection.ALL:
            return None
        elif selection == LayerSelection.LAST:
            return [-1]
        elif selection == LayerSelection.LAST_4:
            return [-4, -3, -2, -1]
        elif selection == LayerSelection.LAST_HALF:
            return None  # 需要运行时确定
        return None
    
    def _get_layer_indices(self, model) -> Optional[List[int]]:
        """获取要提取的层索引。"""
        # 使用配置中的层或所有层
        return None
    
    def _base_features_exist(self, sample_id: str) -> bool:
        """检查基础特征是否已存在。"""
        for bf_type in self.base_features_needed.keys():
            feature_path = self.output_dir / "base" / bf_type.value / f"{sample_id}.pt"
            if not feature_path.exists():
                return False
        return True
    
    def _save_base_features(self, sample_id: str, features: Dict[str, Any]):
        """保存基础特征。"""
        # 保存 full_attention
        if "full_attention" in features:
            path = self.output_dir / "base" / "full_attention" / f"{sample_id}.pt"
            torch.save({
                "full_attention": features["full_attention"],
                "prompt_len": features["prompt_len"],
                "response_len": features["response_len"],
                "label": features["label"],
            }, path)
        
        # 保存 hidden_states
        if "hidden_states" in features:
            path = self.output_dir / "base" / "hidden_states" / f"{sample_id}.pt"
            torch.save({
                "hidden_states": features["hidden_states"],
                "prompt_len": features["prompt_len"],
                "response_len": features["response_len"],
                "label": features["label"],
            }, path)
        
        # 保存 token_probs
        if "token_probs" in features:
            path = self.output_dir / "base" / "token_probs" / f"{sample_id}.pt"
            torch.save({
                "token_probs": features["token_probs"],
                "token_entropy": features.get("token_entropy"),
                "perplexity": features.get("perplexity"),
                "prompt_len": features["prompt_len"],
                "response_len": features["response_len"],
                "label": features["label"],
            }, path)
    
    # =========================================================================
    # Stage 2: 派生特征计算
    # =========================================================================
    
    def compute_derived_features(
        self,
        sample_ids: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """Stage 2: 从基础特征计算派生特征。
        
        Args:
            sample_ids: 要处理的样本 ID（None = 处理所有）
            
        Returns:
            每个方法的统计信息
        """
        stats = {method: {"success": 0, "failed": 0} for method in self.config.methods}
        
        # 获取所有样本 ID
        if sample_ids is None:
            sample_ids = self._get_available_sample_ids()
        
        logger.info(f"Stage 2: Computing derived features for {len(sample_ids)} samples")
        
        for sample_id in tqdm(sample_ids, desc="Computing derived features"):
            # 加载基础特征
            base_features = self._load_base_features(sample_id)
            if base_features is None:
                for method in self.config.methods:
                    stats[method]["failed"] += 1
                continue
            
            # 为每个方法计算派生特征
            for method in self.config.methods:
                try:
                    derived = self._compute_method_derived(method, base_features)
                    
                    if self.config.save_derived_features:
                        self._save_derived_features(method, sample_id, derived)
                    
                    stats[method]["success"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to compute {method} features for {sample_id}: {e}")
                    stats[method]["failed"] += 1
        
        logger.info(f"Stage 2 complete: {stats}")
        return stats
    
    def _get_available_sample_ids(self) -> List[str]:
        """获取所有可用的样本 ID。"""
        sample_ids = set()
        
        # 从任一基础特征目录获取
        for bf_type in self.base_features_needed.keys():
            feature_dir = self.output_dir / "base" / bf_type.value
            if feature_dir.exists():
                for path in feature_dir.glob("*.pt"):
                    sample_ids.add(path.stem)
                break
        
        return list(sample_ids)
    
    def _load_base_features(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """加载基础特征。"""
        features = {}
        
        # 加载 full_attention
        attn_path = self.output_dir / "base" / "full_attention" / f"{sample_id}.pt"
        if attn_path.exists():
            data = torch.load(attn_path, map_location="cpu")
            features.update(data)
        
        # 加载 hidden_states
        hs_path = self.output_dir / "base" / "hidden_states" / f"{sample_id}.pt"
        if hs_path.exists():
            data = torch.load(hs_path, map_location="cpu")
            features["hidden_states"] = data["hidden_states"]
        
        # 加载 token_probs
        probs_path = self.output_dir / "base" / "token_probs" / f"{sample_id}.pt"
        if probs_path.exists():
            data = torch.load(probs_path, map_location="cpu")
            features["token_probs"] = data.get("token_probs")
            features["token_entropy"] = data.get("token_entropy")
        
        return features if features else None
    
    def _compute_method_derived(
        self,
        method: str,
        base_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """为特定方法计算派生特征。"""
        req = get_method_requirements(method)
        results = {
            "method": method,
            "prompt_len": base_features.get("prompt_len", 0),
            "response_len": base_features.get("response_len", 0),
            "label": base_features.get("label"),
        }
        
        full_attn = base_features.get("full_attention")
        prompt_len = results["prompt_len"]
        response_len = results["response_len"]
        
        # 计算各派生特征
        for df in req.derived_features:
            if df.feature_type == DerivedFeatureType.ATTENTION_DIAGS and full_attn is not None:
                results["attention_diags"] = compute_attention_diags(
                    full_attn,
                    prompt_len=prompt_len,
                    response_len=response_len,
                    response_only=(df.scope == FeatureScope.RESPONSE_ONLY),
                )
            
            elif df.feature_type == DerivedFeatureType.LAPLACIAN_DIAGS and full_attn is not None:
                results["laplacian_diags"] = compute_laplacian_diags(
                    full_attn,
                    prompt_len=prompt_len,
                    response_len=response_len,
                    response_only=(df.scope == FeatureScope.RESPONSE_ONLY),
                )
            
            elif df.feature_type == DerivedFeatureType.ATTENTION_ENTROPY and full_attn is not None:
                results["attention_entropy"] = compute_attention_entropy(
                    full_attn,
                    prompt_len=prompt_len,
                    response_len=response_len,
                    response_only=(df.scope == FeatureScope.RESPONSE_ONLY),
                )
            
            elif df.feature_type == DerivedFeatureType.LOOKBACK_RATIO and full_attn is not None:
                results["lookback_ratio"] = compute_lookback_ratio(
                    full_attn,
                    prompt_len=prompt_len,
                    response_len=response_len,
                )
            
            elif df.feature_type == DerivedFeatureType.MVA_FEATURES and full_attn is not None:
                mva = compute_mva_features(
                    full_attn,
                    prompt_len=prompt_len,
                    response_len=response_len,
                )
                results["mva_features"] = mva
        
        return results
    
    def _save_derived_features(
        self,
        method: str,
        sample_id: str,
        features: Dict[str, Any],
    ):
        """保存派生特征。"""
        path = self.output_dir / "derived" / method / f"{sample_id}.pt"
        torch.save(features, path)


# =============================================================================
# 便捷函数
# =============================================================================

def create_pipeline(
    methods: List[str],
    output_dir: str,
    **kwargs,
) -> FeatureExtractionPipeline:
    """创建特征提取管线的便捷函数。"""
    config = PipelineConfig(
        methods=methods,
        output_dir=output_dir,
        **kwargs,
    )
    return FeatureExtractionPipeline(config)


def analyze_method_requirements(methods: List[str]) -> str:
    """分析方法需求并返回人类可读的报告。"""
    union_req = compute_union_requirements(methods)
    return describe_requirements(union_req)
