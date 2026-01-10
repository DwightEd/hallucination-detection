#!/usr/bin/env python3
"""Generate activations (features) for hallucination detection.

Usage:
    # 训练集
    python scripts/generate_activations.py dataset.name=ragtruth dataset.split_name=train
    
    # 测试集
    python scripts/generate_activations.py dataset.name=ragtruth dataset.split_name=test
"""
import sys
import re
import json
import logging
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    DatasetConfig, ModelConfig, FeaturesConfig, Sample, ExtractedFeatures,
    set_seed, setup_logging, TaskType, SplitType,
)
from src.models import get_model, unload_all_models
from src.features import create_extractor, create_extractor_from_requirements
from src.utils import (
    clear_gpu_memory, log_gpu_memory, MemoryTracker,
    get_device_info, get_all_cuda_devices, synchronize_all_gpus,
    is_multi_gpu_model, log_device_distribution,
)

from utils.checkpoint import CheckpointManager, get_pending_samples
from utils.async_saver import MemoryEfficientSaver
from utils.feature_manager import (
    FeatureManager, compute_union_requirements, create_feature_manager
)

logger = logging.getLogger(__name__)


def log_system_info():
    """记录系统和GPU信息。"""
    logger.info("=" * 70)
    logger.info("System Information")
    logger.info("=" * 70)
    
    logger.info(f"PyTorch version: {torch.__version__}")
    
    device_info = get_device_info()
    
    if device_info["cuda_available"]:
        logger.info(f"CUDA version: {device_info.get('cuda_version', 'N/A')}")
        logger.info(f"GPU count: {device_info['device_count']}")
        
        for i in range(device_info['device_count']):
            dev_key = f"device_{i}"
            if dev_key in device_info:
                dev = device_info[dev_key]
                logger.info(
                    f"  GPU {i}: {dev['name']} | "
                    f"Total: {dev['total_memory_gb']:.1f}GB | "
                    f"Free: {dev.get('free_memory_gb', 'N/A'):.1f}GB"
                )
    else:
        logger.warning("CUDA not available - using CPU")
    
    logger.info("=" * 70)


def log_model_filter_info(cfg: DictConfig, samples: List[Sample]):
    """记录模型过滤信息。"""
    source_models = set()
    for sample in samples:
        src_model = sample.metadata.get("source_model")
        if src_model:
            source_models.add(src_model)
    
    model_filter = cfg.dataset.get("models", None)
    
    logger.info("-" * 50)
    logger.info("Data Source Models")
    logger.info("-" * 50)
    
    if model_filter:
        logger.info(f"Filter applied: {model_filter}")
    else:
        logger.info("Filter: None (all models)")
    
    logger.info(f"Models in loaded data: {sorted(source_models)}")
    
    model_counts = {}
    for sample in samples:
        src_model = sample.metadata.get("source_model", "unknown")
        model_counts[src_model] = model_counts.get(src_model, 0) + 1
    
    for model, count in sorted(model_counts.items()):
        logger.info(f"  {model}: {count} samples")
    
    logger.info("-" * 50)


def parse_task_types_to_list(task_types: Any) -> Optional[List[str]]:
    """将各种格式的 task_types 解析为标准列表。"""
    if task_types is None:
        return None
    
    task_str = str(task_types).strip()
    if task_str.lower() in ('null', 'none', '[]', ''):
        return None
    
    if isinstance(task_types, (list, ListConfig)):
        if len(task_types) == 0:
            return None
        cleaned = [str(t).strip().strip("'\"") for t in task_types]
        return [c for c in cleaned if c]
    
    if task_str.startswith('[') and task_str.endswith(']'):
        inner = task_str[1:-1].strip()
        if not inner:
            return None
        parts = re.split(r'[,\s]+', inner)
        cleaned = [p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")]
        return cleaned if cleaned else None
    
    return [task_str.strip("'\"")]


def get_task_types_from_config(cfg: DictConfig) -> Optional[List[str]]:
    """从配置中获取 task_types 列表。"""
    task_type = cfg.dataset.get('task_type', None)
    if task_type is not None:
        parsed = parse_task_types_to_list(task_type)
        if parsed:
            return parsed
    
    task_types = cfg.dataset.get('task_types', None)
    return parse_task_types_to_list(task_types)


def get_task_suffix(cfg: DictConfig) -> str:
    """获取用于目录名的 task 后缀。"""
    task_types = get_task_types_from_config(cfg)
    
    if task_types is None or len(task_types) == 0:
        return "all"
    
    return "_".join(task_types)


def get_model_short_name(cfg: DictConfig) -> str:
    """获取模型短名称。"""
    if hasattr(cfg.model, 'short_name') and cfg.model.short_name:
        return cfg.model.short_name
    return cfg.model.name.split("/")[-1]


def build_output_dir(cfg: DictConfig, split_name: str = "train") -> Path:
    """构建输出目录路径。"""
    base_dir = Path(cfg.get("features_dir", "outputs/features"))
    dataset_name = cfg.dataset.name
    
    task_suffix = get_task_suffix(cfg)
    
    if split_name == "test":
        task_suffix = f"{task_suffix}_test"
    
    model_name = get_model_short_name(cfg)
    
    return base_dir / dataset_name / model_name / f"seed_{cfg.seed}" / task_suffix


def load_samples_from_splits(
    dataset_name: str,
    task_types_filter: Optional[List[str]] = None,
    model_filter: Optional[List[str]] = None,
    split: str = "train"
) -> List[Sample]:
    """从 outputs/splits/{dataset}/train.json 或 test.json 加载样本。"""
    splits_dir = Path("outputs/splits") / dataset_name
    split_file = splits_dir / f"{split}.json"
    
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            f"Please run 'dvc repro split_dataset' first."
        )
    
    logger.info(f"Loading samples from {split_file}")
    
    with open(split_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        item_task_type = item.get('task_type', '')
        if task_types_filter:
            if item_task_type not in task_types_filter:
                continue
        
        if model_filter:
            item_model = item.get('model', '')
            matched = False
            for filter_model in model_filter:
                if filter_model.lower() in item_model.lower():
                    matched = True
                    break
            if not matched:
                continue
        
        task_type = None
        if item_task_type:
            try:
                task_type = TaskType(item_task_type)
            except (ValueError, KeyError):
                pass
        
        split_type = None
        split_str = item.get('split', '')
        if split_str:
            split_str_lower = split_str.lower()
            if split_str_lower in ['train', 'training']:
                split_type = SplitType.TRAIN
            elif split_str_lower in ['test', 'testing', 'val', 'validation']:
                split_type = SplitType.TEST
        
        metadata = item.get('metadata', {})
        sample = Sample(
            id=str(item.get('id', '')),
            prompt=item.get('prompt', ''),
            response=item.get('response', ''),
            label=item.get('label', 0),
            task_type=task_type,
            split=split_type,
            metadata={
                'source_model': item.get('model', ''),
                'hallucination_spans': metadata.get('hallucination_spans', []),
                'source_id': metadata.get('source_id', ''),
            }
        )
        samples.append(sample)
    
    logger.info(f"Loaded {len(samples)} samples from {split} split (after filtering)")
    return samples


def extract_features_dict(features: ExtractedFeatures) -> Dict[str, Any]:
    """将ExtractedFeatures转换为字典格式。"""
    result = {
        "sample_id": features.sample_id,
        "prompt_len": features.prompt_len,
        "response_len": features.response_len,
        "label": features.label,
        "layers": features.layers,
    }
    
    for attr in ['attn_diags', 'laplacian_diags', 'attn_entropy', 
                 'hidden_states', 'token_probs', 'token_entropy']:
        value = getattr(features, attr, None)
        if value is not None:
            result[attr] = value
    
    if features.full_attention is not None:
        result['full_attentions'] = features.full_attention
    
    return result


def save_sample_answer(sample: Sample, features: ExtractedFeatures) -> Dict[str, Any]:
    """创建样本答案条目。"""
    return {
        "id": sample.id,
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
        "task_type": sample.task_type.value if sample.task_type else None,
        "split": sample.split.value if sample.split else None,
        "source_model": sample.metadata.get("source_model"),
        "labels": sample.metadata.get("hallucination_spans", []),
        "prompt_len": features.prompt_len,
        "response_len": features.response_len,
    }


class ProgressTracker:
    """进度跟踪器。"""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = datetime.now()
    
    def update(self, n: int = 1):
        self.current += n
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / max(elapsed, 1)
        remaining = (self.total - self.current) / max(rate, 0.001)
        
        pct = 100 * self.current / max(self.total, 1)
        logger.info(
            f"[{self.desc}] {self.current}/{self.total} ({pct:.1f}%) "
            f"| {rate:.2f} samples/sec | ETA: {remaining:.0f}s"
        )


def compute_stable_config_hash(cfg: DictConfig, split_name: str) -> Dict[str, Any]:
    """计算稳定的配置哈希字典。"""
    return {
        "dataset_name": cfg.dataset.name,
        "task_type": get_task_suffix(cfg),
        "model_name": get_model_short_name(cfg),
        "seed": cfg.seed,
        "methods": sorted(list(cfg.get("methods", []))),
        "allow_full_attention": cfg.get("allow_full_attention", False),
        "features_mode": cfg.features.get("mode", "teacher_forcing"),
        "max_length": cfg.features.get("max_length", 4096),
        "split_name": split_name,
    }


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """内存优化的特征提取主入口。"""
    setup_logging(level=logging.INFO)
    set_seed(cfg.seed)
    
    # 从 dataset 配置获取 split_name（默认为train）
    split_name = cfg.dataset.get("split_name", "train")
    
    logger.info("=" * 70)
    logger.info(f"Generate Activations (Memory-Optimized) - Split: {split_name}")
    logger.info("=" * 70)
    
    log_system_info()
    
    output_dir = build_output_dir(cfg, split_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    checkpoint_manager = CheckpointManager(output_dir)
    saver = MemoryEfficientSaver(output_dir, max_workers=2)
    
    dataset_name = cfg.dataset.name
    task_types_filter = get_task_types_from_config(cfg)
    
    model_filter = cfg.dataset.get("models", None)
    if isinstance(model_filter, ListConfig):
        model_filter = list(model_filter)
    
    logger.info(f"Loading dataset: {dataset_name}")
    logger.info(f"Task types filter: {task_types_filter}")
    logger.info(f"Model filter: {model_filter}")
    logger.info(f"Split: {split_name}")
    
    try:
        all_samples = load_samples_from_splits(
            dataset_name=dataset_name,
            task_types_filter=task_types_filter,
            model_filter=model_filter,
            split=split_name
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    if len(all_samples) == 0:
        logger.error("No samples loaded! Check your filters (dataset.models, dataset.task_types)")
        return
    
    logger.info(f"Loaded {len(all_samples)} samples")
    
    log_model_filter_info(cfg, all_samples)
    
    config_for_hash = compute_stable_config_hash(cfg, split_name)
    resume = cfg.get("resume", True)
    is_resume = checkpoint_manager.initialize(
        total_samples=len(all_samples),
        config=config_for_hash,
        force_restart=not resume
    )
    
    pending_samples = get_pending_samples(all_samples, checkpoint_manager)
    logger.info(
        f"Pending samples: {len(pending_samples)} "
        f"(skipping {len(all_samples) - len(pending_samples)} already processed)"
    )
    
    if len(pending_samples) == 0:
        logger.info("All samples already processed!")
        finalize_outputs(output_dir, checkpoint_manager, all_samples, cfg)
        return
    
    OmegaConf.save(cfg, output_dir / "config.yaml")
    
    methods = cfg.get("methods", ["lapeigvals"])
    if isinstance(methods, str):
        methods = [methods]
    
    allow_full_attention = cfg.get("allow_full_attention", False)
    
    feature_manager = create_feature_manager(
        methods=methods,
        allow_full_attention=allow_full_attention,
    )
    
    logger.info(feature_manager.describe())
    
    max_length = cfg.features.get("max_length", 4096)
    mem_estimate = feature_manager.estimate_memory_per_sample(
        seq_len=max_length,
        n_layers=cfg.model.get("n_layers", 32),
        n_heads=cfg.model.get("n_heads", 32),
        hidden_size=cfg.model.get("hidden_size", 4096),
    )
    
    logger.info(f"Estimated memory per sample: {mem_estimate['total_mb']:.1f} MB")
    
    if mem_estimate['total_gb'] > 10:
        logger.warning(
            f"⚠️ High memory usage expected: {mem_estimate['total_gb']:.1f} GB/sample"
        )
    
    model_config = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    logger.info(f"Loading model: {model_config.name}")
    log_gpu_memory("Before model load")
    
    model = get_model(model_config)
    log_gpu_memory("After model load")
    
    if is_multi_gpu_model(model.model):
        logger.info("Model loaded in multi-GPU mode")
        log_device_distribution(model.model, logger)
    else:
        logger.info(f"Model loaded on: {model.get_device()}")
    
    features_config_dict = OmegaConf.to_container(cfg.features, resolve=True)
    features_config_dict.update(feature_manager.to_features_config())
    features_config = FeaturesConfig(**features_config_dict)
    
    extractor = create_extractor_from_requirements(
        model=model,
        config=features_config,
        feature_requirements=feature_manager.get_combined_requirements().to_dict(),
        allow_full_attention=allow_full_attention,
    )
    
    extractor_mem = extractor.get_memory_estimate(seq_len=max_length)
    logger.info(f"Extractor memory estimate: {extractor_mem['total_mb']:.1f} MB/sample")
    
    logger.info(f"Extracting features (batch_size=1, async_save=True)...")
    progress = ProgressTracker(len(pending_samples), "Extracting")
    
    answers = []
    
    for sample in pending_samples:
        try:
            with MemoryTracker(f"Sample {sample.id}"):
                features = extractor.extract(sample)
            
            features_dict = extract_features_dict(features)
            metadata = {
                "model_name": model_config.name,
                "label": sample.label,
                "source_model": sample.metadata.get("source_model"),
            }
            
            saver.save_and_release(sample.id, features_dict, metadata, force_gc=True)
            
            answers.append(save_sample_answer(sample, features))
            
            checkpoint_manager.mark_completed(sample.id)
            
            del features
            
        except Exception as e:
            logger.warning(f"Failed to process {sample.id}: {e}")
            checkpoint_manager.mark_failed(sample.id, str(e))
        
        progress.update()
        
        if progress.current % 10 == 0:
            clear_gpu_memory()
            
            if torch.cuda.device_count() > 1:
                synchronize_all_gpus()
            
            log_gpu_memory(f"After {progress.current} samples")
    
    logger.info("Waiting for async saves to complete...")
    save_stats = saver.finalize()
    logger.info(f"Save stats: {save_stats}")
    saver.shutdown()
    
    answers_path = output_dir / "answers.json"
    
    existing_answers = []
    if is_resume and answers_path.exists():
        try:
            with open(answers_path) as f:
                existing_answers = json.load(f)
            existing_ids = {a["id"] for a in existing_answers}
            answers = [a for a in answers if a["id"] not in existing_ids]
        except Exception:
            pass
    
    all_answers = existing_answers + answers
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(all_answers, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(all_answers)} answers to {answers_path}")
    
    finalize_outputs(output_dir, checkpoint_manager, all_samples, cfg)
    
    unload_all_models()
    clear_gpu_memory()
    
    logger.info("=" * 70)
    logger.info("Done!")
    logger.info("=" * 70)


def finalize_outputs(
    output_dir: Path,
    checkpoint_manager: CheckpointManager,
    samples: List[Sample],
    cfg: DictConfig
):
    """整合单个特征文件为合并格式。"""
    logger.info("Consolidating features (memory-optimized streaming mode)...")
    
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)
    
    consolidated = checkpoint_manager.consolidate_features_streaming(features_dir)
    
    labels = torch.tensor([s.label if s.label is not None else -1 for s in samples])
    torch.save(labels, output_dir / "labels.pt")
    logger.info(f"Saved labels: {len(labels)} samples")
    
    progress = checkpoint_manager.get_progress()
    metadata = {
        "n_samples": len(samples),
        "n_processed": progress["processed"],
        "n_failed": progress["failed"],
        "sample_ids": consolidated.get("sample_ids", []),
        "saved_features": consolidated.get("saved_features", []),
        "config": OmegaConf.to_container(cfg, resolve=True) if cfg else {},
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Consolidated features saved to {features_dir}")
    
    cleanup_individual = cfg.get("cleanup_individual_features", False)
    if cleanup_individual:
        logger.info("Cleaning up individual feature files...")
        for pt_file in (output_dir / "features_individual").glob("*.pt"):
            try:
                pt_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove {pt_file}: {e}")


if __name__ == "__main__":
    main()
