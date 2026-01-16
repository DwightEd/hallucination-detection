#!/usr/bin/env python3
"""快速评估已训练模型的脚本（修复版）。

修复内容:
1. 正确加载和使用 test split 数据
2. 修复 HaloScope 的 response_start >= seq_len 问题

用法:
    python scripts/quick_eval_fixed.py --methods lapeigvals lookback_lens haloscope hsdmvaf \
                                  --dataset ragtruth --model Mistral-7B-Instruct-v0.3 \
                                  --task_type QA --seed 42
"""
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_model_path(
    models_dir: Path,
    dataset: str,
    model: str,
    seed: int,
    task: str,
    method: str,
    level: str = "sample"
) -> Optional[Path]:
    """查找模型文件路径。"""
    possible_paths = [
        models_dir / dataset / model / f"seed_{seed}" / task / method / level / "model.pkl",
        models_dir / dataset / model / f"seed_{seed}" / task / method / "probe" / "model.pkl",
        models_dir / dataset / model / f"seed_{seed}" / task / method / "model.pkl",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def find_features_dir(
    features_dir: Path,
    dataset: str,
    model: str,
    seed: int,
    task: str,
    split: str = "train"
) -> Optional[Path]:
    """查找特征目录。"""
    # 优先尝试带 split 后缀的目录
    task_suffix = f"{task}_test" if split == "test" else task
    
    possible_paths = [
        features_dir / dataset / model / f"seed_{seed}" / task_suffix,
        features_dir / dataset / model / f"seed_{seed}" / task,
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def load_split_ids(splits_dir: Path, dataset: str, split: str = "test") -> Optional[Set[str]]:
    """从 outputs/splits 加载测试集 ID。
    
    Args:
        splits_dir: outputs/splits 目录
        dataset: 数据集名称
        split: 'train' 或 'test'
        
    Returns:
        sample IDs 集合，如果文件不存在返回 None
    """
    split_file = splits_dir / dataset / f"{split}.json"
    if not split_file.exists():
        logger.debug(f"Split file not found: {split_file}")
        return None
    
    try:
        with open(split_file) as f:
            data = json.load(f)
        
        ids = set()
        for item in data:
            sample_id = item.get("id")
            if sample_id:
                ids.add(str(sample_id))
        
        logger.info(f"Loaded {len(ids)} {split} IDs from {split_file}")
        return ids
    except Exception as e:
        logger.warning(f"Failed to load split file {split_file}: {e}")
        return None


def load_features_and_labels(features_dir: Path, test_ids: Optional[Set[str]] = None) -> tuple:
    """加载特征和标签。
    
    Args:
        features_dir: 特征目录
        test_ids: 测试集 sample IDs，如果提供则只返回这些样本
        
    Returns:
        (features_list, samples) 或 (test_features, train_features, samples)
    """
    import torch
    from src.core import ExtractedFeatures, Sample, TaskType, SplitType
    
    metadata_path = features_dir / "metadata.json"
    answers_path = features_dir / "answers.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {features_dir}")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    samples = []
    if answers_path.exists():
        with open(answers_path) as f:
            answers = json.load(f)
        for ans in answers:
            task_type = None
            if ans.get("task_type"):
                try:
                    task_type = TaskType(ans["task_type"])
                except (ValueError, KeyError):
                    pass
            
            # 解析 split 字段
            split = None
            split_str = ans.get("split", "")
            if split_str:
                split_lower = split_str.lower()
                if split_lower in ['test', 'testing', 'val', 'validation', 'dev']:
                    split = SplitType.TEST
                elif split_lower in ['train', 'training']:
                    split = SplitType.TRAIN
            
            # 如果 answers.json 没有 split 字段，根据 test_ids 判断
            sample_id = str(ans["id"])
            if split is None and test_ids is not None:
                if sample_id in test_ids:
                    split = SplitType.TEST
                else:
                    split = SplitType.TRAIN
            
            samples.append(Sample(
                id=sample_id,
                prompt=ans.get("prompt", ""),
                response=ans.get("response", ""),
                label=ans.get("label", 0),
                task_type=task_type,
                split=split,
                metadata={
                    "prompt_len": ans.get("prompt_len", 0),
                    "response_len": ans.get("response_len", 0),
                }
            ))
    
    # 加载特征文件
    features_subdir = features_dir / "features"
    if not features_subdir.exists():
        features_subdir = features_dir
    
    feature_files = {
        "attn_diags": "attn_diags.pt",
        "laplacian_diags": "laplacian_diags.pt",
        "attn_entropy": "attn_entropy.pt",
        "token_probs": "token_probs.pt",
        "token_entropy": "token_entropy.pt",
    }
    
    loaded_features = {}
    for key, filename in feature_files.items():
        filepath = features_subdir / filename
        if filepath.exists():
            loaded_features[key] = torch.load(filepath, weights_only=False)
    
    # 衍生特征
    if "laplacian_diags" not in loaded_features and "attn_diags" in loaded_features:
        attn_diags_data = loaded_features["attn_diags"]
        laplacian_diags_data = {}
        for sid, attn_diag in attn_diags_data.items():
            if isinstance(attn_diag, torch.Tensor):
                laplacian_diags_data[sid] = 1.0 - attn_diag
        if laplacian_diags_data:
            loaded_features["laplacian_diags"] = laplacian_diags_data
    
    # 加载大特征索引
    large_feature_indexes = {}
    for key, filename in [("hidden_states", "hidden_states_index.json"), 
                          ("full_attentions", "full_attentions_index.json")]:
        filepath = features_subdir / filename
        if filepath.exists():
            with open(filepath) as f:
                large_feature_indexes[key] = json.load(f)
    
    # 加载标签
    labels_path = features_dir / "labels.pt"
    if labels_path.exists():
        labels = torch.load(labels_path, weights_only=False)
    else:
        labels = torch.tensor([s.label or 0 for s in samples])
    
    sample_ids = metadata.get("sample_ids", [s.id for s in samples])
    features_list = []
    
    for i, sample_id in enumerate(sample_ids):
        sample_features = {}
        
        for key, data in loaded_features.items():
            if isinstance(data, dict):
                if sample_id in data:
                    sample_features[key] = data[sample_id]
                elif str(sample_id) in data:
                    sample_features[key] = data[str(sample_id)]
        
        feature_paths = {}
        for feature_key in ["hidden_states", "full_attentions"]:
            if feature_key in large_feature_indexes:
                index_data = large_feature_indexes[feature_key]
                if "index" in index_data and sample_id in index_data["index"]:
                    feature_paths[feature_key] = index_data["index"][sample_id]
        
        sample = samples[i] if i < len(samples) else None
        
        features_list.append(ExtractedFeatures(
            sample_id=sample_id,
            prompt_len=sample.metadata.get("prompt_len", 0) if sample else 0,
            response_len=sample.metadata.get("response_len", 0) if sample else 0,
            label=int(labels[i]) if i < len(labels) else 0,
            attn_diags=sample_features.get("attn_diags"),
            laplacian_diags=sample_features.get("laplacian_diags"),
            attn_entropy=sample_features.get("attn_entropy"),
            token_probs=sample_features.get("token_probs"),
            token_entropy=sample_features.get("token_entropy"),
            metadata={"_feature_paths": feature_paths},
        ))
    
    return features_list, samples


def split_by_field(features_list, samples, test_ids: Optional[Set[str]] = None):
    """按split字段分割数据。
    
    Args:
        features_list: 特征列表
        samples: 样本列表
        test_ids: 可选的测试集 ID 集合（优先使用）
    """
    from src.core import SplitType
    
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    
    for feat, sample in zip(features_list, samples):
        label = feat.label if feat.label is not None else (sample.label or 0)
        sample_id = str(feat.sample_id)
        
        is_test = False
        
        # 优先使用 test_ids
        if test_ids is not None:
            is_test = sample_id in test_ids
        # 然后检查 sample.split 字段
        elif sample.split == SplitType.TEST:
            is_test = True
        
        if is_test:
            test_features.append(feat)
            test_labels.append(label)
        else:
            train_features.append(feat)
            train_labels.append(label)
    
    return train_features, train_labels, test_features, test_labels


def evaluate_method(
    method_name: str,
    model_path: Path,
    test_features: List,
    test_labels: List[int],
) -> Dict[str, Any]:
    """评估单个方法。"""
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    from src.methods import create_method
    from src.core import MethodConfig
    
    try:
        method_config = MethodConfig(name=method_name)
        method = create_method(method_name, config=method_config)
        method.load(model_path)
        
        # === 关键修复：对于 HaloScope，我们需要修复 response_start 问题 ===
        if method_name.lower() in ['haloscope', 'halo', 'haloscope_svd']:
            # 临时修改 HaloScope 的配置，使其不依赖 response_start
            # 或者在 predict 之前调整特征
            _fix_haloscope_features(test_features)
        
        predictions = method.predict_batch(test_features)
        scores = np.array([p.score for p in predictions])
        y_true = np.array(test_labels[:len(scores)])
        
        results = {
            "method": method_name,
            "model_path": str(model_path),
            "n_samples": len(scores),
            "n_positive": int(y_true.sum()),
            "n_negative": len(y_true) - int(y_true.sum()),
        }
        
        if len(np.unique(y_true)) >= 2:
            results["auroc"] = float(roc_auc_score(y_true, scores))
            results["aupr"] = float(average_precision_score(y_true, scores))
            y_pred = (scores > 0.5).astype(int)
            results["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        else:
            results["auroc"] = 0.5
            results["aupr"] = 0.0
            results["f1"] = 0.0
            results["warning"] = "Only one class in test set"
        
        return results
        
    except Exception as e:
        import traceback
        return {
            "method": method_name,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _fix_haloscope_features(features_list: List) -> None:
    """修复 HaloScope 特征的 prompt_len 问题。
    
    问题：prompt_len 是原始 prompt 的 token 数，但 hidden_states 可能被截断了。
    修复：检查实际 hidden_states 的 seq_len，并调整 prompt_len。
    """
    import torch
    
    for feat in features_list:
        # 尝试获取 hidden_states 的实际长度
        hidden_states = feat.hidden_states
        if hidden_states is None:
            hidden_states = feat.get_hidden_states()
        
        if hidden_states is not None:
            if isinstance(hidden_states, torch.Tensor):
                # hidden_states shape: [n_layers, seq_len, hidden_dim]
                if len(hidden_states.shape) == 3:
                    actual_seq_len = hidden_states.shape[1]
                elif len(hidden_states.shape) == 2:
                    actual_seq_len = hidden_states.shape[0]
                else:
                    continue
                
                # 如果 prompt_len >= actual_seq_len，说明 hidden_states 被截断了
                # 需要调整 prompt_len 使其有意义
                original_prompt_len = feat.prompt_len
                original_response_len = feat.response_len
                total_original = original_prompt_len + original_response_len
                
                if original_prompt_len >= actual_seq_len:
                    # 根据比例计算新的 prompt_len
                    if total_original > 0:
                        ratio = original_prompt_len / total_original
                        new_prompt_len = int(actual_seq_len * ratio)
                        # 至少保留一个 response token
                        new_prompt_len = min(new_prompt_len, actual_seq_len - 1)
                        new_prompt_len = max(0, new_prompt_len)
                    else:
                        new_prompt_len = 0
                    
                    # 更新 prompt_len
                    feat.prompt_len = new_prompt_len
                    feat.response_len = actual_seq_len - new_prompt_len
                    
                    logger.debug(
                        f"Fixed prompt_len for {feat.sample_id}: "
                        f"{original_prompt_len} -> {new_prompt_len} "
                        f"(seq_len={actual_seq_len})"
                    )
            
            # 释放大特征以节省内存
            feat.release_large_features()


def main():
    parser = argparse.ArgumentParser(description="快速评估已训练的模型（修复版）")
    parser.add_argument("--methods", nargs="+", required=True, help="要评估的方法列表")
    parser.add_argument("--dataset", default="ragtruth", help="数据集名称")
    parser.add_argument("--model", default="Mistral-7B-Instruct-v0.3", help="模型名称")
    parser.add_argument("--task_type", default="QA", help="任务类型")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--features_dir", default="outputs/features", help="特征目录")
    parser.add_argument("--models_dir", default="outputs/models", help="模型目录")
    parser.add_argument("--splits_dir", default="outputs/splits", help="数据划分目录")
    parser.add_argument("--output", default=None, help="结果输出文件")
    parser.add_argument("--level", default="sample", help="分类级别")
    
    args = parser.parse_args()
    
    features_dir = Path(args.features_dir)
    models_dir = Path(args.models_dir)
    splits_dir = Path(args.splits_dir)
    
    logger.info("=" * 60)
    logger.info("Quick Evaluation (Fixed Version)")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Task: {args.task_type}")
    logger.info(f"Methods: {args.methods}")
    logger.info("=" * 60)
    
    # === 修复1：先加载测试集 ID ===
    test_ids = load_split_ids(splits_dir, args.dataset, "test")
    if test_ids:
        logger.info(f"Loaded {len(test_ids)} test sample IDs from splits directory")
    else:
        logger.warning("No split file found, will try to use split field from answers.json")
    
    # 查找特征目录
    feat_dir = find_features_dir(features_dir, args.dataset, args.model, args.seed, args.task_type, "train")
    if feat_dir is None:
        logger.error(f"Features directory not found")
        logger.error(f"Searched in: {features_dir}/{args.dataset}/{args.model}/seed_{args.seed}/{args.task_type}")
        return
    
    logger.info(f"Loading features from: {feat_dir}")
    
    # 加载特征
    try:
        features_list, samples = load_features_and_labels(feat_dir, test_ids)
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # === 修复2：使用 test_ids 进行分割 ===
    train_features, train_labels, test_features, test_labels = split_by_field(
        features_list, samples, test_ids
    )
    
    if not test_features:
        logger.warning("No test split found, using all data for evaluation")
        logger.warning("  - Check if outputs/splits/{dataset}/test.json exists")
        logger.warning("  - Or run: python scripts/split_dataset.py")
        test_features = features_list
        test_labels = [f.label or 0 for f in features_list]
    else:
        logger.info(f"Train samples: {len(train_features)} ({sum(train_labels)} positive)")
    
    logger.info(f"Test samples: {len(test_features)} ({sum(test_labels)} positive)")
    
    # 评估每个方法
    all_results = []
    
    for method_name in args.methods:
        logger.info(f"\n>>> Evaluating: {method_name}")
        
        model_path = find_model_path(
            models_dir, args.dataset, args.model, args.seed, 
            args.task_type, method_name, args.level
        )
        
        if model_path is None:
            logger.error(f"  Model not found for {method_name}")
            logger.error(f"  Searched: {models_dir}/{args.dataset}/{args.model}/seed_{args.seed}/{args.task_type}/{method_name}/")
            all_results.append({
                "method": method_name,
                "error": "Model file not found"
            })
            continue
        
        logger.info(f"  Loading from: {model_path}")
        
        result = evaluate_method(method_name, model_path, test_features, test_labels)
        all_results.append(result)
        
        if "error" in result:
            logger.error(f"  Error: {result['error']}")
        else:
            logger.info(f"  AUROC: {result.get('auroc', 0):.4f}")
            logger.info(f"  AUPR:  {result.get('aupr', 0):.4f}")
            logger.info(f"  F1:    {result.get('f1', 0):.4f}")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'AUROC':<10} {'AUPR':<10} {'F1':<10}")
    print("-" * 60)
    
    for result in all_results:
        if "error" in result:
            print(f"{result['method']:<20} ERROR: {result['error'][:30]}")
        else:
            print(f"{result['method']:<20} {result.get('auroc', 0):.4f}     {result.get('aupr', 0):.4f}     {result.get('f1', 0):.4f}")
    
    print("=" * 60)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "config": {
                    "dataset": args.dataset,
                    "model": args.model,
                    "task_type": args.task_type,
                    "seed": args.seed,
                },
                "results": all_results,
            }, f, indent=2)
        logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
