#!/usr/bin/env python3
"""
Perplexity-based Hallucination Detection (Training-Free Baseline)

基于论文: "Out-of-Distribution Detection and Selective Generation for 
Conditional Language Models" (ICLR 2023)
https://arxiv.org/abs/2209.15558

原论文核心结论:
1. Perplexity 是 OOD 检测的弱 baseline（AUROC 通常在 0.5-0.6）
2. Perplexity 在分布内数据上与质量相关，但在 OOD 数据上相关性下降
3. 结合 embedding-based OOD score 效果更好

本脚本实现:
- 从 token_probs 计算 perplexity: PPL = exp(-1/N * sum(log(p_i)))
- 直接用 perplexity（或其负值）作为 hallucination score
- 计算 AUROC, AUPRC, F1 等指标

Usage:
    python scripts/eval_perplexity.py \
        --features_dir outputs/features/ragtruth/Mistral-7B-Instruct-v0.3/seed_42/QA_test \
        --output_dir outputs/results/perplexity
    
    # 批量评估所有 *_test 目录
    python scripts/eval_perplexity.py \
        --base_dir outputs/features/ragtruth/Mistral-7B-Instruct-v0.3/seed_42 \
        --output_dir outputs/results/perplexity \
        --all_test
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    f1_score, 
    precision_score, 
    recall_score,
    accuracy_score,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# 核心计算函数 - 完全按照原论文实现
# =============================================================================

def compute_perplexity_from_token_probs(
    token_probs: torch.Tensor,
    eps: float = 1e-10,
) -> float:
    """
    计算困惑度（原论文公式）。
    
    PPL = exp(-1/N * sum(log(p_i)))
    
    其中 p_i 是每个 token 的预测概率。
    
    Args:
        token_probs: [seq_len] 每个 token 的预测概率
        eps: 数值稳定性
        
    Returns:
        困惑度值
    """
    if token_probs is None or len(token_probs) == 0:
        return float('nan')
    
    if isinstance(token_probs, torch.Tensor):
        token_probs = token_probs.float().cpu().numpy()
    
    # 确保概率在有效范围内
    token_probs = np.clip(token_probs, eps, 1.0)
    
    # 计算平均负对数概率
    log_probs = np.log(token_probs)
    avg_neg_log_prob = -np.mean(log_probs)
    
    # 困惑度
    perplexity = np.exp(avg_neg_log_prob)
    
    return float(perplexity)


def compute_average_neg_log_prob(
    token_probs: torch.Tensor,
    eps: float = 1e-10,
) -> float:
    """
    计算平均负对数概率（与 perplexity 单调相关）。
    
    NLL = -1/N * sum(log(p_i))
    
    这是原论文中用于比较的另一个指标。
    """
    if token_probs is None or len(token_probs) == 0:
        return float('nan')
    
    if isinstance(token_probs, torch.Tensor):
        token_probs = token_probs.float().cpu().numpy()
    
    token_probs = np.clip(token_probs, eps, 1.0)
    log_probs = np.log(token_probs)
    
    return float(-np.mean(log_probs))


# =============================================================================
# 数据加载
# =============================================================================

@dataclass
class SampleData:
    """样本数据"""
    sample_id: str
    label: int
    perplexity: float
    avg_neg_log_prob: float
    token_probs_mean: float
    token_probs_std: float
    token_probs_min: float
    response_len: int


def load_features_from_dir(features_dir: Path) -> Tuple[List[SampleData], Dict[str, Any]]:
    """
    从特征目录加载数据。
    
    支持两种格式：
    1. 合并格式: features/token_probs.pt + metadata.json
    2. 单独文件格式: features_individual/*.pt
    
    Args:
        features_dir: 特征目录
        
    Returns:
        (sample_data_list, metadata)
    """
    features_dir = Path(features_dir)
    
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    # 加载元数据
    metadata_path = features_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    
    # 加载答案/标签
    answers_path = features_dir / "answers.json"
    labels_path = features_dir / "labels.pt"
    
    sample_labels = {}
    if answers_path.exists():
        with open(answers_path) as f:
            answers = json.load(f)
        for ans in answers:
            sample_labels[str(ans["id"])] = ans.get("label", 0)
    
    if labels_path.exists():
        labels_tensor = torch.load(labels_path, weights_only=False)
        sample_ids = metadata.get("sample_ids", [])
        for i, sid in enumerate(sample_ids):
            if i < len(labels_tensor):
                sample_labels[str(sid)] = int(labels_tensor[i])
    
    # 尝试从合并的特征文件加载
    consolidated_dir = features_dir / "features"
    individual_dir = features_dir / "features_individual"
    
    samples_data = []
    
    # 方法1: 从合并文件加载
    token_probs_path = consolidated_dir / "token_probs.pt"
    if token_probs_path.exists():
        logger.info(f"Loading from consolidated features: {token_probs_path}")
        token_probs_dict = torch.load(token_probs_path, weights_only=False)
        
        for sample_id, token_probs in token_probs_dict.items():
            sample_id_str = str(sample_id)
            label = sample_labels.get(sample_id_str, 0)
            
            if token_probs is None or (isinstance(token_probs, torch.Tensor) and token_probs.numel() == 0):
                continue
            
            if isinstance(token_probs, torch.Tensor):
                probs_np = token_probs.float().cpu().numpy()
            else:
                probs_np = np.array(token_probs)
            
            if len(probs_np) == 0:
                continue
            
            samples_data.append(SampleData(
                sample_id=sample_id_str,
                label=label,
                perplexity=compute_perplexity_from_token_probs(probs_np),
                avg_neg_log_prob=compute_average_neg_log_prob(probs_np),
                token_probs_mean=float(np.mean(probs_np)),
                token_probs_std=float(np.std(probs_np)),
                token_probs_min=float(np.min(probs_np)),
                response_len=len(probs_np),
            ))
    
    # 方法2: 从单独文件加载
    elif individual_dir.exists():
        logger.info(f"Loading from individual features: {individual_dir}")
        feature_files = sorted(individual_dir.glob("*.pt"))
        
        for feat_path in feature_files:
            try:
                data = torch.load(feat_path, weights_only=False)
                
                # 提取信息
                info = data.get("info", {})
                tensors = data.get("tensors", {})
                
                sample_id = str(info.get("sample_id", feat_path.stem))
                label = info.get("label", sample_labels.get(sample_id, 0))
                
                # 获取 token_probs
                token_probs = tensors.get("token_probs")
                
                # 如果 tensors 中没有，尝试从 info 中获取 perplexity
                if token_probs is None:
                    # 检查是否有预计算的 perplexity
                    precomputed_ppl = info.get("perplexity")
                    if precomputed_ppl is not None:
                        samples_data.append(SampleData(
                            sample_id=sample_id,
                            label=label,
                            perplexity=float(precomputed_ppl),
                            avg_neg_log_prob=np.log(precomputed_ppl) if precomputed_ppl > 0 else float('nan'),
                            token_probs_mean=0.0,
                            token_probs_std=0.0,
                            token_probs_min=0.0,
                            response_len=info.get("response_len", 0),
                        ))
                    continue
                
                if isinstance(token_probs, torch.Tensor):
                    probs_np = token_probs.float().cpu().numpy()
                else:
                    probs_np = np.array(token_probs)
                
                if len(probs_np) == 0:
                    continue
                
                samples_data.append(SampleData(
                    sample_id=sample_id,
                    label=label,
                    perplexity=compute_perplexity_from_token_probs(probs_np),
                    avg_neg_log_prob=compute_average_neg_log_prob(probs_np),
                    token_probs_mean=float(np.mean(probs_np)),
                    token_probs_std=float(np.std(probs_np)),
                    token_probs_min=float(np.min(probs_np)),
                    response_len=len(probs_np),
                ))
                
            except Exception as e:
                logger.warning(f"Failed to load {feat_path}: {e}")
                continue
    
    else:
        raise FileNotFoundError(
            f"No features found. Expected either:\n"
            f"  - {consolidated_dir}/token_probs.pt\n"
            f"  - {individual_dir}/*.pt"
        )
    
    return samples_data, metadata


# =============================================================================
# 评估函数
# =============================================================================

@dataclass
class EvaluationResults:
    """评估结果"""
    # 基本信息
    task_name: str
    n_samples: int
    n_positive: int  # hallucinated
    n_negative: int  # correct
    
    # 核心指标 (使用 perplexity 作为 score, 越高越可能是幻觉)
    auroc: float
    auprc: float
    
    # 分类指标 (使用最优阈值)
    best_threshold: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    
    # 统计信息
    ppl_mean_hallucinated: float
    ppl_mean_correct: float
    ppl_std_hallucinated: float
    ppl_std_correct: float


def evaluate_perplexity(
    samples_data: List[SampleData],
    task_name: str = "unknown",
) -> EvaluationResults:
    """
    评估困惑度方法的幻觉检测性能。
    
    原论文使用方式：
    - 将 perplexity（或负对数似然）作为 OOD score
    - 高 perplexity 可能表示 OOD / 幻觉
    
    但对于幻觉检测，论文发现：
    - perplexity 作为单独指标效果不好（AUROC ~0.5-0.6）
    - 需要结合 embedding-based OOD score
    
    Args:
        samples_data: 样本数据列表
        task_name: 任务名称
        
    Returns:
        评估结果
    """
    if len(samples_data) == 0:
        raise ValueError("No samples to evaluate")
    
    # 过滤无效样本
    valid_samples = [s for s in samples_data if not np.isnan(s.perplexity)]
    
    if len(valid_samples) == 0:
        raise ValueError("All samples have invalid perplexity values")
    
    logger.info(f"Valid samples: {len(valid_samples)}/{len(samples_data)}")
    
    # 提取数据
    labels = np.array([s.label for s in valid_samples])
    perplexities = np.array([s.perplexity for s in valid_samples])
    
    n_positive = int(np.sum(labels == 1))
    n_negative = int(np.sum(labels == 0))
    
    logger.info(f"Label distribution: {n_positive} hallucinated, {n_negative} correct")
    
    # 检查是否有两个类别
    if n_positive == 0 or n_negative == 0:
        logger.warning("Only one class present, metrics may be undefined")
        return EvaluationResults(
            task_name=task_name,
            n_samples=len(valid_samples),
            n_positive=n_positive,
            n_negative=n_negative,
            auroc=0.5,
            auprc=n_positive / len(valid_samples) if len(valid_samples) > 0 else 0.0,
            best_threshold=np.median(perplexities),
            f1=0.0,
            precision=0.0,
            recall=0.0,
            accuracy=max(n_positive, n_negative) / len(valid_samples),
            ppl_mean_hallucinated=float(np.mean(perplexities[labels == 1])) if n_positive > 0 else 0.0,
            ppl_mean_correct=float(np.mean(perplexities[labels == 0])) if n_negative > 0 else 0.0,
            ppl_std_hallucinated=float(np.std(perplexities[labels == 1])) if n_positive > 0 else 0.0,
            ppl_std_correct=float(np.std(perplexities[labels == 0])) if n_negative > 0 else 0.0,
        )
    
    # 使用 perplexity 作为 score
    # 假设: 高 perplexity = 更可能是幻觉 (label=1)
    scores = perplexities
    
    # 计算 AUROC 和 AUPRC
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    
    # 如果 AUROC < 0.5，说明关系是反向的（低 perplexity = 幻觉）
    # 这种情况在某些数据集上可能发生
    if auroc < 0.5:
        logger.warning(f"AUROC < 0.5 ({auroc:.4f}), trying reversed scores")
        scores_reversed = -perplexities
        auroc_reversed = roc_auc_score(labels, scores_reversed)
        if auroc_reversed > auroc:
            logger.info(f"Using reversed scores: AUROC {auroc:.4f} -> {auroc_reversed:.4f}")
            scores = scores_reversed
            auroc = auroc_reversed
            auprc = average_precision_score(labels, scores)
    
    # 找最优阈值 (基于 F1)
    thresholds = np.percentile(scores, np.arange(1, 100))
    best_f1 = 0.0
    best_threshold = np.median(scores)
    
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    # 使用最优阈值计算指标
    final_preds = (scores >= best_threshold).astype(int)
    
    # 统计信息
    ppl_hallucinated = perplexities[labels == 1]
    ppl_correct = perplexities[labels == 0]
    
    return EvaluationResults(
        task_name=task_name,
        n_samples=len(valid_samples),
        n_positive=n_positive,
        n_negative=n_negative,
        auroc=float(auroc),
        auprc=float(auprc),
        best_threshold=float(best_threshold),
        f1=float(f1_score(labels, final_preds, zero_division=0)),
        precision=float(precision_score(labels, final_preds, zero_division=0)),
        recall=float(recall_score(labels, final_preds, zero_division=0)),
        accuracy=float(accuracy_score(labels, final_preds)),
        ppl_mean_hallucinated=float(np.mean(ppl_hallucinated)),
        ppl_mean_correct=float(np.mean(ppl_correct)),
        ppl_std_hallucinated=float(np.std(ppl_hallucinated)),
        ppl_std_correct=float(np.std(ppl_correct)),
    )


# =============================================================================
# 主函数
# =============================================================================

def find_test_dirs(base_dir: Path) -> List[Path]:
    """
    查找所有 *_test 目录。
    
    Args:
        base_dir: 基础目录
        
    Returns:
        所有 *_test 目录的路径列表
    """
    test_dirs = []
    
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.endswith("_test"):
            test_dirs.append(item)
    
    return sorted(test_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Perplexity-based Hallucination Detection (Training-Free)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 单个目录评估
    python scripts/eval_perplexity.py \\
        --features_dir outputs/features/ragtruth/model/seed_42/QA_test

    # 批量评估所有 *_test 目录
    python scripts/eval_perplexity.py --base_dir outputs/features/ragtruth/Mistral-7B-Instruct-v0.3/seed_42 --all_test

    # 指定输出目录
    python scripts/eval_perplexity.py \\
        --base_dir outputs/features/ragtruth/Mistral-7B-Instruct-v0.3/seed_42 \\
        --all_test
        --output_dir outputs/results/perplexity
        """
    )
    
    # 输入选项（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--features_dir",
        type=Path,
        help="单个特征目录路径"
    )
    input_group.add_argument(
        "--base_dir",
        type=Path,
        help="基础目录（与 --all_test 一起使用）"
    )
    
    parser.add_argument(
        "--all_test",
        action="store_true",
        help="评估 base_dir 下所有 *_test 目录"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="outputs/results/perplexity",
        help="输出目录（默认打印到控制台）"
    )
    
    args = parser.parse_args()
    
    # 确定要评估的目录
    if args.features_dir:
        features_dirs = [args.features_dir]
    elif args.base_dir and args.all_test:
        features_dirs = find_test_dirs(args.base_dir)
        if not features_dirs:
            logger.error(f"No *_test directories found in {args.base_dir}")
            sys.exit(1)
        logger.info(f"Found {len(features_dirs)} test directories:")
        for d in features_dirs:
            logger.info(f"  - {d.name}")
    else:
        parser.error("--base_dir requires --all_test flag")
    
    # 评估每个目录
    all_results = []
    
    for features_dir in features_dirs:
        task_name = features_dir.name.replace("_test", "")
        
        logger.info("")
        logger.info(f"Perplexity Training-Free Evaluation: {features_dir.name}")
        logger.info("=" * 60)
        
        try:
            # 加载数据
            samples_data, metadata = load_features_from_dir(features_dir)
            
            if len(samples_data) == 0:
                logger.error("No valid samples")
                continue
            
            logger.info(f"Loaded {len(samples_data)} samples")
            
            # 评估
            results = evaluate_perplexity(samples_data, task_name)
            all_results.append(results)
            
            # 打印结果
            logger.info("")
            logger.info(f"Results for {task_name}:")
            logger.info(f"  Samples: {results.n_samples} ({results.n_positive} hallucinated, {results.n_negative} correct)")
            logger.info(f"  AUROC: {results.auroc:.4f}")
            logger.info(f"  AUPRC: {results.auprc:.4f}")
            logger.info(f"  F1: {results.f1:.4f} (threshold={results.best_threshold:.2f})")
            logger.info(f"  Precision: {results.precision:.4f}")
            logger.info(f"  Recall: {results.recall:.4f}")
            logger.info(f"  Accuracy: {results.accuracy:.4f}")
            logger.info(f"  PPL (hallucinated): {results.ppl_mean_hallucinated:.2f} +/- {results.ppl_std_hallucinated:.2f}")
            logger.info(f"  PPL (correct): {results.ppl_mean_correct:.2f} +/- {results.ppl_std_correct:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {features_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果
    if args.output_dir and all_results:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        results_path = args.output_dir / "perplexity_results.json"
        with open(results_path, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        logger.info(f"\nResults saved to {results_path}")
        
        # 保存汇总表
        summary_path = args.output_dir / "perplexity_summary.txt"
        with open(summary_path, "w") as f:
            f.write("Perplexity-based Hallucination Detection Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"{'Task':<20} {'AUROC':<10} {'AUPRC':<10} {'F1':<10} {'P':<10} {'R':<10}\n")
            f.write("-" * 80 + "\n")
            for r in all_results:
                f.write(f"{r.task_name:<20} {r.auroc:<10.4f} {r.auprc:<10.4f} {r.f1:<10.4f} {r.precision:<10.4f} {r.recall:<10.4f}\n")
            
            # 平均值
            if len(all_results) > 1:
                f.write("-" * 80 + "\n")
                avg_auroc = np.mean([r.auroc for r in all_results])
                avg_auprc = np.mean([r.auprc for r in all_results])
                avg_f1 = np.mean([r.f1 for r in all_results])
                avg_p = np.mean([r.precision for r in all_results])
                avg_r = np.mean([r.recall for r in all_results])
                f.write(f"{'Average':<20} {avg_auroc:<10.4f} {avg_auprc:<10.4f} {avg_f1:<10.4f} {avg_p:<10.4f} {avg_r:<10.4f}\n")
        
        logger.info(f"Summary saved to {summary_path}")
    
    # 打印最终汇总
    if len(all_results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"{'Task':<20} {'AUROC':<10} {'AUPRC':<10} {'F1':<10}")
        logger.info("-" * 50)
        for r in all_results:
            logger.info(f"{r.task_name:<20} {r.auroc:<10.4f} {r.auprc:<10.4f} {r.f1:<10.4f}")
        logger.info("-" * 50)
        avg_auroc = np.mean([r.auroc for r in all_results])
        avg_auprc = np.mean([r.auprc for r in all_results])
        avg_f1 = np.mean([r.f1 for r in all_results])
        logger.info(f"{'Average':<20} {avg_auroc:<10.4f} {avg_auprc:<10.4f} {avg_f1:<10.4f}")


if __name__ == "__main__":
    main()