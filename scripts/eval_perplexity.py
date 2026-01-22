#!/usr/bin/env python
"""Perplexity Training-Free 评估脚本 (独立版本)。

不依赖 Hydra，直接指定特征目录运行。

=============================================================================
使用方式
=============================================================================

# 基本用法
python eval_perplexity_simple.py --features_dir outputs/features/ragtruth/llama3-8b/seed_42/qa/test

# 指定输出目录
python scripts/eval_perplexity.py   --features_dir outputs/features/ragtruth/Mistral-7B-Instruct-v0.3/seed_42/QA_test --output_dir outputs/results/perplexity

# 同时评估多个目录
python eval_perplexity_simple.py --features_dir outputs/features/ragtruth/llama3-8b/seed_42/qa/train --features_dir outputs/features/ragtruth/llama3-8b/seed_42/QA_test

=============================================================================
原理 (Ren et al., ICLR 2023)
=============================================================================
- Training-Free: 直接用 perplexity 值作为分数
- Higher perplexity = 更可能是幻觉
- PPL = exp(-1/T × Σ log P(token))
"""
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def safe_to_numpy(tensor) -> Optional[np.ndarray]:
    """安全地将张量转换为 NumPy 数组。"""
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.float32)
    
    import torch
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().float().numpy()
    
    return np.asarray(tensor, dtype=np.float32)


def compute_perplexity(token_probs: np.ndarray, eps: float = 1e-10) -> float:
    """从 token 概率计算 perplexity。"""
    if token_probs is None or len(token_probs) == 0:
        return np.nan
    probs = np.clip(token_probs.flatten(), eps, 1.0)
    return float(np.exp(-np.mean(np.log(probs))))


def get_perplexity_score(features) -> float:
    """从 ExtractedFeatures 获取 perplexity。"""
    # 优先使用预计算的 perplexity
    if hasattr(features, 'perplexity') and features.perplexity is not None:
        return float(features.perplexity)
    
    # 从 token_probs 计算
    if hasattr(features, 'token_probs') and features.token_probs is not None:
        token_probs = safe_to_numpy(features.token_probs)
        
        # 只使用 response 部分
        prompt_len = getattr(features, 'prompt_len', 0)
        response_len = getattr(features, 'response_len', 0)
        
        if response_len > 0 and prompt_len < len(token_probs):
            start = prompt_len
            end = min(start + response_len, len(token_probs))
            if end > start:
                token_probs = token_probs[start:end]
        
        return compute_perplexity(token_probs)
    
    return np.nan


def load_features(features_dir: Path) -> list:
    """加载特征文件。"""
    from src.core import ExtractedFeatures
    
    features_dir = Path(features_dir)
    if not features_dir.exists():
        raise FileNotFoundError(f"Directory not found: {features_dir}")
    
    feature_files = sorted(features_dir.glob("*.pt"))
    if not feature_files:
        raise FileNotFoundError(f"No .pt files in {features_dir}")
    
    logger.info(f"Loading {len(feature_files)} files from {features_dir}")
    
    features_list = []
    for f in feature_files:
        try:
            feat = ExtractedFeatures.load(f)
            features_list.append(feat)
        except Exception as e:
            logger.warning(f"Failed to load {f.name}: {e}")
    
    return features_list


def evaluate(features_list: list, name: str = "eval") -> Dict[str, Any]:
    """评估 perplexity 方法。"""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, precision_score, recall_score, roc_curve
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Perplexity Training-Free Evaluation: {name}")
    logger.info(f"{'='*60}")
    
    scores, labels, sample_ids = [], [], []
    
    for feat in features_list:
        ppl = get_perplexity_score(feat)
        label = getattr(feat, 'label', None)
        
        if label is not None and np.isfinite(ppl):
            scores.append(ppl)
            labels.append(label)
            sample_ids.append(getattr(feat, 'sample_id', 'unknown'))
    
    if not scores:
        logger.error("No valid samples")
        return {"error": "no_valid_samples"}
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    n_total = len(scores)
    n_pos = int(labels.sum())
    n_neg = n_total - n_pos
    
    logger.info(f"Samples: {n_total} (positive={n_pos}, negative={n_neg})")
    logger.info(f"PPL range: [{scores.min():.2f}, {scores.max():.2f}], mean={scores.mean():.2f}")
    
    if n_pos > 0 and n_neg > 0:
        ppl_pos = scores[labels == 1]
        ppl_neg = scores[labels == 0]
        logger.info(f"PPL (hallucinated): {ppl_pos.mean():.2f} ± {ppl_pos.std():.2f}")
        logger.info(f"PPL (faithful): {ppl_neg.mean():.2f} ± {ppl_neg.std():.2f}")
    
    results = {
        "name": name,
        "n_samples": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "ppl_min": float(scores.min()),
        "ppl_max": float(scores.max()),
        "ppl_mean": float(scores.mean()),
        "ppl_std": float(scores.std()),
    }
    
    if n_pos > 0 and n_neg > 0:
        # Higher PPL = more likely hallucination
        auroc = roc_auc_score(labels, scores)
        aupr = average_precision_score(labels, scores)
        
        # 最优阈值
        fpr, tpr, thresholds = roc_curve(labels, scores)
        best_idx = np.argmax(tpr - fpr)
        best_thresh = thresholds[best_idx]
        
        y_pred = (scores >= best_thresh).astype(int)
        f1 = f1_score(labels, y_pred)
        prec = precision_score(labels, y_pred, zero_division=0)
        rec = recall_score(labels, y_pred, zero_division=0)
        
        results.update({
            "auroc": float(auroc),
            "aupr": float(aupr),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "optimal_threshold": float(best_thresh),
        })
        
        logger.info(f"\n>>> RESULTS:")
        logger.info(f"    AUROC: {auroc:.4f}")
        logger.info(f"    AUPR:  {aupr:.4f}")
        logger.info(f"    F1:    {f1:.4f} (threshold={best_thresh:.2f})")
        logger.info(f"    Precision: {prec:.4f}")
        logger.info(f"    Recall: {rec:.4f}")
    else:
        results["auroc"] = 0.5
        results["aupr"] = n_pos / n_total if n_total > 0 else 0.0
        logger.warning("Only one class, AUROC=0.5")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Perplexity Training-Free Evaluation (Ren et al., ICLR 2023)"
    )
    parser.add_argument(
        "--features_dir", "-f",
        type=str,
        required=True,
        action="append",
        help="Features directory (can specify multiple times)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="perplexity_results.json",
        help="Output filename"
    )
    
    args = parser.parse_args()
    
    logger.info("\n" + "=" * 60)
    logger.info("Perplexity Training-Free Evaluation")
    logger.info("Paper: Ren et al., ICLR 2023")
    logger.info("=" * 60)
    
    all_results = {}
    
    for features_dir in args.features_dir:
        features_dir = Path(features_dir)
        name = features_dir.name  # e.g., "train" or "test"
        
        try:
            features_list = load_features(features_dir)
            results = evaluate(features_list, name)
            all_results[str(features_dir)] = results
        except Exception as e:
            logger.error(f"Failed to evaluate {features_dir}: {e}")
            all_results[str(features_dir)] = {"error": str(e)}
    
    # 保存结果
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / args.output_file
    else:
        output_file = Path(args.output_file)
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for path, results in all_results.items():
        if "error" not in results:
            auroc = results.get("auroc", 0.5)
            aupr = results.get("aupr", 0.0)
            f1 = results.get("f1", 0.0)
            name = results.get("name", Path(path).name)
            logger.info(f"[{name}] AUROC={auroc:.4f}, AUPR={aupr:.4f}, F1={f1:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()