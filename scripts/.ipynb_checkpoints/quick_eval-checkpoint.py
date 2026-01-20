#!/usr/bin/env python3
"""快速评估已训练模型的脚本（修复版 v2）。

核心修复:
1. 正确从 {task}_test 目录加载测试数据（而不是从 train 目录分割）
2. 支持跨任务评估（--cross_task）

用法:
    # 同任务评估（在 QA_test 上评估 QA 训练的模型）
    python scripts/quick_eval.py --methods lapeigvals hsdmvaf \
        --dataset ragtruth --model Mistral-7B-Instruct-v0.3 \
        --task_type QA --seed 42

    # 跨任务评估（在所有任务的 test 集上评估 QA 训练的模型）
    python scripts/quick_eval.py --methods lapeigvals hsdmvaf \
        --dataset ragtruth --model Mistral-7B-Instruct-v0.3 \
        --train_task QA --cross_task --seed 42

    # 指定评估任务（在 Summary_test 上评估 QA 训练的模型）
    python scripts/quick_eval.py --methods lapeigvals hsdmvaf \
        --dataset ragtruth --model Mistral-7B-Instruct-v0.3 \
        --train_task QA --eval_task Summary --seed 42
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


def find_test_features_dir(
    features_dir: Path,
    dataset: str,
    model: str,
    seed: int,
    task: str,
) -> Optional[Path]:
    """查找测试集特征目录。
    
    优先查找 {task}_test 目录。
    """
    # 测试集目录应该是 {task}_test
    possible_paths = [
        features_dir / dataset / model / f"seed_{seed}" / f"{task}_test",
        features_dir / dataset / model / f"seed_{seed}" / task / "test",  # 备选
    ]
    
    for path in possible_paths:
        if path.exists() and (path / "answers.json").exists():
            return path
    
    return None


def find_train_features_dir(
    features_dir: Path,
    dataset: str,
    model: str,
    seed: int,
    task: str,
) -> Optional[Path]:
    """查找训练集特征目录。"""
    possible_paths = [
        features_dir / dataset / model / f"seed_{seed}" / task,
        features_dir / dataset / model / f"seed_{seed}" / f"{task}_train",
    ]
    
    for path in possible_paths:
        if path.exists() and (path / "answers.json").exists():
            return path
    
    return None


def load_features_from_dir(features_dir: Path) -> tuple:
    """从指定目录加载特征和标签。
    
    Args:
        features_dir: 特征目录（如 QA_test/）
        
    Returns:
        (features_list, labels)
    """
    import torch
    from src.core import ExtractedFeatures
    
    metadata_path = features_dir / "metadata.json"
    answers_path = features_dir / "answers.json"
    
    if not answers_path.exists():
        raise FileNotFoundError(f"answers.json not found in {features_dir}")
    
    with open(answers_path) as f:
        answers = json.load(f)
    
    # 构建 sample_id -> answer 的映射
    answers_by_id = {str(ans["id"]): ans for ans in answers}
    
    # 获取 sample_ids 顺序
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        sample_ids = metadata.get("sample_ids", list(answers_by_id.keys()))
    else:
        sample_ids = list(answers_by_id.keys())
    
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
        "hidden_states": "hidden_states.pt",
    }
    
    loaded_features = {}
    for key, filename in feature_files.items():
        filepath = features_subdir / filename
        if filepath.exists():
            loaded_features[key] = torch.load(filepath, weights_only=False)
    
    # 衍生特征：laplacian_diags
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
        labels_tensor = torch.load(labels_path, weights_only=False)
    else:
        labels_tensor = None
    
    # 构建特征列表
    features_list = []
    labels = []
    
    for i, sample_id in enumerate(sample_ids):
        sample_id = str(sample_id)
        
        if sample_id not in answers_by_id:
            continue
        
        ans = answers_by_id[sample_id]
        
        sample_features = {}
        for key, data in loaded_features.items():
            if isinstance(data, dict):
                if sample_id in data:
                    sample_features[key] = data[sample_id]
                elif str(sample_id) in data:
                    sample_features[key] = data[str(sample_id)]
        
        # 大特征路径
        feature_paths = {}
        for feature_key in ["hidden_states", "full_attentions"]:
            if feature_key in large_feature_indexes:
                index_data = large_feature_indexes[feature_key]
                if "index" in index_data and sample_id in index_data["index"]:
                    feature_paths[feature_key] = index_data["index"][sample_id]
        
        # 获取标签
        if labels_tensor is not None and i < len(labels_tensor):
            label = int(labels_tensor[i])
        else:
            label = ans.get("label", 0)
        
        features_list.append(ExtractedFeatures(
            sample_id=sample_id,
            prompt_len=ans.get("prompt_len", 0),
            response_len=ans.get("response_len", 0),
            label=label,
            attn_diags=sample_features.get("attn_diags"),
            laplacian_diags=sample_features.get("laplacian_diags"),
            attn_entropy=sample_features.get("attn_entropy"),
            token_probs=sample_features.get("token_probs"),
            token_entropy=sample_features.get("token_entropy"),
            hidden_states=sample_features.get("hidden_states"),
            metadata={"_feature_paths": feature_paths},
        ))
        labels.append(label)
    
    return features_list, labels


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
        
        # 对于 HaloScope，修复 response_start 问题
        if method_name.lower() in ['haloscope', 'halo', 'haloscope_svd']:
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
    """修复 HaloScope 特征的 prompt_len 问题。"""
    import torch
    
    for feat in features_list:
        hidden_states = feat.hidden_states
        if hidden_states is None:
            hidden_states = feat.get_hidden_states()
        
        if hidden_states is not None:
            if isinstance(hidden_states, torch.Tensor):
                if len(hidden_states.shape) == 3:
                    actual_seq_len = hidden_states.shape[1]
                elif len(hidden_states.shape) == 2:
                    actual_seq_len = hidden_states.shape[0]
                else:
                    continue
                
                original_prompt_len = feat.prompt_len
                original_response_len = feat.response_len
                total_original = original_prompt_len + original_response_len
                
                if original_prompt_len >= actual_seq_len:
                    if total_original > 0:
                        ratio = original_prompt_len / total_original
                        new_prompt_len = int(actual_seq_len * ratio)
                        new_prompt_len = min(new_prompt_len, actual_seq_len - 1)
                        new_prompt_len = max(0, new_prompt_len)
                    else:
                        new_prompt_len = 0
                    
                    feat.prompt_len = new_prompt_len
                    feat.response_len = actual_seq_len - new_prompt_len
                    
                    logger.debug(
                        f"Fixed prompt_len for {feat.sample_id}: "
                        f"{original_prompt_len} -> {new_prompt_len} "
                        f"(seq_len={actual_seq_len})"
                    )
            
            feat.release_large_features()


def main():
    parser = argparse.ArgumentParser(description="快速评估已训练的模型（修复版 v2）")
    parser.add_argument("--methods", nargs="+", required=True, help="要评估的方法列表")
    parser.add_argument("--dataset", default="ragtruth", help="数据集名称")
    parser.add_argument("--model", default="Mistral-7B-Instruct-v0.3", help="模型名称")
    
    # 任务相关参数
    parser.add_argument("--task_type", default=None, help="任务类型（同时用于模型和评估）")
    parser.add_argument("--train_task", default=None, help="模型训练时的任务类型")
    parser.add_argument("--eval_task", default=None, help="评估时的任务类型（默认同 train_task）")
    parser.add_argument("--cross_task", action="store_true", 
                       help="跨任务评估：在所有任务的 test 集上评估")
    parser.add_argument("--all_tasks", nargs="+", default=["QA", "Summary", "Data2txt"],
                       help="跨任务评估时的所有任务列表")
    
    parser.add_argument("--seed", type=int, default=42, help="默认随机种子")
    parser.add_argument("--method_seeds", nargs="+", default=[], 
                       help="为特定方法指定不同的 seed，格式: method:seed (如 haloscope:41)")
    parser.add_argument("--features_dir", default="outputs/features", help="特征目录")
    parser.add_argument("--models_dir", default="outputs/models", help="模型目录")
    parser.add_argument("--results_dir", default="outputs/results", help="结果目录（用于保存 eval_results.json）")
    parser.add_argument("--output", default=None, help="结果输出文件")
    parser.add_argument("--level", default="sample", help="分类级别")
    parser.add_argument("--save_eval_results", action="store_true",
                       help="保存 eval_results.json 到 results_dir（可被 aggregate_results.py 识别）")
    
    args = parser.parse_args()
    
    # 解析 per-method seeds
    method_seed_map = {}
    for item in args.method_seeds:
        if ':' in item:
            method, seed_str = item.split(':', 1)
            method_seed_map[method] = int(seed_str)
    
    def get_seed_for_method(method_name: str) -> int:
        """获取特定方法的 seed"""
        return method_seed_map.get(method_name, args.seed)
    
    # 处理任务参数
    if args.task_type:
        # 兼容旧用法
        train_task = args.task_type
        eval_tasks = [args.task_type]
    elif args.train_task:
        train_task = args.train_task
        if args.cross_task:
            eval_tasks = args.all_tasks
        elif args.eval_task:
            eval_tasks = [args.eval_task]
        else:
            eval_tasks = [train_task]
    else:
        parser.error("请指定 --task_type 或 --train_task")
        return
    
    features_dir = Path(args.features_dir)
    models_dir = Path(args.models_dir)
    
    logger.info("=" * 60)
    logger.info("Quick Evaluation (Fixed Version v3)")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Train task: {train_task}")
    logger.info(f"Eval tasks: {eval_tasks}")
    logger.info(f"Cross-task: {args.cross_task}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Default seed: {args.seed}")
    if method_seed_map:
        logger.info(f"Method-specific seeds: {method_seed_map}")
    logger.info("=" * 60)
    
    all_results = []
    
    # 对每个评估任务
    for eval_task in eval_tasks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating on: {eval_task}_test")
        logger.info(f"{'='*60}")
        
        # 按照 seed 分组处理方法
        # 对于每个唯一的 seed，加载一次对应的测试特征
        methods_by_seed = {}
        for method_name in args.methods:
            method_seed = get_seed_for_method(method_name)
            if method_seed not in methods_by_seed:
                methods_by_seed[method_seed] = []
            methods_by_seed[method_seed].append(method_name)
        
        # 处理每个 seed 的方法组
        for seed_val, methods_for_seed in methods_by_seed.items():
            logger.info(f"\n--- Processing methods with seed={seed_val}: {methods_for_seed} ---")
            
            # 查找该 seed 对应的测试集特征目录
            test_feat_dir = find_test_features_dir(
                features_dir, args.dataset, args.model, seed_val, eval_task
            )
            
            if test_feat_dir is None:
                logger.error(f"Test features directory not found for seed={seed_val}, task={eval_task}")
                logger.error(f"Searched: {features_dir}/{args.dataset}/{args.model}/seed_{seed_val}/{eval_task}_test")
                for method_name in methods_for_seed:
                    all_results.append({
                        "method": method_name,
                        "train_task": train_task,
                        "eval_task": eval_task,
                        "seed": seed_val,
                        "error": f"Test features not found for seed={seed_val}"
                    })
                continue
            
            logger.info(f"Loading test features from: {test_feat_dir}")
            
            # 加载测试集特征
            try:
                test_features, test_labels = load_features_from_dir(test_feat_dir)
            except Exception as e:
                logger.error(f"Failed to load test features: {e}")
                import traceback
                traceback.print_exc()
                for method_name in methods_for_seed:
                    all_results.append({
                        "method": method_name,
                        "train_task": train_task,
                        "eval_task": eval_task,
                        "seed": seed_val,
                        "error": str(e)
                    })
                continue
            
            n_positive = sum(test_labels)
            n_negative = len(test_labels) - n_positive
            logger.info(f"Loaded {len(test_features)} test samples ({n_positive} positive, {n_negative} negative)")
            
            if n_positive == 0 or n_negative == 0:
                logger.warning(f"⚠️ Only one class in test set! Results may be meaningless.")
            
            # 评估该 seed 对应的每个方法
            for method_name in methods_for_seed:
                logger.info(f"\n>>> Evaluating: {method_name} (seed={seed_val}, trained on {train_task}, eval on {eval_task})")
                
                # 查找模型（使用该方法的 seed）
                model_path = find_model_path(
                    models_dir, args.dataset, args.model, seed_val, 
                    train_task, method_name, args.level
                )
                
                if model_path is None:
                    logger.error(f"  Model not found for {method_name}")
                    logger.error(f"  Searched: {models_dir}/{args.dataset}/{args.model}/seed_{seed_val}/{train_task}/{method_name}/")
                    all_results.append({
                        "method": method_name,
                        "train_task": train_task,
                        "eval_task": eval_task,
                        "seed": seed_val,
                        "error": "Model file not found"
                    })
                    continue
                
                logger.info(f"  Model: {model_path}")
                
                result = evaluate_method(method_name, model_path, test_features, test_labels)
                result["train_task"] = train_task
                result["eval_task"] = eval_task
                result["seed"] = seed_val
                all_results.append(result)
                
                if "error" in result:
                    logger.error(f"  Error: {result['error']}")
                else:
                    logger.info(f"  AUROC: {result.get('auroc', 0):.4f}")
                    logger.info(f"  AUPR:  {result.get('aupr', 0):.4f}")
                    logger.info(f"  F1:    {result.get('f1', 0):.4f}")
    
    # 汇总结果
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    
    # 检查是否有不同的 seed
    has_multiple_seeds = len(set(r.get("seed", args.seed) for r in all_results)) > 1
    
    if args.cross_task or has_multiple_seeds:
        print(f"{'Method':<15} {'Seed':<6} {'Train':<12} {'Eval':<12} {'AUROC':<10} {'AUPR':<10} {'F1':<10}")
        print("-" * 90)
        for result in all_results:
            seed_val = result.get("seed", args.seed)
            if "error" in result:
                print(f"{result['method']:<15} {seed_val:<6} {result.get('train_task', 'N/A'):<12} {result.get('eval_task', 'N/A'):<12} ERROR: {result['error'][:20]}")
            else:
                print(f"{result['method']:<15} {seed_val:<6} {result.get('train_task', 'N/A'):<12} {result.get('eval_task', 'N/A'):<12} "
                      f"{result.get('auroc', 0):.4f}     {result.get('aupr', 0):.4f}     {result.get('f1', 0):.4f}")
    else:
        print(f"{'Method':<20} {'AUROC':<10} {'AUPR':<10} {'F1':<10} {'N_samples':<10}")
        print("-" * 60)
        for result in all_results:
            if "error" in result:
                print(f"{result['method']:<20} ERROR: {result['error'][:30]}")
            else:
                print(f"{result['method']:<20} {result.get('auroc', 0):.4f}     {result.get('aupr', 0):.4f}     "
                      f"{result.get('f1', 0):.4f}     {result.get('n_samples', 0)}")
    
    print("=" * 90)
    
    # 保存 eval_results.json 到 results_dir（可被 aggregate_results.py 识别）
    if args.save_eval_results:
        results_base = Path(args.results_dir)
        for result in all_results:
            if "error" in result:
                continue
            
            method = result.get("method", "unknown")
            train_t = result.get("train_task", train_task)
            eval_t = result.get("eval_task", train_task)
            method_seed = result.get("seed", args.seed)  # 使用该方法的 seed
            
            # 构建路径: results/{dataset}/{model}/seed_{seed}/train_{train_task}_eval_{eval_task}/{method}/
            result_dir = (results_base / args.dataset / args.model / 
                         f"seed_{method_seed}" / f"train_{train_t}_eval_{eval_t}" / method)
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # 转换为 aggregate_results.py 期望的格式
            eval_results = {
                "level": args.level,
                "config": {
                    "dataset": args.dataset,
                    "model": args.model,
                    "method": method,
                    "seed": method_seed,
                    "train_task": train_t,
                    "eval_task": eval_t,
                },
                "train_metrics": {},  # quick_eval 不计算训练集指标
                "test_metrics": {
                    "auroc": result.get("auroc", 0),
                    "aupr": result.get("aupr", 0),
                    "f1": result.get("f1", 0),
                    "n_samples": result.get("n_samples", 0),
                    "n_positive": result.get("n_positive", 0),
                    "n_negative": result.get("n_negative", 0),
                },
            }
            
            eval_file = result_dir / "eval_results.json"
            with open(eval_file, "w") as f:
                json.dump(eval_results, f, indent=2)
            logger.info(f"Saved: {eval_file}")
        
        logger.info(f"\nEval results saved to {results_base}")
        logger.info("Run `python scripts/aggregate_results.py` to aggregate all results.")
    
    # 保存结果（旧格式）
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "config": {
                    "dataset": args.dataset,
                    "model": args.model,
                    "train_task": train_task,
                    "eval_tasks": eval_tasks,
                    "cross_task": args.cross_task,
                    "seed": args.seed,
                },
                "results": all_results,
            }, f, indent=2)
        logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()