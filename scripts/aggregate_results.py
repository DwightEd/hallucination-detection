#!/usr/bin/env python3
"""Aggregate results from all experiments.

按任务类型汇总结果，输出每个数据集的每个任务类型的 AUROC 和 AUPRC 指标。

新目录结构：
  outputs/models/{dataset}/{model}/seed_{seed}/{task_type}/{method}/eval_results.json

旧目录结构（向后兼容）：
  outputs/models/{dataset}_{task_type}/{model}/{method}/seed_{seed}/eval_results.json

输出：
  - summary.json: 详细汇总结果
  - results_by_task.csv: 按任务类型的结果表格
"""
import sys
import json
import logging
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import setup_logging

logger = logging.getLogger(__name__)


def parse_path_info_new(eval_file: Path) -> Dict[str, str]:
    """从新目录结构路径中解析实验信息。
    
    新路径格式: outputs/models/{dataset}/{model}/seed_{seed}/{task_type}/{method}/eval_results.json
    
    Returns:
        包含 dataset, task_type, model, method, seed 的字典
    """
    parts = eval_file.parts
    
    try:
        models_idx = parts.index("models")
    except ValueError:
        return {}
    
    # 新结构需要 models 后面至少6个部分: dataset/model/seed_X/task_type/method/eval_results.json
    if len(parts) < models_idx + 6:
        return {}
    
    dataset = parts[models_idx + 1]
    model = parts[models_idx + 2]
    seed_dir = parts[models_idx + 3]
    task_type = parts[models_idx + 4]
    method = parts[models_idx + 5]
    
    # 验证 seed_dir 格式
    seed_match = re.match(r"seed_(\d+)", seed_dir)
    if not seed_match:
        return {}
    
    seed = seed_match.group(1)
    
    return {
        "dataset": dataset,
        "task_type": task_type,
        "model": model,
        "method": method,
        "seed": int(seed),
    }


def parse_path_info_old(eval_file: Path) -> Dict[str, str]:
    """从旧目录结构路径中解析实验信息。
    
    旧路径格式: outputs/models/{dataset}_{task_type}/{model}/{method}/seed_{seed}/eval_results.json
    
    Returns:
        包含 dataset, task_type, model, method, seed 的字典
    """
    parts = eval_file.parts
    
    try:
        models_idx = parts.index("models")
    except ValueError:
        return {}
    
    if len(parts) < models_idx + 5:
        return {}
    
    dataset_task = parts[models_idx + 1]  # e.g., "ragtruth_QA"
    model = parts[models_idx + 2]
    method = parts[models_idx + 3]
    seed_dir = parts[models_idx + 4]  # e.g., "seed_42"
    
    # 解析 dataset 和 task_type
    if "_" in dataset_task:
        last_underscore = dataset_task.rfind("_")
        dataset = dataset_task[:last_underscore]
        task_type = dataset_task[last_underscore + 1:]
    else:
        dataset = dataset_task
        task_type = "all"
    
    # 解析 seed
    seed_match = re.match(r"seed_(\d+)", seed_dir)
    seed = seed_match.group(1) if seed_match else "0"
    
    return {
        "dataset": dataset,
        "task_type": task_type,
        "model": model,
        "method": method,
        "seed": int(seed),
    }


def parse_path_info(eval_file: Path) -> Dict[str, str]:
    """自动检测并解析路径信息。
    
    尝试新格式，如果失败则尝试旧格式。
    """
    # 先尝试新格式
    info = parse_path_info_new(eval_file)
    if info:
        return info
    
    # 回退到旧格式
    return parse_path_info_old(eval_file)


def main():
    setup_logging(level=logging.INFO)

    logger.info("=" * 80)
    logger.info("Aggregating Results by Task Type")
    logger.info("=" * 80)

    models_dir = Path("outputs/models")
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有结果
    all_experiments = []
    by_dataset_task = defaultdict(list)  # {dataset}_{task_type} -> [experiments]
    by_method = defaultdict(list)
    by_task_type = defaultdict(list)

    # 查找所有 eval_results.json 文件
    eval_files = list(models_dir.rglob("eval_results.json"))
    logger.info(f"Found {len(eval_files)} evaluation result files")

    for eval_file in eval_files:
        try:
            with open(eval_file) as f:
                eval_data = json.load(f)

            path_info = parse_path_info(eval_file)
            
            if not path_info:
                logger.warning(f"Could not parse path: {eval_file}")
                continue
            
            metrics = eval_data.get("metrics", {})

            experiment = {
                "dataset": path_info.get("dataset", "unknown"),
                "task_type": path_info.get("task_type", "all"),
                "model": path_info.get("model", "unknown"),
                "method": path_info.get("method", "unknown"),
                "seed": path_info.get("seed", 0),
                "auroc": metrics.get("auroc", 0),
                "auprc": metrics.get("auprc", 0),
                "f1": metrics.get("f1", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "accuracy": metrics.get("accuracy", 0),
                "n_samples": metrics.get("n_samples", eval_data.get("n_samples", 0)),
                "n_positive": metrics.get("n_positive", 0),
                "n_negative": metrics.get("n_negative", 0),
                "path": str(eval_file),
            }

            all_experiments.append(experiment)
            
            # 分组
            key = f"{experiment['dataset']}_{experiment['task_type']}"
            by_dataset_task[key].append(experiment)
            by_method[experiment["method"]].append(experiment)
            by_task_type[experiment["task_type"]].append(experiment)

        except Exception as e:
            logger.warning(f"Failed to load {eval_file}: {e}")

    # ===========================================================================
    # 构建汇总结果
    # ===========================================================================
    
    # 1. 按数据集+任务类型汇总（主要结果）
    results_by_dataset_task = {}
    for key, exps in by_dataset_task.items():
        methods_results = {}
        for exp in exps:
            methods_results[exp["method"]] = {
                "auroc": exp["auroc"],
                "auprc": exp["auprc"],
                "f1": exp["f1"],
                "n_samples": exp["n_samples"],
            }
        
        # 计算该任务类型的平均值
        aurocs = [e["auroc"] for e in exps]
        auprcs = [e["auprc"] for e in exps]
        
        results_by_dataset_task[key] = {
            "methods": methods_results,
            "mean_auroc": sum(aurocs) / len(aurocs) if aurocs else 0,
            "mean_auprc": sum(auprcs) / len(auprcs) if auprcs else 0,
            "best_method": max(exps, key=lambda x: x["auroc"])["method"] if exps else None,
            "n_methods": len(methods_results),
        }

    # 2. 按方法汇总
    results_by_method = {}
    for method, exps in by_method.items():
        aurocs = [e["auroc"] for e in exps]
        auprcs = [e["auprc"] for e in exps]
        results_by_method[method] = {
            "mean_auroc": sum(aurocs) / len(aurocs) if aurocs else 0,
            "mean_auprc": sum(auprcs) / len(auprcs) if auprcs else 0,
            "max_auroc": max(aurocs) if aurocs else 0,
            "min_auroc": min(aurocs) if aurocs else 0,
            "n_experiments": len(exps),
        }

    # 3. 按任务类型汇总
    results_by_task_type = {}
    for task_type, exps in by_task_type.items():
        aurocs = [e["auroc"] for e in exps]
        auprcs = [e["auprc"] for e in exps]
        results_by_task_type[task_type] = {
            "mean_auroc": sum(aurocs) / len(aurocs) if aurocs else 0,
            "mean_auprc": sum(auprcs) / len(auprcs) if auprcs else 0,
            "max_auroc": max(aurocs) if aurocs else 0,
            "best_method": max(exps, key=lambda x: x["auroc"])["method"] if exps else None,
            "n_experiments": len(exps),
        }

    # ===========================================================================
    # 保存结果
    # ===========================================================================
    
    summary = {
        "total_experiments": len(all_experiments),
        "datasets": list(set(e["dataset"] for e in all_experiments)),
        "task_types": list(set(e["task_type"] for e in all_experiments)),
        "methods": list(set(e["method"] for e in all_experiments)),
        "models": list(set(e["model"] for e in all_experiments)),
        
        # 主要结果：按数据集+任务类型
        "by_dataset_task": results_by_dataset_task,
        
        # 按方法汇总
        "by_method": results_by_method,
        
        # 按任务类型汇总
        "by_task_type": results_by_task_type,
        
        # 所有实验详情
        "experiments": all_experiments,
    }

    # 保存 JSON
    output_path = results_dir / "summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=list)
    logger.info(f"Summary saved to {output_path}")

    # 保存 CSV（按任务类型的结果表格）
    csv_path = results_dir / "results_by_task.csv"
    save_results_csv(results_by_dataset_task, by_method.keys(), csv_path)
    logger.info(f"CSV results saved to {csv_path}")

    # ===========================================================================
    # 打印结果
    # ===========================================================================
    print_results(summary, results_by_dataset_task, results_by_method, results_by_task_type)


def save_results_csv(results_by_dataset_task: Dict, methods: List[str], output_path: Path):
    """保存结果为 CSV 格式。"""
    methods = sorted(methods)
    
    with open(output_path, "w") as f:
        # Header
        header = ["Dataset_TaskType"]
        for method in methods:
            header.extend([f"{method}_AUROC", f"{method}_AUPRC"])
        header.append("Mean_AUROC")
        header.append("Mean_AUPRC")
        f.write(",".join(header) + "\n")
        
        # Data rows
        for key in sorted(results_by_dataset_task.keys()):
            data = results_by_dataset_task[key]
            row = [key]
            for method in methods:
                method_data = data["methods"].get(method, {})
                row.append(f"{method_data.get('auroc', 0):.4f}")
                row.append(f"{method_data.get('auprc', 0):.4f}")
            row.append(f"{data['mean_auroc']:.4f}")
            row.append(f"{data['mean_auprc']:.4f}")
            f.write(",".join(row) + "\n")


def print_results(summary: Dict, by_dataset_task: Dict, by_method: Dict, by_task_type: Dict):
    """打印汇总结果。"""
    print("\n" + "=" * 90)
    print("HALLUCINATION DETECTION RESULTS SUMMARY")
    print("=" * 90)
    
    print(f"\nTotal experiments: {summary['total_experiments']}")
    print(f"Datasets: {', '.join(summary['datasets'])}")
    print(f"Task types: {', '.join(summary['task_types'])}")
    print(f"Methods: {', '.join(summary['methods'])}")
    print(f"Models: {', '.join(summary['models'])}")

    # ===========================================================================
    # 主要结果：每个数据集的每个任务类型的 AUROC 和 AUPRC
    # ===========================================================================
    print("\n" + "=" * 90)
    print("RESULTS BY DATASET AND TASK TYPE (AUROC / AUPRC)")
    print("=" * 90)
    
    # 获取所有方法
    all_methods = sorted(by_method.keys())
    
    # 打印表头
    header = f"{'Dataset_Task':<25}"
    for method in all_methods:
        header += f" {method[:12]:>12}"
    header += f" {'Mean':>12}"
    print(header)
    print("-" * (25 + 13 * (len(all_methods) + 1)))
    
    # 打印每个数据集+任务类型的结果
    for key in sorted(by_dataset_task.keys()):
        data = by_dataset_task[key]
        
        # AUROC 行
        row_auroc = f"{key:<25}"
        for method in all_methods:
            method_data = data["methods"].get(method, {})
            auroc = method_data.get("auroc", 0)
            row_auroc += f" {auroc:>12.4f}"
        row_auroc += f" {data['mean_auroc']:>12.4f}"
        print(row_auroc)
        
        # AUPRC 行（缩进显示）
        row_auprc = f"{'  (AUPRC)':<25}"
        for method in all_methods:
            method_data = data["methods"].get(method, {})
            auprc = method_data.get("auprc", 0)
            row_auprc += f" {auprc:>12.4f}"
        row_auprc += f" {data['mean_auprc']:>12.4f}"
        print(row_auprc)
        print()

    # ===========================================================================
    # 按方法汇总
    # ===========================================================================
    print("\n" + "-" * 90)
    print("SUMMARY BY METHOD (averaged across all task types)")
    print("-" * 90)
    print(f"{'Method':<20} {'Mean AUROC':>12} {'Mean AUPRC':>12} {'Max AUROC':>12} {'N':>6}")
    print("-" * 62)
    
    for method in sorted(by_method.keys()):
        stats = by_method[method]
        print(f"{method:<20} {stats['mean_auroc']:>12.4f} {stats['mean_auprc']:>12.4f} "
              f"{stats['max_auroc']:>12.4f} {stats['n_experiments']:>6}")

    # ===========================================================================
    # 按任务类型汇总
    # ===========================================================================
    print("\n" + "-" * 90)
    print("SUMMARY BY TASK TYPE (averaged across all methods)")
    print("-" * 90)
    print(f"{'Task Type':<20} {'Mean AUROC':>12} {'Mean AUPRC':>12} {'Best Method':>15} {'N':>6}")
    print("-" * 65)
    
    for task_type in sorted(by_task_type.keys()):
        stats = by_task_type[task_type]
        best = stats.get('best_method', 'N/A')[:15]
        print(f"{task_type:<20} {stats['mean_auroc']:>12.4f} {stats['mean_auprc']:>12.4f} "
              f"{best:>15} {stats['n_experiments']:>6}")

    print("\n" + "=" * 90)
    print("Results saved to outputs/results/summary.json and outputs/results/results_by_task.csv")
    print("=" * 90)


if __name__ == "__main__":
    main()
