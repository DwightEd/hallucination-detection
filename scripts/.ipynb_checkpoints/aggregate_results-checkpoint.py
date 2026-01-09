#!/usr/bin/env python3
"""Aggregate results from all experiments.

Generates:
- summary.json: Complete results with train/test metrics
- comparison.csv: Method comparison table
- performance.csv: Performance metrics (time, memory, model size)

Directory structure:
  outputs/models/{dataset}/{model}/seed_{seed}/{task_type}/{method}/eval_results.json
"""
import sys
import json
import logging
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import setup_logging

logger = logging.getLogger(__name__)


def parse_path_info(eval_file: Path) -> Dict[str, Any]:
    """Parse experiment info from path.
    
    Path format: outputs/models/{dataset}/{model}/seed_{seed}/{task_type}/{method}/eval_results.json
    """
    parts = eval_file.parts
    
    try:
        models_idx = parts.index("models")
    except ValueError:
        return {}
    
    # Need at least: models/dataset/model/seed_X/task_type/method/eval_results.json
    if len(parts) < models_idx + 6:
        return {}
    
    dataset = parts[models_idx + 1]
    model = parts[models_idx + 2]
    seed_dir = parts[models_idx + 3]
    task_type = parts[models_idx + 4]
    method = parts[models_idx + 5]
    
    # Validate seed_dir format
    seed_match = re.match(r"seed_(\d+)", seed_dir)
    if not seed_match:
        return {}
    
    return {
        "dataset": dataset,
        "task_type": task_type,
        "model": model,
        "method": method,
        "seed": int(seed_match.group(1)),
    }


def load_experiment_data(eval_file: Path) -> Dict[str, Any]:
    """Load experiment data from eval_results.json."""
    with open(eval_file) as f:
        data = json.load(f)
    
    path_info = parse_path_info(eval_file)
    if not path_info:
        return {}
    
    # Extract metrics (support both old and new format)
    metrics = data.get("metrics", {})
    train_metrics = data.get("train_metrics", {})
    test_metrics = data.get("test_metrics", metrics)
    
    # Load performance metrics if available
    perf_metrics = {}
    probe_dir = eval_file.parent / "probe"
    train_metrics_file = probe_dir / "train_metrics.json"
    if train_metrics_file.exists():
        with open(train_metrics_file) as f:
            train_data = json.load(f)
            perf_metrics = train_data.get("performance", {})
    
    return {
        **path_info,
        # Test metrics (out-of-sample)
        "test_auroc": test_metrics.get("auroc", metrics.get("auroc", 0)),
        "test_aupr": test_metrics.get("aupr", metrics.get("aupr", metrics.get("auprc", 0))),
        "test_f1": test_metrics.get("f1", metrics.get("f1", 0)),
        "test_n_samples": test_metrics.get("n_samples", 0),
        # Train metrics (in-sample)
        "train_auroc": train_metrics.get("auroc", train_metrics.get("train_auroc", 0)),
        "train_aupr": train_metrics.get("aupr", train_metrics.get("train_aupr", 0)),
        "train_n_samples": train_metrics.get("n_samples", train_metrics.get("train_n_samples", 0)),
        # Performance metrics
        "training_time_s": perf_metrics.get("training_time_seconds", 0),
        "peak_cpu_mb": perf_metrics.get("peak_cpu_memory_mb", 0),
        "peak_gpu_mb": perf_metrics.get("peak_gpu_memory_mb", 0),
        "model_size_mb": perf_metrics.get("model_size_mb", 0),
        # File path
        "path": str(eval_file),
    }


def format_comparison_table(experiments: List[Dict], by_field: str = "method") -> str:
    """Format experiments as comparison table."""
    if not experiments:
        return "No experiments found."
    
    # Group by field
    groups = defaultdict(list)
    for exp in experiments:
        groups[exp.get(by_field, "unknown")].append(exp)
    
    # Calculate averages
    rows = []
    for name, exps in sorted(groups.items()):
        test_aurocs = [e["test_auroc"] for e in exps if e["test_auroc"]]
        test_auprs = [e["test_aupr"] for e in exps if e["test_aupr"]]
        train_aurocs = [e["train_auroc"] for e in exps if e["train_auroc"]]
        train_auprs = [e["train_aupr"] for e in exps if e["train_aupr"]]
        times = [e["training_time_s"] for e in exps if e["training_time_s"]]
        sizes = [e["model_size_mb"] for e in exps if e["model_size_mb"]]
        
        rows.append({
            "name": name,
            "n": len(exps),
            "train_auroc": sum(train_aurocs) / len(train_aurocs) if train_aurocs else 0,
            "train_aupr": sum(train_auprs) / len(train_auprs) if train_auprs else 0,
            "test_auroc": sum(test_aurocs) / len(test_aurocs) if test_aurocs else 0,
            "test_aupr": sum(test_auprs) / len(test_auprs) if test_auprs else 0,
            "time_s": sum(times) / len(times) if times else 0,
            "size_mb": sum(sizes) / len(sizes) if sizes else 0,
        })
    
    # Format table
    header = f"{'Method':<20} {'N':>4} │ {'Train AUROC':>11} {'Train AUPR':>11} │ {'Test AUROC':>11} {'Test AUPR':>11} │ {'Time(s)':>9} {'Size(MB)':>9}"
    sep = "─" * len(header)
    
    lines = [sep, header, sep]
    
    for r in rows:
        line = f"{r['name']:<20} {r['n']:>4} │ {r['train_auroc']:>11.4f} {r['train_aupr']:>11.4f} │ {r['test_auroc']:>11.4f} {r['test_aupr']:>11.4f} │ {r['time_s']:>9.2f} {r['size_mb']:>9.2f}"
        lines.append(line)
    
    lines.append(sep)
    return "\n".join(lines)


def save_comparison_csv(experiments: List[Dict], output_path: Path):
    """Save comparison results as CSV."""
    if not experiments:
        return
    
    # Group by dataset_task_model
    groups = defaultdict(lambda: defaultdict(dict))
    for exp in experiments:
        key = f"{exp['dataset']}_{exp['task_type']}_{exp['model']}"
        groups[key][exp["method"]] = exp
    
    # Get all methods
    all_methods = sorted(set(e["method"] for e in experiments))
    
    with open(output_path, "w") as f:
        # Header
        header = ["Dataset_Task_Model"]
        for method in all_methods:
            header.extend([
                f"{method}_train_auroc",
                f"{method}_train_aupr",
                f"{method}_test_auroc",
                f"{method}_test_aupr",
            ])
        f.write(",".join(header) + "\n")
        
        # Data rows
        for key in sorted(groups.keys()):
            row = [key]
            for method in all_methods:
                exp = groups[key].get(method, {})
                row.extend([
                    f"{exp.get('train_auroc', 0):.4f}",
                    f"{exp.get('train_aupr', 0):.4f}",
                    f"{exp.get('test_auroc', 0):.4f}",
                    f"{exp.get('test_aupr', 0):.4f}",
                ])
            f.write(",".join(row) + "\n")


def save_performance_csv(experiments: List[Dict], output_path: Path):
    """Save performance metrics as CSV."""
    if not experiments:
        return
    
    with open(output_path, "w") as f:
        # Header
        f.write("method,dataset,task_type,model,seed,training_time_s,peak_cpu_mb,peak_gpu_mb,model_size_mb\n")
        
        # Data rows
        for exp in sorted(experiments, key=lambda x: (x["method"], x["dataset"])):
            row = [
                exp["method"],
                exp["dataset"],
                exp["task_type"],
                exp["model"],
                str(exp["seed"]),
                f"{exp['training_time_s']:.2f}",
                f"{exp['peak_cpu_mb']:.1f}",
                f"{exp['peak_gpu_mb']:.1f}",
                f"{exp['model_size_mb']:.2f}",
            ]
            f.write(",".join(row) + "\n")


def main():
    setup_logging(level=logging.INFO)

    logger.info("=" * 80)
    logger.info("Aggregating Results")
    logger.info("=" * 80)

    models_dir = Path("outputs/models")
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Collect all results
    experiments = []
    eval_files = list(models_dir.rglob("eval_results.json"))
    logger.info(f"Found {len(eval_files)} evaluation files")

    for eval_file in eval_files:
        try:
            exp_data = load_experiment_data(eval_file)
            if exp_data:
                experiments.append(exp_data)
        except Exception as e:
            logger.warning(f"Failed to load {eval_file}: {e}")

    if not experiments:
        logger.warning("No experiments found!")
        return

    # Group by various dimensions
    by_method = defaultdict(list)
    by_task = defaultdict(list)
    by_dataset = defaultdict(list)
    
    for exp in experiments:
        by_method[exp["method"]].append(exp)
        by_task[exp["task_type"]].append(exp)
        by_dataset[exp["dataset"]].append(exp)

    # ==========================================================================
    # Print Results
    # ==========================================================================
    
    print("\n" + "=" * 90)
    print("HALLUCINATION DETECTION RESULTS")
    print("=" * 90)
    print(f"\nTotal experiments: {len(experiments)}")
    print(f"Datasets: {', '.join(sorted(by_dataset.keys()))}")
    print(f"Task types: {', '.join(sorted(by_task.keys()))}")
    print(f"Methods: {', '.join(sorted(by_method.keys()))}")

    # Method comparison
    print("\n" + "=" * 90)
    print("METHOD COMPARISON (Train vs Test)")
    print("=" * 90)
    print(format_comparison_table(experiments, "method"))

    # By task type
    print("\n" + "=" * 90)
    print("RESULTS BY TASK TYPE")
    print("=" * 90)
    for task_type in sorted(by_task.keys()):
        task_exps = by_task[task_type]
        test_aurocs = [e["test_auroc"] for e in task_exps if e["test_auroc"]]
        test_auprs = [e["test_aupr"] for e in task_exps if e["test_aupr"]]
        
        mean_auroc = sum(test_aurocs) / len(test_aurocs) if test_aurocs else 0
        mean_aupr = sum(test_auprs) / len(test_auprs) if test_auprs else 0
        best = max(task_exps, key=lambda x: x["test_auroc"])["method"] if task_exps else "N/A"
        
        print(f"\n{task_type}:")
        print(f"  Mean Test AUROC: {mean_auroc:.4f}")
        print(f"  Mean Test AUPR:  {mean_aupr:.4f}")
        print(f"  Best Method:     {best}")

    # Performance summary
    print("\n" + "=" * 90)
    print("PERFORMANCE SUMMARY BY METHOD")
    print("=" * 90)
    print(f"{'Method':<20} {'Avg Time(s)':>12} {'Avg CPU(MB)':>12} {'Avg GPU(MB)':>12} {'Avg Size(MB)':>12}")
    print("-" * 68)
    
    for method in sorted(by_method.keys()):
        exps = by_method[method]
        times = [e["training_time_s"] for e in exps if e["training_time_s"]]
        cpus = [e["peak_cpu_mb"] for e in exps if e["peak_cpu_mb"]]
        gpus = [e["peak_gpu_mb"] for e in exps if e["peak_gpu_mb"]]
        sizes = [e["model_size_mb"] for e in exps if e["model_size_mb"]]
        
        avg_time = sum(times) / len(times) if times else 0
        avg_cpu = sum(cpus) / len(cpus) if cpus else 0
        avg_gpu = sum(gpus) / len(gpus) if gpus else 0
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        
        print(f"{method:<20} {avg_time:>12.2f} {avg_cpu:>12.1f} {avg_gpu:>12.1f} {avg_size:>12.2f}")

    # ==========================================================================
    # Save Results
    # ==========================================================================
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_experiments": len(experiments),
        "datasets": list(set(e["dataset"] for e in experiments)),
        "task_types": list(set(e["task_type"] for e in experiments)),
        "methods": list(set(e["method"] for e in experiments)),
        "models": list(set(e["model"] for e in experiments)),
        "by_method": {
            method: {
                "mean_train_auroc": sum(e["train_auroc"] for e in exps) / len(exps) if exps else 0,
                "mean_train_aupr": sum(e["train_aupr"] for e in exps) / len(exps) if exps else 0,
                "mean_test_auroc": sum(e["test_auroc"] for e in exps) / len(exps) if exps else 0,
                "mean_test_aupr": sum(e["test_aupr"] for e in exps) / len(exps) if exps else 0,
                "mean_training_time_s": sum(e["training_time_s"] for e in exps) / len(exps) if exps else 0,
                "mean_model_size_mb": sum(e["model_size_mb"] for e in exps) / len(exps) if exps else 0,
                "n_experiments": len(exps),
            }
            for method, exps in by_method.items()
        },
        "by_task_type": {
            task: {
                "mean_test_auroc": sum(e["test_auroc"] for e in exps) / len(exps) if exps else 0,
                "mean_test_aupr": sum(e["test_aupr"] for e in exps) / len(exps) if exps else 0,
                "best_method": max(exps, key=lambda x: x["test_auroc"])["method"] if exps else None,
                "n_experiments": len(exps),
            }
            for task, exps in by_task.items()
        },
        "experiments": experiments,
    }

    # Save JSON
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")

    # Save comparison CSV
    comparison_path = results_dir / "comparison.csv"
    save_comparison_csv(experiments, comparison_path)
    logger.info(f"Comparison CSV saved to {comparison_path}")

    # Save performance CSV
    performance_path = results_dir / "performance.csv"
    save_performance_csv(experiments, performance_path)
    logger.info(f"Performance CSV saved to {performance_path}")

    print("\n" + "=" * 90)
    print("Results saved to outputs/results/")
    print("  - summary.json: Complete results")
    print("  - comparison.csv: Method comparison by dataset/task")
    print("  - performance.csv: Performance metrics")
    print("=" * 90)


if __name__ == "__main__":
    main()
