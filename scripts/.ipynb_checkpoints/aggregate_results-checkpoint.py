#!/usr/bin/env python3
"""Aggregate results from all experiments.

Generates:
- summary.json: Complete results with train/test metrics
- comparison_sample.csv: Sample-level comparison
- comparison_token.csv: Token-level comparison

Directory structure (v2 - with level):
  outputs/models/{dataset}/{model}/seed_{seed}/{task}/{method}/{level}/eval_results.json
  outputs/results/{dataset}/{model}/seed_{seed}/train_{train_task}_eval_{eval_task}/{method}/eval_results.json
"""
import sys
import json
import logging
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
    
    Supports both old and new path structures:
    - Old: outputs/models/{dataset}/{model}/seed_{seed}/{task}/{method}/eval_results.json
    - Old: outputs/models/{dataset}/{model}/seed_{seed}/{task}/{method}/probe/eval_results.json
    - New: outputs/models/{dataset}/{model}/seed_{seed}/{task}/{method}/{level}/eval_results.json
    """
    parts = eval_file.parts
    
    # Try results directory format (for cross-task evaluation)
    try:
        results_idx = parts.index("results")
        if len(parts) >= results_idx + 6:
            dataset = parts[results_idx + 1]
            model = parts[results_idx + 2]
            seed_dir = parts[results_idx + 3]
            train_eval_dir = parts[results_idx + 4]
            method = parts[results_idx + 5]
            
            # Check if there's a level directory
            level = "sample"  # default
            if len(parts) >= results_idx + 7:
                potential_level = parts[results_idx + 6]
                if potential_level in ["sample", "token"]:
                    level = potential_level
            
            seed_match = re.match(r"seed_(\d+)", seed_dir)
            train_eval_match = re.match(r"train_(.+)_eval_(.+)", train_eval_dir)
            
            if seed_match and train_eval_match:
                train_task = train_eval_match.group(1)
                eval_task = train_eval_match.group(2)
                return {
                    "dataset": dataset,
                    "model": model,
                    "method": method,
                    "level": level,
                    "seed": int(seed_match.group(1)),
                    "train_task": train_task,
                    "eval_task": eval_task,
                    "is_cross_task": train_task != eval_task,
                }
    except ValueError:
        pass
    
    # Try models directory format (both old and new)
    try:
        models_idx = parts.index("models")
        if len(parts) >= models_idx + 6:
            dataset = parts[models_idx + 1]
            model = parts[models_idx + 2]
            seed_dir = parts[models_idx + 3]
            task_type = parts[models_idx + 4]
            method = parts[models_idx + 5]
            
            # Check for level directory (new format)
            level = "sample"  # default
            if len(parts) >= models_idx + 7:
                potential_level = parts[models_idx + 6]
                if potential_level in ["sample", "token", "probe"]:
                    # "probe" is treated as "sample" for backward compatibility
                    level = "sample" if potential_level == "probe" else potential_level
            
            seed_match = re.match(r"seed_(\d+)", seed_dir)
            if seed_match:
                return {
                    "dataset": dataset,
                    "model": model,
                    "method": method,
                    "level": level,
                    "seed": int(seed_match.group(1)),
                    "train_task": task_type,
                    "eval_task": task_type,
                    "is_cross_task": False,
                }
    except ValueError:
        pass
    
    return {}


def load_experiment_data(eval_file: Path) -> Dict[str, Any]:
    """Load experiment data from eval_results.json."""
    with open(eval_file) as f:
        data = json.load(f)
    
    path_info = parse_path_info(eval_file)
    if not path_info:
        return {}
    
    result = {**path_info}
    
    # 获取 level 信息 (优先从文件内容，然后从路径)
    level = data.get("level", path_info.get("level", "sample"))
    result["level"] = level
    
    # Sample-level metrics - 支持多种键名格式
    # 格式1: sample_level_train/sample_level_test (旧格式)
    # 格式2: train_metrics/test_metrics (evaluate.py 产生的格式)
    sample_train = data.get("sample_level_train", {})
    sample_test = data.get("sample_level_test", {})
    
    # 如果是 sample level，使用 train_metrics/test_metrics
    if level == "sample":
        if not sample_train:
            sample_train = data.get("train_metrics", {})
        if not sample_test:
            sample_test = data.get("test_metrics", data.get("metrics", {}))
    
    result.update({
        "sample_train_auroc": sample_train.get("auroc", 0),
        "sample_train_aupr": sample_train.get("aupr", 0),
        "sample_test_auroc": sample_test.get("auroc", 0),
        "sample_test_aupr": sample_test.get("aupr", 0),
        "sample_test_f1": sample_test.get("f1", 0),
    })
    
    # Token-level metrics
    token_train = data.get("token_level_train", {})
    token_test = data.get("token_level_test", {})
    
    # 如果是 token level，使用 train_metrics/test_metrics
    if level == "token":
        if not token_train:
            token_train = data.get("train_metrics", {})
        if not token_test:
            token_test = data.get("test_metrics", {})
    
    result.update({
        "token_train_auroc": token_train.get("auroc", 0),
        "token_train_aupr": token_train.get("aupr", 0),
        "token_test_auroc": token_test.get("auroc", 0),
        "token_test_aupr": token_test.get("aupr", 0),
        "token_test_f1": token_test.get("f1", 0),
    })
    
    result["path"] = str(eval_file)
    
    return result


def format_results_table(experiments: List[Dict], level: str = "sample") -> str:
    """Format experiments as comparison table."""
    if not experiments:
        return "No experiments found."
    
    prefix = f"{level}_"
    
    groups = defaultdict(list)
    for exp in experiments:
        # Group by method+level combination
        method_key = f"{exp.get('method', 'unknown')}_{exp.get('level', 'sample')}"
        groups[method_key].append(exp)
    
    rows = []
    for method_key, exps in sorted(groups.items()):
        train_aurocs = [e[f"{prefix}train_auroc"] for e in exps if e.get(f"{prefix}train_auroc")]
        test_aurocs = [e[f"{prefix}test_auroc"] for e in exps if e.get(f"{prefix}test_auroc")]
        test_auprs = [e[f"{prefix}test_aupr"] for e in exps if e.get(f"{prefix}test_aupr")]
        
        rows.append({
            "method": method_key,
            "n": len(exps),
            "train_auroc": sum(train_aurocs) / len(train_aurocs) if train_aurocs else 0,
            "test_auroc": sum(test_aurocs) / len(test_aurocs) if test_aurocs else 0,
            "test_aupr": sum(test_auprs) / len(test_auprs) if test_auprs else 0,
        })
    
    header = f"{'Method':<30} {'N':>4} │ {'Train AUROC':>11} │ {'Test AUROC':>11} {'Test AUPR':>11}"
    sep = "─" * len(header)
    
    lines = [sep, header, sep]
    for r in rows:
        line = f"{r['method']:<30} {r['n']:>4} │ {r['train_auroc']:>11.4f} │ {r['test_auroc']:>11.4f} {r['test_aupr']:>11.4f}"
        lines.append(line)
    lines.append(sep)
    
    return "\n".join(lines)


def save_comparison_csv(experiments: List[Dict], output_path: Path, level: str = "sample"):
    """Save comparison results as CSV."""
    if not experiments:
        return
    
    prefix = f"{level}_"
    
    with open(output_path, "w") as f:
        f.write("dataset,model,train_task,eval_task,method,level,is_cross_task,")
        f.write("train_auroc,train_aupr,test_auroc,test_aupr,test_f1\n")
        
        for exp in sorted(experiments, key=lambda x: (x["dataset"], x["method"], x.get("level", "sample"), x["train_task"], x["eval_task"])):
            row = [
                exp["dataset"],
                exp["model"],
                exp["train_task"],
                exp["eval_task"],
                exp["method"],
                exp.get("level", "sample"),
                "1" if exp["is_cross_task"] else "0",
                f"{exp.get(f'{prefix}train_auroc', 0):.4f}",
                f"{exp.get(f'{prefix}train_aupr', 0):.4f}",
                f"{exp.get(f'{prefix}test_auroc', 0):.4f}",
                f"{exp.get(f'{prefix}test_aupr', 0):.4f}",
                f"{exp.get(f'{prefix}test_f1', 0):.4f}",
            ]
            f.write(",".join(row) + "\n")


def main():
    setup_logging(level=logging.INFO)

    logger.info("=" * 80)
    logger.info("Aggregating Results")
    logger.info("=" * 80)

    search_dirs = [Path("outputs/models"), Path("outputs/results")]
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments = []
    for search_dir in search_dirs:
        if search_dir.exists():
            eval_files = list(search_dir.rglob("eval_results.json"))
            logger.info(f"Found {len(eval_files)} evaluation files in {search_dir}")
            
            for eval_file in eval_files:
                try:
                    exp_data = load_experiment_data(eval_file)
                    if exp_data:
                        experiments.append(exp_data)
                except Exception as e:
                    logger.warning(f"Failed to load {eval_file}: {e}")

    if not experiments:
        logger.warning("No experiments found!")
        with open(results_dir / "summary.json", "w") as f:
            json.dump({"error": "No experiments found", "generated_at": datetime.now().isoformat()}, f)
        return

    same_task_exps = [e for e in experiments if not e["is_cross_task"]]
    cross_task_exps = [e for e in experiments if e["is_cross_task"]]

    by_method = defaultdict(list)
    for exp in experiments:
        by_method[exp["method"]].append(exp)

    # Print results
    print("\n" + "=" * 90)
    print("HALLUCINATION DETECTION RESULTS")
    print("=" * 90)
    print(f"\nTotal experiments: {len(experiments)}")
    print(f"  Same-task evaluations: {len(same_task_exps)}")
    print(f"  Cross-task evaluations: {len(cross_task_exps)}")
    print(f"Methods: {', '.join(sorted(by_method.keys()))}")
    
    # Count by level
    sample_exps = [e for e in experiments if e.get("level", "sample") == "sample"]
    token_exps = [e for e in experiments if e.get("level") == "token"]
    print(f"  Sample-level: {len(sample_exps)}, Token-level: {len(token_exps)}")

    print("\n" + "=" * 90)
    print("SAMPLE-LEVEL RESULTS (Same-Task)")
    print("=" * 90)
    sample_same_task = [e for e in same_task_exps if e.get("level", "sample") == "sample"]
    print(format_results_table(sample_same_task, "sample"))

    token_same_task = [e for e in same_task_exps if e.get("level") == "token"]
    if token_same_task:
        print("\n" + "=" * 90)
        print("TOKEN-LEVEL RESULTS (Same-Task)")
        print("=" * 90)
        print(format_results_table(token_same_task, "token"))

    # Build summary
    def calc_avg(exps, key):
        vals = [e.get(key, 0) for e in exps if e.get(key)]
        return sum(vals) / len(vals) if vals else 0
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_experiments": len(experiments),
        "same_task_count": len(same_task_exps),
        "cross_task_count": len(cross_task_exps),
        "sample_level_count": len(sample_exps),
        "token_level_count": len(token_exps),
        "datasets": list(set(e["dataset"] for e in experiments)),
        "methods": list(set(e["method"] for e in experiments)),
        "levels": list(set(e.get("level", "sample") for e in experiments)),
        "sample_level_by_method": {
            method: {
                "mean_test_auroc": calc_avg([e for e in exps if not e["is_cross_task"] and e.get("level", "sample") == "sample"], "sample_test_auroc"),
                "mean_test_aupr": calc_avg([e for e in exps if not e["is_cross_task"] and e.get("level", "sample") == "sample"], "sample_test_aupr"),
                "n_experiments": len([e for e in exps if not e["is_cross_task"] and e.get("level", "sample") == "sample"]),
            }
            for method, exps in by_method.items()
        },
        "token_level_by_method": {
            method: {
                "mean_test_auroc": calc_avg([e for e in exps if not e["is_cross_task"] and e.get("level") == "token"], "token_test_auroc"),
                "n_experiments": len([e for e in exps if not e["is_cross_task"] and e.get("level") == "token"]),
            }
            for method, exps in by_method.items()
        },
        "experiments": experiments,
    }

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to {results_dir / 'summary.json'}")

    save_comparison_csv(experiments, results_dir / "comparison_sample.csv", "sample")
    save_comparison_csv(experiments, results_dir / "comparison_token.csv", "token")
    logger.info("Comparison CSVs saved")

    print("\n" + "=" * 90)
    print("Results saved to outputs/results/")
    print("=" * 90)


if __name__ == "__main__":
    main()
