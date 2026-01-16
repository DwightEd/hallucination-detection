#!/usr/bin/env python3
"""
一键运行完整流水线

使用方法:
    python run_pipeline.py                      # 运行完整流水线
    python run_pipeline.py --stage train        # 只运行训练阶段
    python run_pipeline.py --stage evaluate     # 只运行评估阶段
    python run_pipeline.py --dry-run            # 只显示将要运行的命令
    python run_pipeline.py --use-dvc            # 使用DVC运行
"""
import os
import sys
import yaml
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """流水线运行器"""
    
    def __init__(self, params_file: str = "params.yaml", dry_run: bool = False):
        self.params_file = params_file
        self.dry_run = dry_run
        self.params = self._load_params()
        
    def _load_params(self) -> Dict[str, Any]:
        with open(self.params_file) as f:
            return yaml.safe_load(f)
    
    def _run_command(self, cmd: str, description: str = "") -> int:
        if description:
            logger.info(f"[{description}]")
        
        if self.dry_run:
            logger.info(f"[DRY-RUN] {cmd}")
            return 0
        
        logger.info(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True)
        
        if result.returncode != 0:
            logger.error(f"Command failed with return code {result.returncode}")
        
        return result.returncode
    
    def get_datasets(self) -> List[str]:
        return [d['name'] for d in self.params['datasets']]
    
    def get_task_types(self) -> List[str]:
        return self.params.get('task_types', ['all'])
    
    def get_models(self) -> List[Dict[str, str]]:
        return self.params['models']
    
    def get_methods(self) -> List[str]:
        return self.params['methods']
    
    def get_levels(self) -> List[str]:
        """获取级别列表（统一字段）。"""
        return self.params.get('levels', ['sample'])
    
    def get_eval_matrix(self) -> List[Dict[str, str]]:
        """获取评估矩阵。
        
        如果未定义，则自动生成同任务评估矩阵。
        """
        eval_matrix = self.params.get('eval_matrix', None)
        if eval_matrix:
            return eval_matrix
        
        # 默认：每个任务类型在自己的数据上评估
        task_types = self.get_task_types()
        if task_types and task_types != ['all']:
            return [{'train_task': t, 'eval_task': t} for t in task_types]
        
        # 如果没有指定任务类型，返回空列表（跳过此阶段）
        return []
    
    def get_seed(self) -> int:
        return self.params.get('seed', 42)
    
    def stage_split(self) -> int:
        logger.info("=" * 60)
        logger.info("Stage 1: Split Dataset")
        logger.info("=" * 60)
        return self._run_command("python scripts/split_dataset.py", "Splitting dataset")
    
    def stage_features(self) -> int:
        logger.info("=" * 60)
        logger.info("Stage 2: Generate Activations")
        logger.info("=" * 60)
        
        datasets = self.get_datasets()
        task_types = self.get_task_types()
        models = self.get_models()
        seed = self.get_seed()
        
        total = len(datasets) * len(task_types) * len(models)
        current = 0
        
        for dataset in datasets:
            for task_type in task_types:
                for model in models:
                    current += 1
                    model_name = model['name']
                    model_short = model.get('short_name', model_name)
                    
                    cmd = (
                        f"python scripts/generate_activations.py "
                        f"dataset.name={dataset} "
                        f"dataset.task_type={task_type} "
                        f"model={model_name} "
                        f"model.short_name={model_short} "
                        f"seed={seed}"
                    )
                    
                    ret = self._run_command(cmd, f"Features [{current}/{total}] {dataset}/{task_type}/{model_short}")
                    if ret != 0:
                        return ret
        
        return 0
    
    def stage_train(self) -> int:
        logger.info("=" * 60)
        logger.info("Stage 3: Train Probe")
        logger.info("=" * 60)
        
        datasets = self.get_datasets()
        task_types = self.get_task_types()
        models = self.get_models()
        methods = self.get_methods()
        levels = self.get_levels()
        seed = self.get_seed()
        
        total = len(datasets) * len(task_types) * len(models) * len(methods) * len(levels)
        current = 0
        
        for dataset in datasets:
            for task_type in task_types:
                for model in models:
                    for method in methods:
                        for level in levels:
                            current += 1
                            model_name = model['name']
                            model_short = model.get('short_name', model_name)
                            
                            cmd = (
                                f"python scripts/train_probe.py "
                                f"dataset.name={dataset} "
                                f"dataset.task_type={task_type} "
                                f"model={model_name} "
                                f"model.short_name={model_short} "
                                f"method={method} "
                                f"method.level={level} "
                                f"seed={seed}"
                            )
                            
                            ret = self._run_command(cmd, f"Train [{current}/{total}] {dataset}/{task_type}/{method}/{level}")
                            if ret != 0:
                                logger.warning("Training failed, continuing...")
        
        return 0
    
    def stage_evaluate(self) -> int:
        logger.info("=" * 60)
        logger.info("Stage 4: Evaluate")
        logger.info("=" * 60)
        
        datasets = self.get_datasets()
        models = self.get_models()
        methods = self.get_methods()
        levels = self.get_levels()
        eval_matrix = self.get_eval_matrix()
        seed = self.get_seed()
        
        total = len(datasets) * len(models) * len(methods) * len(levels) * len(eval_matrix)
        current = 0
        
        for dataset in datasets:
            for model in models:
                for method in methods:
                    for level in levels:
                        for pair in eval_matrix:
                            current += 1
                            train_task = pair['train_task']
                            eval_task = pair['eval_task']
                            
                            model_name = model['name']
                            model_short = model.get('short_name', model_name)
                            
                            cmd = (
                                f"python scripts/evaluate.py "
                                f"dataset.name={dataset} "
                                f"model={model_name} "
                                f"model.short_name={model_short} "
                                f"method={method} "
                                f"method.level={level} "
                                f"seed={seed} "
                                f"train_eval.train_task_types=[{train_task}] "
                                f"train_eval.eval_task_types=[{eval_task}]"
                            )
                            
                            ret = self._run_command(cmd, f"Eval [{current}/{total}] {train_task}->{eval_task}/{method}")
                            if ret != 0:
                                logger.warning("Evaluation failed, continuing...")
        
        return 0
    
    def stage_aggregate(self) -> int:
        logger.info("=" * 60)
        logger.info("Stage 5: Aggregate Results")
        logger.info("=" * 60)
        return self._run_command("python scripts/aggregate_results.py", "Aggregating results")
    
    def run_all(self) -> int:
        stages = [
            ("split", self.stage_split),
            ("features", self.stage_features),
            ("train", self.stage_train),
            ("evaluate", self.stage_evaluate),
            ("aggregate", self.stage_aggregate),
        ]
        
        start_time = datetime.now()
        
        for stage_name, stage_func in stages:
            logger.info(f"\n{'='*60}\nStarting stage: {stage_name}\n{'='*60}\n")
            ret = stage_func()
            if ret != 0:
                logger.error(f"Stage {stage_name} failed!")
                return ret
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\nPipeline completed! Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        
        return 0
    
    def run_stage(self, stage: str) -> int:
        stage_map = {
            "split": self.stage_split,
            "features": self.stage_features,
            "train": self.stage_train,
            "evaluate": self.stage_evaluate,
            "aggregate": self.stage_aggregate,
            "all": self.run_all,
        }
        
        if stage not in stage_map:
            logger.error(f"Unknown stage: {stage}")
            return 1
        
        return stage_map[stage]()


def run_with_dvc(stage: str = "all") -> int:
    stage_map = {
        "all": "dvc repro",
        "split": "dvc repro split_dataset",
        "features": "dvc repro generate_activations",
        "train": "dvc repro train_probe",
        "evaluate": "dvc repro evaluate",
        "aggregate": "dvc repro aggregate",
    }
    
    if stage not in stage_map:
        logger.error(f"Unknown stage: {stage}")
        return 1
    
    cmd = stage_map[stage]
    logger.info(f"Running with DVC: {cmd}")
    return subprocess.run(cmd, shell=True).returncode


def main():
    parser = argparse.ArgumentParser(description="Run hallucination detection pipeline")
    
    parser.add_argument(
        "--stage",
        choices=["all", "split", "features", "train", "evaluate", "aggregate"],
        default="all",
        help="Stage to run (default: all)"
    )
    
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--use-dvc", action="store_true", help="Use DVC for pipeline execution")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml file")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Hallucination Detection Pipeline")
    logger.info(f"Stage: {args.stage}, Dry-run: {args.dry_run}, Use DVC: {args.use_dvc}")
    logger.info("=" * 60)
    
    if args.use_dvc:
        return run_with_dvc(args.stage)
    else:
        runner = PipelineRunner(args.params, args.dry_run)
        return runner.run_stage(args.stage)


if __name__ == "__main__":
    sys.exit(main())
