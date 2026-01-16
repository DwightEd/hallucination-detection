#!/bin/bash
# =============================================================================
# 完整训练 + 评估流水线（绕过 DVC）
# =============================================================================
# 用法: bash run_full_pipeline.sh [--skip-train] [--skip-eval]
# 
# 选项:
#   --skip-train    跳过训练阶段，只进行评估
#   --skip-eval     跳过评估阶段，只进行训练
#   --methods       指定方法列表（逗号分隔）
#   --tasks         指定任务列表（逗号分隔）
# =============================================================================

set -e  # 遇到错误立即退出

# =============================================================================
# 配置（从 params.yaml 提取）
# =============================================================================
DATASET="ragtruth"
MODEL_NAME="mistral_7b"
MODEL_SHORT="Mistral-7B-Instruct-v0.3"
SEED=42

# 所有方法
ALL_METHODS="lapeigvals lookback_lens haloscope hsdmvaf hypergraph semantic_entropy_probes"

# 所有任务类型
ALL_TASKS="QA Summary Data2txt"

# 默认值
METHODS="$ALL_METHODS"
TASKS="$ALL_TASKS"
SKIP_TRAIN=false
SKIP_EVAL=false

# =============================================================================
# 解析命令行参数
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --methods)
            METHODS=$(echo "$2" | tr ',' ' ')
            shift 2
            ;;
        --tasks)
            TASKS=$(echo "$2" | tr ',' ' ')
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# 显示配置
# =============================================================================
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║            Hallucination Detection - Full Pipeline                   ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║ Dataset:    $DATASET"
echo "║ Model:      $MODEL_SHORT"
echo "║ Seed:       $SEED"
echo "║ Methods:    $METHODS"
echo "║ Tasks:      $TASKS"
echo "║ Skip Train: $SKIP_TRAIN"
echo "║ Skip Eval:  $SKIP_EVAL"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# 创建结果目录
RESULTS_DIR="outputs/results/${DATASET}/${MODEL_SHORT}/seed_${SEED}"
mkdir -p "$RESULTS_DIR"

# 计时开始
START_TIME=$(date +%s)

# =============================================================================
# 阶段 1: 训练
# =============================================================================
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STAGE 1: TRAINING"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 统计
    TOTAL_JOBS=0
    for TASK in $TASKS; do
        for METHOD in $METHODS; do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
        done
    done
    
    CURRENT_JOB=0
    FAILED_JOBS=""
    
    for TASK in $TASKS; do
        echo ""
        echo "┌──────────────────────────────────────────────────────────────────────┐"
        echo "│ Task: $TASK"
        echo "└──────────────────────────────────────────────────────────────────────┘"
        
        for METHOD in $METHODS; do
            CURRENT_JOB=$((CURRENT_JOB + 1))
            echo ""
            echo "  [$CURRENT_JOB/$TOTAL_JOBS] Training: $METHOD on $TASK"
            echo "  ────────────────────────────────────────────────────"
            
            # 检查模型是否已存在
            MODEL_PATH="outputs/models/${DATASET}/${MODEL_SHORT}/seed_${SEED}/${TASK}/${METHOD}/sample/model.pkl"
            if [ -f "$MODEL_PATH" ]; then
                echo "  ⏭️  Model already exists, skipping..."
                continue
            fi
            
            # 训练
            if python scripts/train_probe.py \
                dataset.name=$DATASET \
                dataset.task_type=$TASK \
                model=$MODEL_NAME \
                model.short_name=$MODEL_SHORT \
                method=$METHOD \
                seed=$SEED 2>&1; then
                echo "  ✅ Success"
            else
                echo "  ❌ Failed"
                FAILED_JOBS="$FAILED_JOBS $METHOD@$TASK"
            fi
        done
    done
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  TRAINING COMPLETE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ -n "$FAILED_JOBS" ]; then
        echo "  ⚠️  Failed jobs:$FAILED_JOBS"
    else
        echo "  ✅ All training jobs completed successfully"
    fi
fi

# =============================================================================
# 阶段 2: 评估
# =============================================================================
if [ "$SKIP_EVAL" = false ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STAGE 2: EVALUATION"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    for TASK in $TASKS; do
        echo ""
        echo "┌──────────────────────────────────────────────────────────────────────┐"
        echo "│ Evaluating Task: $TASK"
        echo "└──────────────────────────────────────────────────────────────────────┘"
        
        OUTPUT_FILE="${RESULTS_DIR}/${TASK}_results.json"
        
        python scripts/quick_eval.py \
            --methods $METHODS \
            --dataset $DATASET \
            --model $MODEL_SHORT \
            --task_type $TASK \
            --seed $SEED \
            --output "$OUTPUT_FILE" 2>&1 || echo "  ⚠️  Evaluation had some issues"
        
        echo "  📄 Results saved to: $OUTPUT_FILE"
    done
fi

# =============================================================================
# 阶段 3: 汇总结果
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STAGE 3: RESULTS SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 使用 Python 生成漂亮的结果表格
python3 << 'PYTHON_SCRIPT'
import json
import os
from pathlib import Path

# 配置
dataset = "ragtruth"
model = "Mistral-7B-Instruct-v0.3"
seed = 42
results_dir = Path(f"outputs/results/{dataset}/{model}/seed_{seed}")
tasks = ["QA", "Summary", "Data2txt"]
methods = ["lapeigvals", "lookback_lens", "haloscope", "hsdmvaf", "hypergraph", "semantic_entropy_probes"]

# 收集结果
results = {}
for task in tasks:
    result_file = results_dir / f"{task}_results.json"
    if result_file.exists():
        try:
            with open(result_file) as f:
                results[task] = json.load(f)
        except:
            results[task] = {}
    else:
        results[task] = {}

# 打印表格
print("\n" + "═" * 90)
print(" AUROC Results (Sample Level)")
print("═" * 90)

# 表头
header = f"{'Method':<25}"
for task in tasks:
    header += f" {task:>12}"
header += f" {'Average':>12}"
print(header)
print("─" * 90)

# 数据行
method_avgs = {}
for method in methods:
    row = f"{method:<25}"
    scores = []
    for task in tasks:
        task_results = results.get(task, {})
        method_results = task_results.get(method, {})
        auroc = method_results.get("auroc", method_results.get("sample_auroc", None))
        if auroc is not None:
            row += f" {auroc*100:>11.2f}%"
            scores.append(auroc)
        else:
            row += f" {'N/A':>12}"
    
    # 计算平均
    if scores:
        avg = sum(scores) / len(scores)
        row += f" {avg*100:>11.2f}%"
        method_avgs[method] = avg
    else:
        row += f" {'N/A':>12}"
    
    print(row)

print("─" * 90)

# 找出最佳方法
if method_avgs:
    best_method = max(method_avgs, key=method_avgs.get)
    print(f"\n🏆 Best Method: {best_method} (Avg AUROC: {method_avgs[best_method]*100:.2f}%)")

# 按任务显示最佳方法
print("\n📊 Best Method per Task:")
for task in tasks:
    task_results = results.get(task, {})
    best_score = 0
    best = "N/A"
    for method in methods:
        method_results = task_results.get(method, {})
        auroc = method_results.get("auroc", method_results.get("sample_auroc", 0))
        if auroc and auroc > best_score:
            best_score = auroc
            best = method
    print(f"  {task}: {best} ({best_score*100:.2f}%)")

# 保存汇总
summary = {
    "dataset": dataset,
    "model": model,
    "seed": seed,
    "results": results,
    "method_averages": method_avgs,
}
summary_file = results_dir / "summary.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n📄 Summary saved to: {summary_file}")

print("═" * 90)
PYTHON_SCRIPT

# =============================================================================
# 完成
# =============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                        PIPELINE COMPLETE                             ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║ Total Time: ${MINUTES}m ${SECONDS}s"
echo "║ Results:    $RESULTS_DIR"
echo "╚══════════════════════════════════════════════════════════════════════╝"