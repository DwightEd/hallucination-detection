#!/bin/bash
# =============================================================================
# 训练并评估幻觉检测模型
# =============================================================================
# 
# 支持三种评估模式：
# 1. 同任务评估（默认）：在对应任务的 _test 目录上评估
# 2. 跨任务评估：在所有任务的 _test 目录上评估
# 3. 指定评估任务：在指定任务的 _test 目录上评估
#
# 用法:
#   # 训练并同任务评估
#   bash train_and_evaluate.sh
#
#   # 仅评估（跳过训练）
#   bash train_and_evaluate.sh --eval-only
#
#   # 跨任务评估
#   bash train_and_evaluate.sh --cross-task
#
#   # 仅评估 + 跨任务
#   bash train_and_evaluate.sh --eval-only --cross-task
# =============================================================================

set -e  # 遇到错误立即退出

# =============================================================================
# 配置参数
# =============================================================================
DATASET="ragtruth"
MODEL="Mistral-7B-Instruct-v0.3"
SEED=42
METHODS="haloscope lookback_lens  tsv"

# 所有任务类型
ALL_TASKS="QA Summary Data2txt"

# 需要训练的任务（可按需修改）
TRAIN_TASKS="QA Summary Data2txt"

# 默认选项
EVAL_ONLY=false
CROSS_TASK=false

# =============================================================================
# 解析命令行参数
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        --cross-task)
            CROSS_TASK=true
            shift
            ;;
        --tasks)
            TRAIN_TASKS="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--eval-only] [--cross-task] [--tasks 'QA Summary'] [--methods 'lapeigvals hsdmvaf']"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Hallucination Detection: Train & Evaluate"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Seed: $SEED"
echo "Methods: $METHODS"
echo "Tasks to train: $TRAIN_TASKS"
echo "All tasks: $ALL_TASKS"
echo "Eval only: $EVAL_ONLY"
echo "Cross-task eval: $CROSS_TASK"
echo "============================================================"

# =============================================================================
# 步骤 1: 训练（如果不是 eval-only 模式）
# =============================================================================
if [ "$EVAL_ONLY" = false ]; then
    for TASK in $TRAIN_TASKS; do
        echo ""
        echo "============================================================"
        echo "[Train] Task: $TASK"
        echo "============================================================"
        
        # 训练每个方法
        for METHOD in $METHODS; do
            echo "  Training: $METHOD on $TASK"
            python scripts/train_probe.py \
                dataset.name=$DATASET \
                dataset.task_type=$TASK \
                seed=$SEED \
                method=$METHOD
        done
        
        echo "[Train] $TASK Done!"
    done
fi

# =============================================================================
# 步骤 2: 评估
# =============================================================================
echo ""
echo "============================================================"
echo "Evaluation"
echo "============================================================"

if [ "$CROSS_TASK" = true ]; then
    # -----------------------------------------------------------------
    # 跨任务评估：每个训练任务的模型在所有任务上评估
    # -----------------------------------------------------------------
    echo "Mode: Cross-task evaluation"
    echo ""
    
    for TRAIN_TASK in $TRAIN_TASKS; do
        echo ""
        echo "------------------------------------------------------------"
        echo "Evaluating models trained on: $TRAIN_TASK"
        echo "Testing on all tasks: $ALL_TASKS"
        echo "------------------------------------------------------------"
        
        python scripts/quick_eval.py \
            --methods $METHODS \
            --dataset $DATASET \
            --model $MODEL \
            --train_task $TRAIN_TASK \
            --cross_task \
            --all_tasks $ALL_TASKS \
            --seed $SEED \
            --output outputs/results/${TRAIN_TASK}_cross_task_eval.json
    done
    
else
    # -----------------------------------------------------------------
    # 同任务评估：每个任务的模型只在对应的 test 集上评估
    # -----------------------------------------------------------------
    echo "Mode: Same-task evaluation"
    echo ""
    
    for TASK in $TRAIN_TASKS; do
        echo ""
        echo "------------------------------------------------------------"
        echo "Evaluating: $TASK (train on $TASK, eval on ${TASK}_test)"
        echo "------------------------------------------------------------"
        
        python scripts/quick_eval.py \
            --methods $METHODS \
            --dataset $DATASET \
            --model $MODEL \
            --task_type $TASK \
            --seed $SEED \
            --output outputs/results/${TASK}_eval.json
    done
fi

# =============================================================================
# 步骤 3: 汇总
# =============================================================================
echo ""
echo "============================================================"
echo "All tasks completed!"
echo "============================================================"
echo ""
echo "Results saved to:"

if [ "$CROSS_TASK" = true ]; then
    for TASK in $TRAIN_TASKS; do
        echo "  - outputs/results/${TASK}_cross_task_eval.json"
    done
else
    for TASK in $TRAIN_TASKS; do
        echo "  - outputs/results/${TASK}_eval.json"
    done
fi

echo ""
echo "============================================================"