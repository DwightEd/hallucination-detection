#!/bin/bash
# =============================================================================
# 训练剩余 task_type (Summary, Data2txt) 并评估所有类型
# =============================================================================
# 用法: bash run_remaining_tasks.sh
# =============================================================================

set -e  # 遇到错误立即退出

# 配置参数
DATASET="ragtruth"
MODEL="Mistral-7B-Instruct-v0.3"
SEED=42
METHODS="lapeigvals lookback_lens hsdmvaf"

# 剩余需要训练的 task_type（QA 已完成）
REMAINING_TASKS="Data2txt"

# 所有 task_type（用于最终评估）
ALL_TASKS="QA Summary Data2txt"

echo "============================================================"
echo "Training Remaining Tasks & Full Evaluation"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Seed: $SEED"
echo "Methods: $METHODS"
echo "Remaining tasks to train: $REMAINING_TASKS"
echo "All tasks for evaluation: $ALL_TASKS"
echo "============================================================"

# =============================================================================
# 步骤 1: 为剩余 task_type 提取特征并训练
# =============================================================================

for TASK in $REMAINING_TASKS; do
    echo ""
    echo "============================================================"
    echo "Processing task: $TASK"
    echo "============================================================"
    
    # -----------------------------------------------------------------
    # 1.1 提取训练集特征
    # -----------------------------------------------------------------
    # echo ""
    # echo "[${TASK}] Extracting TRAIN features..."
    # python scripts/generate_activations.py \
    #     dataset.name=$DATASET \
    #     dataset.split_name=train \
    #     dataset.task_type=$TASK \
    #     seed=$SEED \
    #     features.hidden_states_pooling=none \
    #     resume=false
    
    # # -----------------------------------------------------------------
    # # 1.2 提取测试集特征
    # # -----------------------------------------------------------------
    # echo ""
    # echo "[${TASK}] Extracting TEST features..."
    # python scripts/generate_activations.py \
    #     dataset.name=$DATASET \
    #     dataset.split_name=test \
    #     dataset.task_type=$TASK \
    #     seed=$SEED \
    #     features.hidden_states_pooling=none \
    #     resume=false
    
    # -----------------------------------------------------------------
    # 1.3 训练每个方法
    # -----------------------------------------------------------------
    echo ""
    echo "[${TASK}] Training probes..."
    for METHOD in $METHODS; do
        echo "  Training method: $METHOD"
        python scripts/train_probe.py \
            dataset.name=$DATASET \
            dataset.task_type=$TASK \
            seed=$SEED \
            method=$METHOD
    done
    
    echo ""
    echo "[${TASK}] Done!"
done

# =============================================================================
# 步骤 2: 评估所有 task_type
# =============================================================================

echo ""
echo "============================================================"
echo "Evaluating all task types..."
echo "============================================================"

for TASK in $ALL_TASKS; do
    echo ""
    echo "[Eval] Task: $TASK"
    echo "------------------------------------------------------------"
    
    python scripts/quick_eval.py \
        --methods $METHODS \
        --dataset $DATASET \
        --model $MODEL \
        --task_type $TASK \
        --seed $SEED \
        --output outputs/results/${TASK}_eval.json
done

# =============================================================================
# 步骤 3: 汇总结果
# =============================================================================

echo ""
echo "============================================================"
echo "All tasks completed!"
echo "============================================================"
echo ""
echo "Results saved to:"
for TASK in $ALL_TASKS; do
    echo "  - outputs/results/${TASK}_eval.json"
done
echo ""
echo "============================================================"