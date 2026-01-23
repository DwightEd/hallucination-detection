#!/bin/bash
# =============================================================================
# HaloScope 复现脚本 (修正版)
# =============================================================================
# 
# 使用方法:
#   bash run_haloscope.sh
#
# 此脚本使用修正后的HaloScope实现，忠实复现论文结果
# 按 task 循环执行完整流程：特征提取(train+test) -> 训练 -> 评估
# =============================================================================
set -e

echo "=============================================="
echo "HaloScope Reproduction"
echo "=============================================="
echo ""
echo "Key corrections applied:"
echo "  ✓ Token selection: last token (not mean pooling)"
echo "  ✓ Layer selection: middle layers 8-14 (not last_half)"
echo "  ✓ SVD score: projection² (not |projection|)"
echo "  ✓ Classifier: 2-layer MLP (not LogisticRegression)"
echo "  ✓ Random seed: 41 (paper specified)"
echo ""

# 配置
DATASET="ragtruth"
MODEL="mistral_7b"
MODEL_SHORT="Mistral-7B-Instruct-v0.3"
SEED=41  # 论文指定的种子值
TASKS=("QA" "Data2txt" "Summary")
SPLITS=("train" "test")

echo "Dataset: $DATASET"
echo "Model: $MODEL_SHORT"
echo "Seed: $SEED"
echo "Tasks: ${TASKS[*]}"
echo ""

# 按 task 循环执行完整流程
for TASK in "${TASKS[@]}"; do
    echo ""
    echo "##############################################"
    echo "# Processing Task: $TASK"
    echo "##############################################"
    echo ""

    # Step 1: 特征提取 (train 和 test)
    echo "=============================================="
    echo "Step 1: Feature Extraction for $TASK"
    echo "=============================================="
    
    for SPLIT in "${SPLITS[@]}"; do
        echo ""
        echo ">>> Extracting features for $SPLIT split..."
        python scripts/generate_activations.py \
            dataset.name=$DATASET \
            dataset.task_type=$TASK \
            dataset.split_name=$SPLIT \
            model=$MODEL \
            model.short_name=$MODEL_SHORT \
            features=haloscope \
            seed=$SEED
    done

    # Step 2: 训练
    echo ""
    echo "=============================================="
    echo "Step 2: Training HaloScope for $TASK"
    echo "=============================================="
    python scripts/train_probe.py \
        dataset.name=$DATASET \
        dataset.task_type=$TASK \
        model=$MODEL \
        model.short_name=$MODEL_SHORT \
        method=haloscope \
        seed=$SEED

    # Step 3: 评估
    echo ""
    echo "=============================================="
    echo "Step 3: Evaluation for $TASK"
    echo "=============================================="
    python scripts/evaluate.py \
        dataset.name=$DATASET \
        dataset.task_type=$TASK \
        model=$MODEL \
        model.short_name=$MODEL_SHORT \
        method=haloscope \
        seed=$SEED

    echo ""
    echo ">>> Task $TASK completed!"
    echo ""
done

echo ""
echo "=============================================="
echo "HaloScope Reproduction Complete!"
echo "=============================================="
echo ""
echo "All tasks processed: ${TASKS[*]}"
echo ""
echo "Expected performance on TruthfulQA:"
echo "  - AUROC: ~78.64% (paper reported)"
echo ""
echo "Results saved to: outputs/$DATASET/$MODEL_SHORT/haloscope/"
