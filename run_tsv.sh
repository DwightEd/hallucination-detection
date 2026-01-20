#!/bin/bash
# =============================================================================
# TSV (Truthfulness Separator Vector) 运行脚本
# =============================================================================
# 
# 基于 ICML 2025 论文: "Steer LLM Latents for Hallucination Detection"
# 
# 使用方法:
#   bash run_tsv.sh
#
# ⚠️ 重要说明:
#   1. TSV 使用 hidden_states (last-token embedding)
#   2. 可以复用 haloscope 提取的特征 (无需重新提取)
#   3. 对于 Mistral-7B，默认使用 8-15 层 (中间层)
#
# =============================================================================
set -e

echo "=============================================="
echo "TSV (Truthfulness Separator Vector) Method"
echo "=============================================="
echo ""
echo "Key features:"
echo "  ✓ Uses hidden_states (last-token embedding)"
echo "  ✓ Learns steering vector to reshape representation space"
echo "  ✓ vMF distribution for classification"
echo "  ✓ Two-stage training with optional pseudo-labeling"
echo ""

# 配置
DATASET="ragtruth"
MODEL="mistral_7b"
MODEL_SHORT="Mistral-7B-Instruct-v0.3"
SEED=42  # TSV 论文默认种子
TASKS=("QA" "Data2txt" "Summary")
SPLITS=("train" "test")

echo "Dataset: $DATASET"
echo "Model: $MODEL_SHORT"
echo "Seed: $SEED"
echo "Tasks: ${TASKS[*]}"
echo ""

# 检查是否有现有的 haloscope 特征
check_existing_features() {
    local TASK=$1
    local SPLIT=$2
    local FEATURE_DIR="outputs/${DATASET}/${MODEL_SHORT}/${TASK}/${SPLIT}/features_individual"
    
    if [ -d "$FEATURE_DIR" ]; then
        local COUNT=$(ls -1 "$FEATURE_DIR"/*.pt 2>/dev/null | wc -l)
        if [ "$COUNT" -gt 0 ]; then
            echo "  Found $COUNT existing feature files in $FEATURE_DIR"
            return 0
        fi
    fi
    return 1
}

# 按 task 循环执行完整流程
for TASK in "${TASKS[@]}"; do
    echo ""
    echo "##############################################"
    echo "# Processing Task: $TASK"
    echo "##############################################"
    echo ""

    # Step 1: 检查特征是否已存在 (可能由 haloscope 提取)
    echo "=============================================="
    echo "Step 1: Check/Extract Features for $TASK"
    echo "=============================================="
    
    NEED_EXTRACT=false
    
    for SPLIT in "${SPLITS[@]}"; do
        if ! check_existing_features "$TASK" "$SPLIT"; then
            echo "  Need to extract features for $SPLIT split"
            NEED_EXTRACT=true
        fi
    done
    
    if [ "$NEED_EXTRACT" = true ]; then
        echo ""
        echo ">>> Extracting features (using haloscope config for hidden_states)..."
        for SPLIT in "${SPLITS[@]}"; do
            echo ""
            echo ">>> Extracting $SPLIT split..."
            python scripts/generate_activations.py \
                dataset.name=$DATASET \
                dataset.task_type=$TASK \
                dataset.split_name=$SPLIT \
                model=$MODEL \
                model.short_name=$MODEL_SHORT \
                features=haloscope \
                seed=$SEED
        done
    else
        echo ""
        echo ">>> Features already exist, skipping extraction"
        echo ">>> (TSV can reuse haloscope features)"
    fi

    # Step 2: 训练 TSV
    echo ""
    echo "=============================================="
    echo "Step 2: Training TSV for $TASK"
    echo "=============================================="
    python scripts/train_probe.py \
        dataset.name=$DATASET \
        dataset.task_type=$TASK \
        model=$MODEL \
        model.short_name=$MODEL_SHORT \
        method=tsv \
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
        method=tsv \
        seed=$SEED

    echo ""
    echo ">>> Task $TASK completed!"
    echo ""
done

echo ""
echo "=============================================="
echo "TSV Training & Evaluation Complete!"
echo "=============================================="
echo ""
echo "All tasks processed: ${TASKS[*]}"
echo ""
echo "Results saved to: outputs/$DATASET/$MODEL_SHORT/tsv/"
echo ""
echo "TSV Parameters used:"
echo "  - steering_strength (λ): 5.0"
echo "  - kappa (κ): 10.0"
echo "  - layer_selection: middle (8-15 for 32-layer model)"
echo "  - epochs_stage1: 20"
echo "  - epochs_stage2: 20"
