#!/bin/bash
# =============================================================================
# Lookback Lens 重新训练 & 跨任务评估
# =============================================================================
# 模型: Mistral-7B-Instruct-v0.3
# 方法: lookback_lens (level=both → 训练 sample 和 token 两个级别)
# 评估: 跨任务评估 (每个任务的模型在所有任务上评估)
# =============================================================================
set -e

# =============================================================================
# 配置参数
# =============================================================================
DATASET="ragtruth"
MODEL="mistral_7b"
MODEL_SHORT="Mistral-7B-Instruct-v0.3"
SEED=42
METHOD="lookback_lens"
TASKS=("QA" "Data2txt" "Summary")

echo "=============================================="
echo "Lookback Lens 重新训练 & 跨任务评估"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Model: $MODEL_SHORT"
echo "Method: $METHOD"
echo "Seed: $SEED"
echo "Tasks: ${TASKS[*]}"
echo "=============================================="
echo ""

# =============================================================================
# Step 1: 删除旧模型
# =============================================================================
echo "=============================================="
echo "Step 1: 删除旧的 lookback_lens 模型"
echo "=============================================="

for TASK in "${TASKS[@]}"; do
    MODEL_DIR="outputs/models/$DATASET/$MODEL_SHORT/seed_$SEED/$TASK/$METHOD"
    if [ -d "$MODEL_DIR" ]; then
        echo ">>> 删除: $MODEL_DIR"
        rm -rf "$MODEL_DIR"
    fi
done
echo "✓ 旧模型已删除"
echo ""

# =============================================================================
# Step 2: 训练 (每个任务分别训练 sample 和 token 两个级别)
# =============================================================================
echo "=============================================="
echo "Step 2: 训练"
echo "=============================================="

for TASK in "${TASKS[@]}"; do
    echo ""
    echo ">>> 训练: $TASK (sample + token)"
    echo "--------------------------------------------"
    python scripts/train_probe.py \
        dataset.name=$DATASET \
        dataset.task_type=$TASK \
        model=$MODEL \
        model.short_name=$MODEL_SHORT \
        method=$METHOD \
        seed=$SEED
done

echo ""
echo "✓ 训练完成"
echo ""

# =============================================================================
# Step 3: 跨任务评估 (使用 quick_eval.py)
# =============================================================================
echo "=============================================="
echo "Step 3: 跨任务评估"
echo "=============================================="

# 创建结果目录
RESULTS_DIR="outputs/results/$DATASET/$MODEL_SHORT/$METHOD"
mkdir -p "$RESULTS_DIR"

# 对每个级别分别评估
for LEVEL in "sample" "token"; do
    echo ""
    echo "=============================================="
    echo "Level: $LEVEL"
    echo "=============================================="
    
    # 遍历每个训练任务
    for TRAIN_TASK in "${TASKS[@]}"; do
        echo ""
        echo ">>> 训练任务: $TRAIN_TASK → 评估所有任务"
        echo "--------------------------------------------"
        
        python scripts/quick_eval.py \
            --methods $METHOD \
            --dataset $DATASET \
            --model $MODEL_SHORT \
            --train_task $TRAIN_TASK \
            --cross_task \
            --all_tasks ${TASKS[*]} \
            --seed $SEED \
            --level $LEVEL \
            --output "$RESULTS_DIR/${LEVEL}_${TRAIN_TASK}_cross_task.json"
    done
done

echo ""
echo "✓ 跨任务评估完成"
echo ""

# =============================================================================
# Step 4: 汇总结果
# =============================================================================
echo "=============================================="
echo "Step 4: 汇总结果"
echo "=============================================="
echo ""

python << 'EOF'
import json
from pathlib import Path

dataset = "ragtruth"
model = "Mistral-7B-Instruct-v0.3"
method = "lookback_lens"
seed = 42
tasks = ["QA", "Data2txt", "Summary"]
levels = ["sample", "token"]
results_dir = Path(f"outputs/results/{dataset}/{model}/{method}")

print("=" * 90)
print(f"Lookback Lens 跨任务评估结果汇总")
print(f"Model: {model}")
print("=" * 90)

for level in levels:
    print(f"\n>>> Level: {level}")
    print("-" * 90)
    print(f"{'Train Task':<12} | {'Eval Task':<12} | {'AUROC':<10} | {'AUPR':<10} | {'F1':<10}")
    print("-" * 90)
    
    level_aurocs = []
    
    for train_task in tasks:
        result_file = results_dir / f"{level}_{train_task}_cross_task.json"
        
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            
            for result in data.get("results", []):
                if "error" not in result:
                    eval_task = result.get("eval_task", "N/A")
                    auroc = result.get("auroc", 0)
                    aupr = result.get("aupr", 0)
                    f1 = result.get("f1", 0)
                    
                    # 标记同任务评估
                    marker = " *" if train_task == eval_task else ""
                    print(f"{train_task:<12} | {eval_task:<12} | {auroc:<10.4f} | {aupr:<10.4f} | {f1:<10.4f}{marker}")
                    
                    level_aurocs.append(auroc)
        else:
            print(f"{train_task:<12} | {'N/A':<12} | {'(file not found)'}")
    
    print("-" * 90)
    if level_aurocs:
        avg_auroc = sum(level_aurocs) / len(level_aurocs)
        print(f"{'Average':<12} | {'All':<12} | {avg_auroc:<10.4f}")
    print()

print("=" * 90)
print("注: * 表示同任务评估 (train_task == eval_task)")
print("=" * 90)
EOF

echo ""
echo "=============================================="
echo "完成！"
echo "=============================================="
echo ""
echo "结果文件位置:"
echo "  $RESULTS_DIR/"
ls -la "$RESULTS_DIR/" 2>/dev/null || echo "  (目录为空或不存在)"
echo ""