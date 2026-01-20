#!/bin/bash
#==============================================================================
# 截断样本检测与重新提取
#
# 检测逻辑：
# 1. 从 answers.json 读取原始文本 → tokenizer 计算原始长度
# 2. 从 features_individual/*.pt 读取存储的长度（截断后的）
# 3. 对比：原始长度 > 存储长度 → 被截断
#==============================================================================

# 配置
BASE_DIR="/share/home/tm902089733300000/a903202310/lys/research/hallucination-detection/outputs/features/ragtruth/Mistral-7B-Instruct-v0.3/seed_42"
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
NEW_MAX_LENGTH=16384
MIN_RESPONSE_THRESHOLD=5
DEVICE="cuda"

echo "============================================================"
echo "截断样本检测与重新提取"
echo "============================================================"

for split_dir in "$BASE_DIR"/*; do
    if [ -d "$split_dir" ]; then
        split_name=$(basename "$split_dir")
        echo ""
        echo "处理: $split_name"
        echo "----------------------------------------"
        
        python scripts/reextract.py \
            --features-dir "$split_dir" \
            --model "$MODEL" \
            --new-max-length $NEW_MAX_LENGTH \
            --min-response-threshold $MIN_RESPONSE_THRESHOLD \
            --device $DEVICE
    fi
done

echo ""
echo "============================================================"
echo "完成"
echo "============================================================"