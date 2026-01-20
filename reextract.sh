#!/bin/bash
#==============================================================================
# 截断样本检测与重新提取
#
# 检测逻辑：
# 1. 从 answers.json 读取原始文本 → tokenizer 计算原始长度
# 2. 从 features_individual/*.pt 读取存储的长度和实际 seq_len
# 3. 检测条件：
#    - 原始长度 > 存储长度 → 被截断
#    - prompt_len >= seq_len → prompt 溢出
#    - actual_response_region <= 阈值 → response 区域过短
#
# 重新提取流程：
# 1. 用更大的 max_length 重新提取基础特征
# 2. 自动计算衍生特征 (laplacian_diags = 1 - attn_diags)
# 3. 更新合并特征文件 (features/*.pt)
# 4. 更新索引文件 (features/*_index.json)
# 5. 更新 metadata.json
#==============================================================================

set -e  # 出错时停止

# ============ 配置 ============
# 根目录（包含 QA, Summary, Data2txt 等子目录）
BASE_DIR="/share/home/tm902089733300000/a903202310/lys/research/hallucination-detection/outputs/features/ragtruth/Mistral-7B-Instruct-v0.3/seed_42"

# 模型路径（用于 tokenizer）
MODEL="mistralai/Mistral-7B-Instruct-v0.3"

# 新的 max_length（应该比原来更大）
NEW_MAX_LENGTH=16384

# response 区域最小阈值（小于等于此值视为问题样本）
MIN_RESPONSE_THRESHOLD=5

# 设备
DEVICE="cuda"

# 是否只检测不提取（设为 true 只输出问题样本列表）
DETECT_ONLY=false

# ============ 开始 ============
echo "============================================================"
echo "截断样本检测与重新提取"
echo "============================================================"
echo "配置:"
echo "  BASE_DIR: $BASE_DIR"
echo "  MODEL: $MODEL"
echo "  NEW_MAX_LENGTH: $NEW_MAX_LENGTH"
echo "  MIN_RESPONSE_THRESHOLD: $MIN_RESPONSE_THRESHOLD"
echo "  DETECT_ONLY: $DETECT_ONLY"
echo "============================================================"

# 检查目录是否存在
if [ ! -d "$BASE_DIR" ]; then
    echo "错误: 目录不存在: $BASE_DIR"
    exit 1
fi

# 构建参数
EXTRA_ARGS=""
if [ "$DETECT_ONLY" = true ]; then
    EXTRA_ARGS="--detect-only"
fi

# 统计
total_dirs=0
processed_dirs=0
failed_dirs=0

for split_dir in "$BASE_DIR"/*; do
    if [ -d "$split_dir" ]; then
        split_name=$(basename "$split_dir")
        total_dirs=$((total_dirs + 1))
        
        echo ""
        echo "=========================================="
        echo "处理: $split_name"
        echo "=========================================="
        
        # 检查必要文件
        if [ ! -f "$split_dir/answers.json" ]; then
            echo "⚠️ 跳过: 缺少 answers.json"
            continue
        fi
        
        if [ ! -d "$split_dir/features_individual" ]; then
            echo "⚠️ 跳过: 缺少 features_individual 目录"
            continue
        fi
        
        # 运行重新提取
        if python scripts/reextract.py \
            --features-dir "$split_dir" \
            --model "$MODEL" \
            --new-max-length $NEW_MAX_LENGTH \
            --min-response-threshold $MIN_RESPONSE_THRESHOLD \
            --device $DEVICE \
            $EXTRA_ARGS; then
            processed_dirs=$((processed_dirs + 1))
            echo "✅ $split_name 完成"
        else
            failed_dirs=$((failed_dirs + 1))
            echo "❌ $split_name 失败"
        fi
    fi
done

echo ""
echo "============================================================"
echo "完成"
echo "============================================================"
echo "总计: $total_dirs 个目录"
echo "成功: $processed_dirs"
echo "失败: $failed_dirs"
echo "============================================================"

# 验证步骤（可选）
if [ "$DETECT_ONLY" = false ] && [ $processed_dirs -gt 0 ]; then
    echo ""
    echo "验证提示："
    echo "1. 检查 truncated_samples.json 确认问题样本列表"
    echo "2. 检查 features/*.pt 确认特征已更新"
    echo "3. 运行训练脚本验证特征可用性："
    echo "   python scripts/train_probe.py method=lapeigvals dataset.task_type=QA"
    echo ""
fi