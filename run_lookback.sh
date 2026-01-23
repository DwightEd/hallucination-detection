#!/bin/bash
# =============================================================================
# LapEigvals (Laplacian Eigenvalue-based) 训练和评估脚本
# =============================================================================
# 
# 基于 EMNLP 2025 论文: 
# "Hallucination Detection in LLMs Using Spectral Features of Attention Maps"
# GitHub: https://github.com/graphml-lab-pwr/lapeigvals
# 
# 使用方法:
#   bash run_lapeigvals.sh                    # 使用默认配置
#   bash run_lapeigvals.sh --task QA          # 仅运行 QA 任务
#   bash run_lapeigvals.sh --skip-train       # 跳过训练，仅评估
#   bash run_lapeigvals.sh --eval-only        # 同上
#
# 特征需求:
#   - full_attention 或 laplacian_diags (预计算的 Laplacian 对角线)
#   - 推荐使用 laplacian_diags 以节省内存
#
# =============================================================================

set -e  # 遇到错误立即退出

# =============================================================================
# 配置参数 (可根据需要修改)
# =============================================================================

# 数据集配置
DATASET="${DATASET:-ragtruth}"
TASKS="${TASKS:-QA Data2txt Summary}"
#QA
# 模型配置
MODEL="${MODEL:-mistral_7b}"
MODEL_SHORT="${MODEL_SHORT:-Mistral-7B-Instruct-v0.3}"

# 训练配置
SEED="${SEED:-42}"
LEVEL="${LEVEL:-sample}"  # sample / token / both

# 特征配置
FEATURES_CONFIG="${FEATURES_CONFIG:-lapeigvals}"  # lapeigvals / diagonal / full

# 控制参数
SKIP_TRAIN="${SKIP_TRAIN:-false}"
SKIP_EVAL="${SKIP_EVAL:-false}"
SKIP_FEATURE_EXTRACT="${SKIP_FEATURE_EXTRACT:-false}"
VERBOSE="${VERBOSE:-false}"

# =============================================================================
# 解析命令行参数
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASKS="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --model-short)
            MODEL_SHORT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --level)
            LEVEL="$2"
            shift 2
            ;;
        --skip-train|--eval-only)
            SKIP_TRAIN="true"
            shift
            ;;
        --skip-eval|--train-only)
            SKIP_EVAL="true"
            shift
            ;;
        --skip-extract)
            SKIP_FEATURE_EXTRACT="true"
            shift
            ;;
        --verbose|-v)
            VERBOSE="true"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --task TASK           Task type (QA, Data2txt, Summary, or space-separated list)"
            echo "  --dataset DATASET     Dataset name (default: ragtruth)"
            echo "  --model MODEL         Model config name (default: mistral_7b)"
            echo "  --model-short NAME    Model short name for output paths"
            echo "  --seed SEED           Random seed (default: 42)"
            echo "  --level LEVEL         Training level: sample, token, both (default: sample)"
            echo "  --skip-train          Skip training, only evaluate"
            echo "  --skip-eval           Skip evaluation, only train"
            echo "  --skip-extract        Skip feature extraction"
            echo "  --verbose, -v         Verbose output"
            echo "  --help, -h            Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# 打印配置信息
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║     LapEigvals - Laplacian Eigenvalue Hallucination Detection        ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║ Dataset:      $DATASET"
echo "║ Model:        $MODEL_SHORT"
echo "║ Tasks:        $TASKS"
echo "║ Seed:         $SEED"
echo "║ Level:        $LEVEL"
echo "║ Skip Train:   $SKIP_TRAIN"
echo "║ Skip Eval:    $SKIP_EVAL"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# 确认 Python 环境
echo ">>> Python environment check..."
python --version
echo ""

# =============================================================================
# 辅助函数
# =============================================================================

check_features_exist() {
    local TASK=$1
    local SPLIT=$2
    local FEATURE_DIR="outputs/features/${DATASET}/${MODEL_SHORT}/seed_${SEED}/${TASK}"
    
    if [ "$SPLIT" == "test" ]; then
        FEATURE_DIR="${FEATURE_DIR}_test"
    fi
    
    if [ -d "$FEATURE_DIR" ]; then
        # 检查是否有特征文件
        local INDIVIDUAL_DIR="${FEATURE_DIR}/features_individual"
        if [ -d "$INDIVIDUAL_DIR" ]; then
            local COUNT=$(ls -1 "$INDIVIDUAL_DIR"/*.pt 2>/dev/null | wc -l)
            if [ "$COUNT" -gt 0 ]; then
                echo "  ✓ Found $COUNT feature files for ${TASK}/${SPLIT}"
                return 0
            fi
        fi
        
        # 检查合并特征
        local FEATURES_DIR="${FEATURE_DIR}/features"
        if [ -d "$FEATURES_DIR" ]; then
            local COUNT=$(ls -1 "$FEATURES_DIR"/*.pt 2>/dev/null | wc -l)
            if [ "$COUNT" -gt 0 ]; then
                echo "  ✓ Found $COUNT consolidated feature files for ${TASK}/${SPLIT}"
                return 0
            fi
        fi
    fi
    
    echo "  ✗ No features found for ${TASK}/${SPLIT}"
    return 1
}

check_model_exists() {
    local TASK=$1
    local LVL=$2
    local MODEL_PATH="outputs/models/${DATASET}/${MODEL_SHORT}/seed_${SEED}/${TASK}/lapeigvals/${LVL}/model.pkl"
    
    if [ -f "$MODEL_PATH" ]; then
        echo "  ✓ Model exists: $MODEL_PATH"
        return 0
    fi
    return 1
}

run_with_log() {
    local CMD="$1"
    local DESC="$2"
    
    echo ""
    echo ">>> $DESC"
    echo ">>> Command: $CMD"
    echo ""
    
    if [ "$VERBOSE" == "true" ]; then
        eval "$CMD"
    else
        eval "$CMD" 2>&1 | tee -a "lapeigvals_${DATASET}_${MODEL_SHORT}.log"
    fi
    
    local STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "  ✗ Failed with status $STATUS"
        return $STATUS
    fi
    echo "  ✓ Completed successfully"
    return 0
}

# =============================================================================
# 主流程
# =============================================================================

# 转换 TASKS 为数组
read -ra TASK_ARRAY <<< "$TASKS"

# 转换 LEVEL 为数组
if [ "$LEVEL" == "both" ]; then
    LEVEL_ARRAY=("sample" "token")
else
    LEVEL_ARRAY=("$LEVEL")
fi

# 创建日志目录
mkdir -p logs

TOTAL_TASKS=${#TASK_ARRAY[@]}
CURRENT_TASK=0
FAILED_TASKS=""

for TASK in "${TASK_ARRAY[@]}"; do
    CURRENT_TASK=$((CURRENT_TASK + 1))
    
    echo ""
    echo "############################################################"
    echo "# [$CURRENT_TASK/$TOTAL_TASKS] Processing Task: $TASK"
    echo "############################################################"
    
    # =========================================================================
    # Step 1: 检查/提取特征
    # =========================================================================
    
    if [ "$SKIP_FEATURE_EXTRACT" != "true" ]; then
        echo ""
        echo "=============================================="
        echo "Step 1: Check/Extract Features"
        echo "=============================================="
        
        NEED_EXTRACT_TRAIN=false
        NEED_EXTRACT_TEST=false
        
        if ! check_features_exist "$TASK" "train"; then
            NEED_EXTRACT_TRAIN=true
        fi
        
        if ! check_features_exist "$TASK" "test"; then
            NEED_EXTRACT_TEST=true
        fi
        
        # 提取训练集特征
        if [ "$NEED_EXTRACT_TRAIN" == "true" ]; then
            echo ""
            echo ">>> Extracting TRAIN features for $TASK..."
            
            CMD="python scripts/generate_activations.py \
                dataset.name=$DATASET \
                dataset.task_type=$TASK \
                dataset.split_name=train \
                model=$MODEL \
                model.short_name=$MODEL_SHORT \
                features=$FEATURES_CONFIG \
                seed=$SEED"
            
            run_with_log "$CMD" "Extract train features"
            if [ $? -ne 0 ]; then
                FAILED_TASKS="$FAILED_TASKS ${TASK}:train_extract"
                continue
            fi
        else
            echo ">>> Train features already exist, skipping extraction"
        fi
        
        # 提取测试集特征
        if [ "$NEED_EXTRACT_TEST" == "true" ]; then
            echo ""
            echo ">>> Extracting TEST features for $TASK..."
            
            CMD="python scripts/generate_activations.py \
                dataset.name=$DATASET \
                dataset.task_type=$TASK \
                dataset.split_name=test \
                model=$MODEL \
                model.short_name=$MODEL_SHORT \
                features=$FEATURES_CONFIG \
                seed=$SEED"
            
            run_with_log "$CMD" "Extract test features"
            if [ $? -ne 0 ]; then
                FAILED_TASKS="$FAILED_TASKS ${TASK}:test_extract"
                continue
            fi
        else
            echo ">>> Test features already exist, skipping extraction"
        fi
    fi
    
    # =========================================================================
    # Step 2: 训练
    # =========================================================================
    
    if [ "$SKIP_TRAIN" != "true" ]; then
        echo ""
        echo "=============================================="
        echo "Step 2: Training LapEigvals"
        echo "=============================================="
        
        for LVL in "${LEVEL_ARRAY[@]}"; do
            echo ""
            echo ">>> Training at level: $LVL"
            
            # 检查模型是否已存在
            if check_model_exists "$TASK" "$LVL"; then
                echo ">>> Model already exists, skipping training (use --force to retrain)"
            else
                CMD="python scripts/train_probe.py \
                    dataset.name=$DATASET \
                    dataset.task_type=$TASK \
                    model=$MODEL \
                    model.short_name=$MODEL_SHORT \
                    method=lapeigvals \
                    method.level=$LVL \
                    seed=$SEED"
                
                run_with_log "$CMD" "Train lapeigvals ($LVL level)"
                if [ $? -ne 0 ]; then
                    FAILED_TASKS="$FAILED_TASKS ${TASK}:train_${LVL}"
                    continue
                fi
            fi
        done
    fi
    
    # =========================================================================
    # Step 3: 评估
    # =========================================================================
    
    if [ "$SKIP_EVAL" != "true" ]; then
        echo ""
        echo "=============================================="
        echo "Step 3: Evaluation"
        echo "=============================================="
        
        for LVL in "${LEVEL_ARRAY[@]}"; do
            echo ""
            echo ">>> Evaluating at level: $LVL"
            
            # 检查模型是否存在
            MODEL_PATH="outputs/models/${DATASET}/${MODEL_SHORT}/seed_${SEED}/${TASK}/lapeigvals/${LVL}/model.pkl"
            if [ ! -f "$MODEL_PATH" ]; then
                echo "  ✗ Model not found: $MODEL_PATH"
                echo "  ✗ Please run training first"
                FAILED_TASKS="$FAILED_TASKS ${TASK}:eval_${LVL}"
                continue
            fi
            
            CMD="python scripts/evaluate.py \
                dataset.name=$DATASET \
                dataset.task_type=$TASK \
                model=$MODEL \
                model.short_name=$MODEL_SHORT \
                method=lapeigvals \
                method.level=$LVL \
                seed=$SEED"
            
            run_with_log "$CMD" "Evaluate lapeigvals ($LVL level)"
            if [ $? -ne 0 ]; then
                FAILED_TASKS="$FAILED_TASKS ${TASK}:eval_${LVL}"
            fi
        done
    fi
    
    echo ""
    echo ">>> Task $TASK completed!"
done

# =============================================================================
# 总结
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    LapEigvals Pipeline Complete!                      ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║ Processed tasks: ${TASK_ARRAY[*]}"
echo "║"

if [ -n "$FAILED_TASKS" ]; then
    echo "║ ⚠️  Some tasks failed:$FAILED_TASKS"
else
    echo "║ ✅ All tasks completed successfully!"
fi

echo "║"
echo "║ Results locations:"
echo "║   Models:  outputs/models/${DATASET}/${MODEL_SHORT}/seed_${SEED}/<task>/lapeigvals/"
echo "║   Metrics: outputs/models/${DATASET}/${MODEL_SHORT}/seed_${SEED}/<task>/lapeigvals/<level>/eval_results.json"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
# 打印结果摘要
# =============================================================================

echo ""
echo ">>> Results Summary:"
echo ""

for TASK in "${TASK_ARRAY[@]}"; do
    for LVL in "${LEVEL_ARRAY[@]}"; do
        RESULTS_FILE="outputs/models/${DATASET}/${MODEL_SHORT}/seed_${SEED}/${TASK}/lapeigvals/${LVL}/eval_results.json"
        if [ -f "$RESULTS_FILE" ]; then
            echo "[$TASK/$LVL]:"
            # 尝试提取关键指标
            if command -v jq &> /dev/null; then
                AUROC=$(jq -r '.auroc // "N/A"' "$RESULTS_FILE" 2>/dev/null)
                AUPR=$(jq -r '.aupr // .average_precision // "N/A"' "$RESULTS_FILE" 2>/dev/null)
                echo "  AUROC: $AUROC"
                echo "  AUPR:  $AUPR"
            else
                # 简单的 grep 方式
                echo "  $(grep -o '"auroc":[^,}]*' "$RESULTS_FILE" 2>/dev/null || echo 'AUROC: see file')"
                echo "  $(grep -o '"aupr":[^,}]*' "$RESULTS_FILE" 2>/dev/null || echo 'AUPR: see file')"
            fi
            echo ""
        else
            echo "[$TASK/$LVL]: No results file found"
            echo ""
        fi
    done
done

# 返回状态码
if [ -n "$FAILED_TASKS" ]; then
    exit 1
fi
exit 0