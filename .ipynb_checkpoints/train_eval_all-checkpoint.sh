#!/bin/bash
# =============================================================================
# 完整训练与评估脚本 - 每个方法训练后立即评估并汇报
# =============================================================================
# 
# 功能:
#   1. 逐个方法训练 + 立即评估
#   2. 实时显示每个方法的 AUROC/AUPR/F1
#   3. 最终汇总所有结果
#   4. 支持多模型、多任务
#
# 用法:
#   bash train_eval_all.sh                    # 默认配置
#   bash train_eval_all.sh --model Mistral    # 指定模型
#   bash train_eval_all.sh --tasks "QA"       # 只跑 QA 任务
#   bash train_eval_all.sh --methods "lookback_lens haloscope"  # 指定方法
#   bash train_eval_all.sh --skip-train       # 跳过训练，只评估
# =============================================================================

set -e  # 遇到错误时继续执行（改为不退出以完成所有方法）
set +e  # 允许单个命令失败

# =============================================================================
# 颜色定义
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# =============================================================================
# 默认配置
# =============================================================================
DATASET="ragtruth"
MODEL_SHORT="Llama-2-7b-chat-hf"
MODEL_NAME="llama2_7b_chat"
SEED=42

# 方法列表
DEFAULT_METHODS="lapeigvals lookback_lens haloscope hsdmvaf semantic_entropy_probes token_entropy"

# 任务列表
DEFAULT_TASKS="QA Summary Data2txt"

# 级别 (sample / token / both)
DEFAULT_LEVEL="sample"

# 选项
SKIP_TRAIN=false
VERBOSE=false

# =============================================================================
# 解析命令行参数
# =============================================================================
METHODS="$DEFAULT_METHODS"
TASKS="$DEFAULT_TASKS"
LEVEL="$DEFAULT_LEVEL"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_SHORT="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --level)
            LEVEL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL_SHORT    模型简称 (default: Llama-2-7b-chat-hf)"
            echo "  --model-name NAME      模型配置名 (default: llama2_7b_chat)"
            echo "  --methods 'M1 M2'      方法列表 (空格分隔)"
            echo "  --tasks 'T1 T2'        任务列表 (空格分隔)"
            echo "  --level LEVEL          训练级别: sample/token/both"
            echo "  --seed SEED            随机种子"
            echo "  --skip-train           跳过训练，只评估"
            echo "  --verbose, -v          详细输出"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# 创建日志和结果目录
# =============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/${TIMESTAMP}"
RESULTS_DIR="outputs/results/${DATASET}/${MODEL_SHORT}/seed_${SEED}"
SUMMARY_FILE="${RESULTS_DIR}/summary_${TIMESTAMP}.txt"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# =============================================================================
# 显示配置
# =============================================================================
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}${BOLD}        幻觉检测框架 - 完整训练与评估流水线${NC}                       ${CYAN}║${NC}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║${NC} 数据集:   ${YELLOW}$DATASET${NC}"
echo -e "${CYAN}║${NC} 模型:     ${YELLOW}$MODEL_SHORT${NC}"
echo -e "${CYAN}║${NC} 随机种子: ${YELLOW}$SEED${NC}"
echo -e "${CYAN}║${NC} 级别:     ${YELLOW}$LEVEL${NC}"
echo -e "${CYAN}║${NC} 方法:     ${YELLOW}$METHODS${NC}"
echo -e "${CYAN}║${NC} 任务:     ${YELLOW}$TASKS${NC}"
echo -e "${CYAN}║${NC} 跳过训练: ${YELLOW}$SKIP_TRAIN${NC}"
echo -e "${CYAN}║${NC} 日志目录: ${YELLOW}$LOG_DIR${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# =============================================================================
# 结果收集数组
# =============================================================================
declare -A RESULTS_AUROC
declare -A RESULTS_AUPR
declare -A RESULTS_F1
declare -A TRAIN_STATUS
declare -A EVAL_STATUS

# =============================================================================
# 辅助函数
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 提取评估指标
extract_metrics() {
    local log_file=$1
    local auroc=""
    local aupr=""
    local f1=""
    
    # 从日志中提取指标
    if [[ -f "$log_file" ]]; then
        auroc=$(grep -i "auroc" "$log_file" | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
        aupr=$(grep -i "aupr" "$log_file" | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
        f1=$(grep -i "f1" "$log_file" | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
    fi
    
    echo "$auroc,$aupr,$f1"
}

# 打印方法结果
print_method_result() {
    local method=$1
    local task=$2
    local auroc=$3
    local aupr=$4
    local f1=$5
    local status=$6
    
    if [[ "$status" == "success" ]]; then
        printf "  ${GREEN}✓${NC} %-20s %-10s AUROC=%-7s AUPR=%-7s F1=%-7s\n" \
            "$method" "$task" "${auroc:-N/A}" "${aupr:-N/A}" "${f1:-N/A}"
    else
        printf "  ${RED}✗${NC} %-20s %-10s ${RED}%s${NC}\n" \
            "$method" "$task" "$status"
    fi
}

# =============================================================================
# 训练单个方法
# =============================================================================
train_method() {
    local method=$1
    local task=$2
    local level=$3
    local log_file="${LOG_DIR}/train_${method}_${task}_${level}.log"
    
    log_info "训练: ${method}/${task}/${level}"
    
    if python scripts/train_probe.py \
        dataset.name=${DATASET} \
        dataset.task_type=${task} \
        model=${MODEL_NAME} \
        model.short_name=${MODEL_SHORT} \
        method=${method} \
        method.level=${level} \
        seed=${SEED} \
        > "$log_file" 2>&1; then
        
        TRAIN_STATUS["${method}_${task}_${level}"]="success"
        log_success "训练完成: ${method}/${task}/${level}"
        return 0
    else
        TRAIN_STATUS["${method}_${task}_${level}"]="failed"
        log_error "训练失败: ${method}/${task}/${level}"
        if [[ "$VERBOSE" == "true" ]]; then
            tail -20 "$log_file"
        fi
        return 1
    fi
}

# =============================================================================
# 评估单个方法
# =============================================================================
evaluate_method() {
    local method=$1
    local task=$2
    local level=$3
    local log_file="${LOG_DIR}/eval_${method}_${task}_${level}.log"
    
    log_info "评估: ${method}/${task}/${level}"
    
    if python scripts/evaluate.py \
        dataset.name=${DATASET} \
        dataset.task_type=${task} \
        model=${MODEL_NAME} \
        model.short_name=${MODEL_SHORT} \
        method=${method} \
        method.level=${level} \
        seed=${SEED} \
        > "$log_file" 2>&1; then
        
        # 提取指标
        local metrics=$(extract_metrics "$log_file")
        IFS=',' read -r auroc aupr f1 <<< "$metrics"
        
        RESULTS_AUROC["${method}_${task}_${level}"]="$auroc"
        RESULTS_AUPR["${method}_${task}_${level}"]="$aupr"
        RESULTS_F1["${method}_${task}_${level}"]="$f1"
        EVAL_STATUS["${method}_${task}_${level}"]="success"
        
        # 也尝试从 JSON 结果文件读取
        local results_json="${RESULTS_DIR}/../../../models/${DATASET}/${MODEL_SHORT}/seed_${SEED}/${task}/${method}/${level}/eval_results.json"
        if [[ -f "$results_json" ]]; then
            local json_auroc=$(python3 -c "import json; d=json.load(open('$results_json')); print(d.get('metrics',d).get('auroc',0))" 2>/dev/null)
            local json_aupr=$(python3 -c "import json; d=json.load(open('$results_json')); print(d.get('metrics',d).get('aupr',0))" 2>/dev/null)
            local json_f1=$(python3 -c "import json; d=json.load(open('$results_json')); print(d.get('metrics',d).get('f1',0))" 2>/dev/null)
            
            [[ -n "$json_auroc" && "$json_auroc" != "0" ]] && RESULTS_AUROC["${method}_${task}_${level}"]="$json_auroc"
            [[ -n "$json_aupr" && "$json_aupr" != "0" ]] && RESULTS_AUPR["${method}_${task}_${level}"]="$json_aupr"
            [[ -n "$json_f1" && "$json_f1" != "0" ]] && RESULTS_F1["${method}_${task}_${level}"]="$json_f1"
        fi
        
        print_method_result "$method" "$task" \
            "${RESULTS_AUROC["${method}_${task}_${level}"]}" \
            "${RESULTS_AUPR["${method}_${task}_${level}"]}" \
            "${RESULTS_F1["${method}_${task}_${level}"]}" \
            "success"
        
        return 0
    else
        EVAL_STATUS["${method}_${task}_${level}"]="failed"
        print_method_result "$method" "$task" "" "" "" "评估失败"
        if [[ "$VERBOSE" == "true" ]]; then
            tail -20 "$log_file"
        fi
        return 1
    fi
}

# =============================================================================
# 主流程
# =============================================================================

# 计算总数
total_jobs=0
for task in $TASKS; do
    for method in $METHODS; do
        total_jobs=$((total_jobs + 1))
    done
done

current_job=0
start_time=$(date +%s)

# 按任务循环
for task in $TASKS; do
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  任务: ${BOLD}$task${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    for method in $METHODS; do
        current_job=$((current_job + 1))
        
        echo -e "${BLUE}────────────────────────────────────────────────────────────────${NC}"
        echo -e "${BLUE}  [${current_job}/${total_jobs}] 方法: ${method}  任务: ${task}  级别: ${LEVEL}${NC}"
        echo -e "${BLUE}────────────────────────────────────────────────────────────────${NC}"
        
        # 训练
        if [[ "$SKIP_TRAIN" != "true" ]]; then
            if ! train_method "$method" "$task" "$LEVEL"; then
                log_warning "训练失败，跳过评估"
                continue
            fi
        fi
        
        # 评估
        evaluate_method "$method" "$task" "$LEVEL"
        
        echo ""
    done
done

# =============================================================================
# 汇总结果
# =============================================================================
end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}${BOLD}                        结果汇总${NC}                                   ${CYAN}║${NC}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════════╣${NC}"

# 创建汇总文件
{
    echo "=========================================="
    echo "幻觉检测评估结果汇总"
    echo "=========================================="
    echo "时间: $(date)"
    echo "数据集: $DATASET"
    echo "模型: $MODEL_SHORT"
    echo "级别: $LEVEL"
    echo ""
    echo "| 方法 | 任务 | AUROC | AUPR | F1 | 状态 |"
    echo "|------|------|-------|------|-----|------|"
} > "$SUMMARY_FILE"

# 打印表头
printf "${CYAN}║${NC} %-18s %-10s %-8s %-8s %-8s %-8s ${CYAN}║${NC}\n" \
    "方法" "任务" "AUROC" "AUPR" "F1" "状态"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════════╣${NC}"

# 统计
n_success=0
n_failed=0
best_auroc=0
best_method=""
best_task=""

for task in $TASKS; do
    for method in $METHODS; do
        key="${method}_${task}_${LEVEL}"
        auroc="${RESULTS_AUROC[$key]:-N/A}"
        aupr="${RESULTS_AUPR[$key]:-N/A}"
        f1="${RESULTS_F1[$key]:-N/A}"
        status="${EVAL_STATUS[$key]:-N/A}"
        
        if [[ "$status" == "success" ]]; then
            printf "${CYAN}║${NC} %-18s %-10s ${GREEN}%-8s${NC} %-8s %-8s ${GREEN}%-8s${NC} ${CYAN}║${NC}\n" \
                "$method" "$task" "$auroc" "$aupr" "$f1" "✓"
            n_success=$((n_success + 1))
            
            # 检查是否是最佳结果
            if [[ "$auroc" != "N/A" ]]; then
                auroc_float=$(echo "$auroc" | tr -d ' ')
                if (( $(echo "$auroc_float > $best_auroc" | bc -l) )); then
                    best_auroc=$auroc_float
                    best_method=$method
                    best_task=$task
                fi
            fi
        else
            printf "${CYAN}║${NC} %-18s %-10s ${RED}%-8s${NC} %-8s %-8s ${RED}%-8s${NC} ${CYAN}║${NC}\n" \
                "$method" "$task" "-" "-" "-" "✗"
            n_failed=$((n_failed + 1))
        fi
        
        # 写入汇总文件
        echo "| $method | $task | $auroc | $aupr | $f1 | $status |" >> "$SUMMARY_FILE"
    done
done

echo -e "${CYAN}╠══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║${NC} ${BOLD}统计${NC}                                                               ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}   成功: ${GREEN}$n_success${NC}    失败: ${RED}$n_failed${NC}    耗时: ${YELLOW}${elapsed}秒${NC}                  ${CYAN}║${NC}"

if [[ -n "$best_method" ]]; then
    echo -e "${CYAN}║${NC}   ${BOLD}最佳结果:${NC} ${GREEN}${best_method}/${best_task}${NC} AUROC=${GREEN}${best_auroc}${NC}              ${CYAN}║${NC}"
fi

echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"

# 完成汇总文件
{
    echo ""
    echo "统计:"
    echo "  成功: $n_success"
    echo "  失败: $n_failed"
    echo "  耗时: ${elapsed}秒"
    if [[ -n "$best_method" ]]; then
        echo "  最佳: ${best_method}/${best_task} AUROC=${best_auroc}"
    fi
} >> "$SUMMARY_FILE"

echo ""
echo -e "详细日志: ${YELLOW}$LOG_DIR${NC}"
echo -e "结果汇总: ${YELLOW}$SUMMARY_FILE${NC}"
echo ""

# =============================================================================
# 低 AUROC 警告
# =============================================================================
echo -e "${YELLOW}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║${NC}${BOLD}                      性能检查${NC}                                     ${YELLOW}║${NC}"
echo -e "${YELLOW}╠══════════════════════════════════════════════════════════════════╣${NC}"

low_perf=false
for task in $TASKS; do
    for method in $METHODS; do
        key="${method}_${task}_${LEVEL}"
        auroc="${RESULTS_AUROC[$key]}"
        
        if [[ "$auroc" != "N/A" && "$auroc" != "" ]]; then
            auroc_float=$(echo "$auroc" | tr -d ' ')
            if (( $(echo "$auroc_float < 0.60" | bc -l) )); then
                echo -e "${YELLOW}║${NC}  ⚠️  ${method}/${task}: AUROC=${auroc} (低于 0.60，可能有问题)       ${YELLOW}║${NC}"
                low_perf=true
            fi
        fi
    done
done

if [[ "$low_perf" == "true" ]]; then
    echo -e "${YELLOW}║${NC}                                                                  ${YELLOW}║${NC}"
    echo -e "${YELLOW}║${NC}  建议检查:                                                       ${YELLOW}║${NC}"
    echo -e "${YELLOW}║${NC}  1. 方法实现是否符合原论文                                       ${YELLOW}║${NC}"
    echo -e "${YELLOW}║${NC}  2. 特征提取是否正确 (full_attention vs attention_diags)        ${YELLOW}║${NC}"
    echo -e "${YELLOW}║${NC}  3. Token-level 模型保存/加载是否正常                            ${YELLOW}║${NC}"
    echo -e "${YELLOW}║${NC}  4. 运行 diagnose_and_retrain.py --diagnose-only 查看详情       ${YELLOW}║${NC}"
else
    echo -e "${YELLOW}║${NC}  ${GREEN}✓ 所有方法 AUROC >= 0.60${NC}                                        ${YELLOW}║${NC}"
fi

echo -e "${YELLOW}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""