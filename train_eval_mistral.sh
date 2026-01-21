#!/bin/bash
# =============================================================================
# Mistral-7B: 跳过特征提取，直接训练和评估所有方法
# =============================================================================
# 前提条件：特征文件已存在于:
#   outputs/features/ragtruth/Mistral-7B-Instruct-v0.3/seed_42/{task_type}/
#   outputs/features/ragtruth/Mistral-7B-Instruct-v0.3/seed_42/{task_type}_test/
#
# 用法:
#   chmod +x run_train_eval_mistral.sh
#   ./run_train_eval_mistral.sh
# =============================================================================

set -e  # 遇到错误立即退出

# =============================================================================
# 配置
# =============================================================================
MODEL_NAME="mistral_7b"
MODEL_SHORT="Mistral-7B-Instruct-v0.3"
DATASET="ragtruth"
SEED=42

# 任务类型
TASK_TYPES=("QA" "Summary" "Data2txt")

# 方法和级别组合 (与 params.yaml 中的 method_levels 一致)
# 格式: "method:level"
METHOD_LEVELS=(
    "lapeigvals:sample"
    "lookback_lens:sample"
    "lookback_lens:token"
    # "haloscope:sample"
    "hsdmvaf:sample"
    "semantic_entropy_probes:sample"
    "token_entropy:sample"
    # "tsv:sample"
    # "hypergraph:sample"
    # "hypergraph:token"
)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# 辅助函数
# =============================================================================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# 前置检查
# =============================================================================
check_prerequisites() {
    log_info "检查前置条件..."
    
    # 检查特征文件是否存在
    local features_base="outputs/features/${DATASET}/${MODEL_SHORT}/seed_${SEED}"
    
    for task in "${TASK_TYPES[@]}"; do
        local train_dir="${features_base}/${task}"
        local test_dir="${features_base}/${task}_test"
        
        if [ ! -d "$train_dir" ]; then
            log_error "训练特征目录不存在: $train_dir"
            log_error "请先运行特征提取步骤"
            exit 1
        fi
        
        if [ ! -d "$test_dir" ]; then
            log_error "测试特征目录不存在: $test_dir"
            log_error "请先运行特征提取步骤"
            exit 1
        fi
        
        # 检查是否有特征文件
        local n_train=$(find "$train_dir" -name "*.npz" 2>/dev/null | wc -l)
        local n_test=$(find "$test_dir" -name "*.npz" 2>/dev/null | wc -l)
        
        log_info "  ${task}: 训练=${n_train}, 测试=${n_test} 个特征文件"
    done
    
    # 检查分割文件
    if [ ! -d "outputs/splits" ]; then
        log_warning "分割目录不存在，将自动创建..."
        python scripts/split_dataset.py
    fi
    
    log_success "前置检查通过"
}

# =============================================================================
# 训练单个方法
# =============================================================================
train_method() {
    local task=$1
    local method=$2
    local level=$3
    
    log_info "训练: ${task}/${method}/${level}"
    
    python scripts/train_probe.py \
        dataset.name=${DATASET} \
        dataset.task_type=${task} \
        model=${MODEL_NAME} \
        model.short_name=${MODEL_SHORT} \
        method=${method} \
        method.level=${level} \
        seed=${SEED} \
        2>&1 | tee -a "logs/train_${MODEL_SHORT}_${task}_${method}_${level}.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "  训练完成: ${method}/${level}"
    else
        log_error "  训练失败: ${method}/${level}"
        return 1
    fi
}

# =============================================================================
# 评估单个方法
# =============================================================================
evaluate_method() {
    local task=$1
    local method=$2
    local level=$3
    
    log_info "评估: ${task}/${method}/${level}"
    
    python scripts/evaluate.py \
        dataset.name=${DATASET} \
        dataset.task_type=${task} \
        model=${MODEL_NAME} \
        model.short_name=${MODEL_SHORT} \
        method=${method} \
        method.level=${level} \
        seed=${SEED} \
        2>&1 | tee -a "logs/eval_${MODEL_SHORT}_${task}_${method}_${level}.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "  评估完成: ${method}/${level}"
    else
        log_error "  评估失败: ${method}/${level}"
        return 1
    fi
}

# =============================================================================
# 主流程
# =============================================================================
main() {
    echo "=============================================="
    echo "  Mistral-7B 训练和评估流程"
    echo "  模型: ${MODEL_SHORT}"
    echo "  数据集: ${DATASET}"
    echo "  Seed: ${SEED}"
    echo "=============================================="
    
    # 创建日志目录
    mkdir -p logs
    
    # 前置检查
    check_prerequisites
    
    # 记录开始时间
    START_TIME=$(date +%s)
    
    # 统计
    local total=0
    local success=0
    local failed=0
    
    # ===================
    # Phase 1: 训练所有方法
    # ===================
    echo ""
    log_info "========== Phase 1: 训练 =========="
    
    for task in "${TASK_TYPES[@]}"; do
        echo ""
        log_info ">>> 任务类型: ${task}"
        
        for ml in "${METHOD_LEVELS[@]}"; do
            IFS=':' read -r method level <<< "$ml"
            total=$((total + 1))
            
            if train_method "$task" "$method" "$level"; then
                success=$((success + 1))
            else
                failed=$((failed + 1))
            fi
        done
    done
    
    # ===================
    # Phase 2: 评估所有方法
    # ===================
    echo ""
    log_info "========== Phase 2: 评估 =========="
    
    for task in "${TASK_TYPES[@]}"; do
        echo ""
        log_info ">>> 任务类型: ${task}"
        
        for ml in "${METHOD_LEVELS[@]}"; do
            IFS=':' read -r method level <<< "$ml"
            
            # 检查模型文件是否存在
            local model_path="outputs/models/${DATASET}/${MODEL_SHORT}/seed_${SEED}/${task}/${method}/${level}/model.pkl"
            if [ -f "$model_path" ]; then
                evaluate_method "$task" "$method" "$level" || true
            else
                log_warning "  跳过评估 (模型不存在): ${method}/${level}"
            fi
        done
    done
    
    # ===================
    # Phase 3: 汇总结果
    # ===================
    echo ""
    log_info "========== Phase 3: 汇总结果 =========="
    
    python scripts/aggregate_results.py
    
    # ===================
    # 输出摘要
    # ===================
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo "=============================================="
    echo "  完成摘要"
    echo "=============================================="
    echo "  模型: ${MODEL_SHORT}"
    echo "  总任务数: $total"
    echo "  成功: $success"
    echo "  失败: $failed"
    echo "  耗时: ${DURATION} 秒"
    echo ""
    echo "  结果文件: outputs/results/summary.json"
    echo "=============================================="
    
    # 显示关键结果
    if [ -f "outputs/results/summary.json" ]; then
        echo ""
        log_info "关键指标摘要:"
        python3 -c "
import json
with open('outputs/results/summary.json') as f:
    data = json.load(f)
    
# 过滤当前模型的结果
results = [r for r in data.get('results', []) if '${MODEL_SHORT}' in r.get('model', '')]

if results:
    print(f'  找到 {len(results)} 个评估结果')
    
    # 按方法分组计算平均 AUROC
    from collections import defaultdict
    method_scores = defaultdict(list)
    
    for r in results:
        method = r.get('method', 'unknown')
        auroc = r.get('sample_auroc') or r.get('auroc')
        if auroc:
            method_scores[method].append(auroc)
    
    print('  ')
    print('  方法平均 AUROC:')
    for method, scores in sorted(method_scores.items()):
        avg = sum(scores) / len(scores)
        print(f'    {method}: {avg:.4f}')
else:
    print('  未找到评估结果')
"
    fi
}

# 运行主流程
main "$@"