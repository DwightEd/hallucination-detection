#!/bin/bash
# =============================================================================
# 一键运行完整流水线
# =============================================================================
# 使用方法:
#   ./run_all.sh                    # 运行完整流水线
#   ./run_all.sh --stage train      # 只运行训练阶段
#   ./run_all.sh --stage evaluate   # 只运行评估阶段
#   ./run_all.sh --dry-run          # 只显示将要运行的命令
#
# 配置在 params.yaml 中修改:
#   - datasets: 数据集列表
#   - task_types: 任务类型列表
#   - models: 模型列表
#   - methods: 方法列表
#   - classification_levels: 分类级别列表
#   - eval_matrix: 跨任务评估矩阵
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
STAGE="all"
DRY_RUN=false
PARALLEL=false
MAX_PARALLEL=2

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --stage STAGE     Run specific stage: split, features, train, evaluate, aggregate, all"
            echo "  --dry-run         Show commands without executing"
            echo "  --parallel        Run tasks in parallel where possible"
            echo "  --max-parallel N  Maximum parallel tasks (default: 2)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 日志函数
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

# 运行命令（支持dry-run模式）
run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] $1"
    else
        log_info "Running: $1"
        eval $1
    fi
}

# 检查依赖
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v python &> /dev/null; then
        log_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    if ! command -v dvc &> /dev/null; then
        log_warning "DVC not found. Using direct Python scripts instead."
        USE_DVC=false
    else
        USE_DVC=true
    fi
    
    log_success "Dependencies check passed"
}

# 从params.yaml读取配置
read_params() {
    if [ ! -f "params.yaml" ]; then
        log_error "params.yaml not found!"
        exit 1
    fi
    
    # 使用Python读取YAML配置
    DATASETS=$(python -c "import yaml; print(' '.join([d['name'] for d in yaml.safe_load(open('params.yaml'))['datasets']]))")
    TASK_TYPES=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('params.yaml'))['task_types']))")
    METHODS=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('params.yaml'))['methods']))")
    CLASSIFICATION_LEVELS=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('params.yaml'))['classification_levels']))")
    SEED=$(python -c "import yaml; print(yaml.safe_load(open('params.yaml'))['seed'])")
    
    log_info "Configuration from params.yaml:"
    log_info "  Datasets: $DATASETS"
    log_info "  Task types: $TASK_TYPES"
    log_info "  Methods: $METHODS"
    log_info "  Classification levels: $CLASSIFICATION_LEVELS"
    log_info "  Seed: $SEED"
}

# =============================================================================
# 阶段函数
# =============================================================================

stage_split() {
    log_info "=========================================="
    log_info "Stage 1: Split Dataset"
    log_info "=========================================="
    run_cmd "python scripts/split_dataset.py"
}

stage_features() {
    log_info "=========================================="
    log_info "Stage 2: Generate Activations (Features)"
    log_info "=========================================="
    
    for dataset in $DATASETS; do
        for task_type in $TASK_TYPES; do
            log_info "Processing: dataset=$dataset, task_type=$task_type"
            run_cmd "python scripts/generate_activations.py \
                dataset.name=$dataset \
                dataset.task_type=$task_type \
                seed=$SEED"
        done
    done
}

stage_train() {
    log_info "=========================================="
    log_info "Stage 3: Train Probe (Sample + Token Level)"
    log_info "=========================================="
    
    for dataset in $DATASETS; do
        for task_type in $TASK_TYPES; do
            for method in $METHODS; do
                for level in $CLASSIFICATION_LEVELS; do
                    log_info "Training: dataset=$dataset, task=$task_type, method=$method, level=$level"
                    run_cmd "python scripts/train_probe.py \
                        dataset.name=$dataset \
                        dataset.task_type=$task_type \
                        method=$method \
                        method.classification_level=$level \
                        seed=$SEED"
                done
            done
        done
    done
}

stage_evaluate() {
    log_info "=========================================="
    log_info "Stage 4: Evaluate (Same-Task + Cross-Task)"
    log_info "=========================================="
    
    # 读取eval_matrix
    EVAL_MATRIX=$(python -c "
import yaml
config = yaml.safe_load(open('params.yaml'))
for pair in config['eval_matrix']:
    print(f\"{pair['train_task']}:{pair['eval_task']}\")
")
    
    for dataset in $DATASETS; do
        for method in $METHODS; do
            for level in $CLASSIFICATION_LEVELS; do
                for pair in $EVAL_MATRIX; do
                    train_task=$(echo $pair | cut -d: -f1)
                    eval_task=$(echo $pair | cut -d: -f2)
                    
                    is_cross="false"
                    if [ "$train_task" != "$eval_task" ]; then
                        is_cross="true"
                    fi
                    
                    log_info "Evaluating: train=$train_task -> eval=$eval_task, method=$method, level=$level"
                    run_cmd "python scripts/evaluate.py \
                        dataset.name=$dataset \
                        method=$method \
                        method.classification_level=$level \
                        seed=$SEED \
                        train_eval.train_task_types=[$train_task] \
                        train_eval.eval_task_types=[$eval_task] \
                        train_eval.cross_task_eval=$is_cross"
                done
            done
        done
    done
}

stage_aggregate() {
    log_info "=========================================="
    log_info "Stage 5: Aggregate Results"
    log_info "=========================================="
    run_cmd "python scripts/aggregate_results.py"
}

# =============================================================================
# 使用DVC运行
# =============================================================================

run_with_dvc() {
    log_info "Running with DVC pipeline..."
    
    case $STAGE in
        all)
            run_cmd "dvc repro"
            ;;
        split)
            run_cmd "dvc repro split_dataset"
            ;;
        features)
            run_cmd "dvc repro generate_activations"
            ;;
        train)
            run_cmd "dvc repro train_probe"
            ;;
        evaluate)
            run_cmd "dvc repro evaluate_same_task evaluate_cross_task"
            ;;
        aggregate)
            run_cmd "dvc repro aggregate"
            ;;
        *)
            log_error "Unknown stage: $STAGE"
            exit 1
            ;;
    esac
}

# =============================================================================
# 直接运行脚本
# =============================================================================

run_direct() {
    log_info "Running scripts directly..."
    
    case $STAGE in
        all)
            stage_split
            stage_features
            stage_train
            stage_evaluate
            stage_aggregate
            ;;
        split)
            stage_split
            ;;
        features)
            stage_features
            ;;
        train)
            stage_train
            ;;
        evaluate)
            stage_evaluate
            ;;
        aggregate)
            stage_aggregate
            ;;
        *)
            log_error "Unknown stage: $STAGE"
            exit 1
            ;;
    esac
}

# =============================================================================
# 主程序
# =============================================================================

main() {
    echo ""
    echo "=============================================="
    echo "  Hallucination Detection - Full Pipeline"
    echo "=============================================="
    echo ""
    
    check_dependencies
    read_params
    
    log_info "Running stage: $STAGE"
    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY-RUN mode: Commands will not be executed"
    fi
    
    START_TIME=$(date +%s)
    
    if [ "$USE_DVC" = true ]; then
        run_with_dvc
    else
        run_direct
    fi
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo ""
    log_success "=============================================="
    log_success "  Pipeline completed!"
    log_success "  Total time: ${ELAPSED}s"
    log_success "=============================================="
    echo ""
    
    if [ "$STAGE" = "all" ] || [ "$STAGE" = "aggregate" ]; then
        log_info "Results saved to:"
        log_info "  - outputs/results/summary.json"
        log_info "  - outputs/results/cross_task_summary.json"
        log_info "  - outputs/results/comparison_*.csv"
    fi
}

# 运行主程序
main
