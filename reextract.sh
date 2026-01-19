BASE_DIR="/share/home/tm902089733300000/a903202310/lys/research/hallucination-detection/outputs/features/ragtruth/Mistral-7B-Instruct-v0.3/seed_42"

# 只检查
for split_dir in "$BASE_DIR"/*; do
    echo "========================================"
    echo "Checking: $split_dir"
    python scripts/reextract.py mark \
        -d "$split_dir" \
        -m mistralai/Mistral-7B-Instruct-v0.3 \
        -v
done

# 检查 + 重新提取
for split_dir in "$BASE_DIR"/*; do
    echo "========================================"
    echo "Processing: $split_dir"
    python scripts/reextract.py all \
        -d "$split_dir" \
        -m mistralai/Mistral-7B-Instruct-v0.3 \
        --update-consolidated \
        -v
done