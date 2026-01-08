#!/usr/bin/env python3
"""analyze_requirements.py - 分析方法的特征需求。

分析指定方法需要哪些特征，帮助规划特征提取策略。

Usage:
    # 分析单个方法
    python scripts/features/analyze_requirements.py --method lapeigvals
    
    # 分析多个方法
    python scripts/features/analyze_requirements.py --methods lapeigvals lookback_lens haloscope
    
    # 分析所有已注册方法
    python scripts/features/analyze_requirements.py --all
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse

from src.features.feature_registry import (
    METHOD_REQUIREMENTS,
    get_method_requirements,
    compute_union_requirements,
    describe_requirements,
    should_store_full_attention,
    BaseFeatureType,
)


def print_separator(char: str = "=", length: int = 70):
    print(char * length)


def analyze_single_method(method: str):
    """分析单个方法的需求。"""
    req = get_method_requirements(method)
    print(describe_requirements(req))


def analyze_multiple_methods(methods: list):
    """分析多个方法的需求并集。"""
    print_separator()
    print(f"Analyzing {len(methods)} methods: {', '.join(methods)}")
    print_separator()
    
    # 打印各方法的需求
    for method in methods:
        print(f"\n--- {method} ---")
        req = get_method_requirements(method)
        print(f"  Needs full_attention: {req.needs_full_attention()}")
        print(f"  Needs hidden_states: {req.needs_hidden_states()}")
        print(f"  Needs token_probs: {req.needs_token_probs()}")
        print(f"  Uses prompt attention: {req.uses_prompt_attention}")
        
        if req.derived_features:
            print("  Derived features:")
            for df in req.derived_features:
                print(f"    - {df.feature_type.value} (scope: {df.scope.name})")
    
    # 计算并集
    print("\n" + "=" * 70)
    print("UNION REQUIREMENTS")
    print("=" * 70)
    
    union_req = compute_union_requirements(methods)
    print(describe_requirements(union_req))
    
    # 推荐策略
    print("\n" + "=" * 70)
    print("RECOMMENDED EXTRACTION STRATEGY")
    print("=" * 70)
    
    store_full_attn = should_store_full_attention(union_req)
    
    if store_full_attn:
        print("\n📦 Store full_attention: YES")
        print("   Reason: Multiple methods need attention features or complex features required")
        print("   ⚠️ High memory usage expected")
    else:
        print("\n📦 Store full_attention: NO")
        print("   Reason: Can compute derived features directly from model outputs")
        print("   ✓ Lower memory usage")
    
    if union_req.needs_hidden_states():
        print("\n📦 Store hidden_states: YES")
        hs_methods = [m for m in methods if get_method_requirements(m).needs_hidden_states()]
        print(f"   Required by: {', '.join(hs_methods)}")
    
    if union_req.needs_token_probs():
        print("\n📦 Store token_probs: YES")
        prob_methods = [m for m in methods if get_method_requirements(m).needs_token_probs()]
        print(f"   Required by: {', '.join(prob_methods)}")
    
    # 内存估算
    print("\n" + "-" * 70)
    print("MEMORY ESTIMATION (per sample, float16)")
    print("-" * 70)
    
    # 假设参数
    n_layers = 32
    n_heads = 32
    seq_len = 512  # 示例
    hidden_dim = 4096
    
    if store_full_attn:
        attn_bytes = n_layers * n_heads * seq_len * seq_len * 2  # float16
        print(f"  full_attention ({n_layers} layers, {n_heads} heads, seq_len={seq_len}):")
        print(f"    {attn_bytes / 1024**2:.1f} MB per sample")
        print(f"    {attn_bytes / 1024**3 * 1000:.1f} GB for 1000 samples")
    
    if union_req.needs_hidden_states():
        hs_bytes = n_layers * seq_len * hidden_dim * 2
        print(f"  hidden_states ({n_layers} layers, dim={hidden_dim}):")
        print(f"    {hs_bytes / 1024**2:.1f} MB per sample")


def list_all_methods():
    """列出所有已注册的方法。"""
    print_separator()
    print("REGISTERED METHODS")
    print_separator()
    
    for method_name, req in METHOD_REQUIREMENTS.items():
        print(f"\n{method_name}:")
        print(f"  - full_attention: {req.needs_full_attention()}")
        print(f"  - hidden_states: {req.needs_hidden_states()}")
        print(f"  - token_probs: {req.needs_token_probs()}")
        print(f"  - uses_prompt_attention: {req.uses_prompt_attention}")
        if req.notes:
            print(f"  - notes: {req.notes}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature requirements for hallucination detection methods"
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        help="Single method to analyze"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Multiple methods to analyze"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="List all registered methods"
    )
    
    args = parser.parse_args()
    
    if args.all:
        list_all_methods()
    elif args.methods:
        analyze_multiple_methods(args.methods)
    elif args.method:
        analyze_single_method(args.method)
    else:
        # 默认行为：分析常用方法组合
        print("No method specified. Analyzing common method combinations...")
        print("\n" + "=" * 70)
        print("MEMORY-EFFICIENT METHODS (no full_attention storage)")
        print("=" * 70)
        analyze_multiple_methods(["lapeigvals", "lookback_lens", "haloscope"])
        
        print("\n\n" + "=" * 70)
        print("FULL METHODS (requires full_attention)")
        print("=" * 70)
        analyze_multiple_methods(["hsdmvaf", "hypergraph"])


if __name__ == "__main__":
    main()
