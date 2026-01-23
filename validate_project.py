#!/usr/bin/env python3
"""Comprehensive validation script for hallucination-detection project.

检查项目的完整性和正确性，包括：
1. 配置文件一致性
2. 方法注册和类路径
3. 特征需求匹配
4. 类型和shape兼容性
5. 级别支持验证

Usage:
    python validate_project.py
"""
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import importlib

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ANSI颜色
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def check_pass(msg: str):
    logger.info(f"{GREEN}✓ PASS{RESET}: {msg}")

def check_fail(msg: str):
    logger.error(f"{RED}✗ FAIL{RESET}: {msg}")

def check_warn(msg: str):
    logger.warning(f"{YELLOW}⚠ WARN{RESET}: {msg}")


class ProjectValidator:
    """项目验证器"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passes = []
    
    def run_all_checks(self):
        """运行所有检查"""
        print("\n" + "=" * 70)
        print("Project Validation Report")
        print("=" * 70 + "\n")
        
        self.check_config_files()
        self.check_method_registrations()
        self.check_feature_requirements()
        self.check_method_implementations()
        self.check_level_support()
        self.check_haloscope_fix()
        self.check_extractor_fix()
        self.check_config_hierarchy()
        
        self.print_summary()
    
    def check_config_files(self):
        """检查配置文件"""
        print("\n--- 1. Config Files Check ---\n")
        
        config_dir = PROJECT_ROOT / "config"
        
        # 检查必要的配置文件
        required_configs = [
            "config.yaml",
            "method/haloscope.yaml",
            "method/lapeigvals.yaml",
            "method/hsdmvaf.yaml",
            "method/hypergraph.yaml",
            "features/default.yaml",
            "features/haloscope.yaml",
        ]
        
        for cfg_path in required_configs:
            full_path = config_dir / cfg_path
            if full_path.exists():
                check_pass(f"Config exists: {cfg_path}")
                self.passes.append(f"Config: {cfg_path}")
            else:
                check_fail(f"Config missing: {cfg_path}")
                self.errors.append(f"Config missing: {cfg_path}")
        
        # 检查 params.yaml
        params_path = PROJECT_ROOT / "params.yaml"
        if params_path.exists():
            check_pass("params.yaml exists")
            self.passes.append("params.yaml")
            
            # 验证 max_length 设置
            import yaml
            with open(params_path) as f:
                params = yaml.safe_load(f)
            
            max_length = params.get("features", {}).get("max_length")
            if max_length and max_length >= 8192:
                check_pass(f"max_length={max_length} (H100 optimized)")
            elif max_length:
                check_warn(f"max_length={max_length} might be too small for H100")
                self.warnings.append(f"max_length={max_length} might be small")
        else:
            check_fail("params.yaml missing")
            self.errors.append("params.yaml missing")
    
    def check_method_registrations(self):
        """检查方法注册"""
        print("\n--- 2. Method Registrations Check ---\n")
        
        try:
            from src.core import METHODS
            from src.methods import create_method
            
            # 检查所有方法是否可以实例化
            methods_to_check = [
                "lapeigvals",
                "lookback_lens", 
                "haloscope",
                "hsdmvaf",
                "hypergraph",
                "token_entropy",
                "semantic_entropy_probes",
                "act",
                "ensemble",
            ]
            
            for method_name in methods_to_check:
                try:
                    method = create_method(method_name)
                    check_pass(f"Method '{method_name}' can be created")
                    self.passes.append(f"Method: {method_name}")
                except Exception as e:
                    check_fail(f"Method '{method_name}' creation failed: {e}")
                    self.errors.append(f"Method {method_name}: {e}")
        
        except Exception as e:
            check_fail(f"Failed to import method registry: {e}")
            self.errors.append(f"Registry import: {e}")
    
    def check_feature_requirements(self):
        """检查特征需求定义"""
        print("\n--- 3. Feature Requirements Check ---\n")
        
        try:
            from src.features.registry import (
                METHOD_FEATURE_REQUIREMENTS,
                get_combined_requirements,
            )
            
            # 检查所有方法都有特征需求定义
            expected_methods = [
                "lapeigvals", "lookback_lens", "haloscope", 
                "hsdmvaf", "hypergraph", "token_entropy",
                "semantic_entropy_probes", "act"
            ]
            
            for method in expected_methods:
                if method in METHOD_FEATURE_REQUIREMENTS:
                    req = METHOD_FEATURE_REQUIREMENTS[method]
                    features = []
                    if req.attention_diags: features.append("attn_diags")
                    if req.hidden_states: features.append("hidden_states")
                    if req.full_attention: features.append("full_attention")
                    if req.token_probs: features.append("token_probs")
                    
                    check_pass(f"'{method}' requirements: {', '.join(features) or 'minimal'}")
                    self.passes.append(f"Requirements: {method}")
                else:
                    check_warn(f"'{method}' has no explicit requirements")
                    self.warnings.append(f"No requirements for: {method}")
            
            # 测试组合需求
            test_methods = ["lapeigvals", "haloscope", "hsdmvaf"]
            combined = get_combined_requirements(test_methods)
            check_pass(f"Combined requirements work for {test_methods}")
            
        except Exception as e:
            check_fail(f"Feature requirements check failed: {e}")
            self.errors.append(f"Feature requirements: {e}")
    
    def check_method_implementations(self):
        """检查方法实现"""
        print("\n--- 4. Method Implementations Check ---\n")
        
        import yaml
        config_dir = PROJECT_ROOT / "config" / "method"
        
        for yaml_file in config_dir.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    config = yaml.safe_load(f)
                
                method_name = config.get("name", yaml_file.stem)
                cls_path = config.get("cls_path")
                
                if cls_path:
                    # 验证类路径是否有效
                    parts = cls_path.rsplit(".", 1)
                    if len(parts) == 2:
                        module_path, class_name = parts
                        try:
                            module = importlib.import_module(module_path)
                            cls = getattr(module, class_name)
                            check_pass(f"'{method_name}': {cls_path} is valid")
                            self.passes.append(f"cls_path: {method_name}")
                        except (ImportError, AttributeError) as e:
                            check_fail(f"'{method_name}': {cls_path} is INVALID - {e}")
                            self.errors.append(f"Invalid cls_path: {method_name}")
                else:
                    check_warn(f"'{method_name}' has no cls_path (uses registry)")
                    
            except Exception as e:
                check_fail(f"Failed to check {yaml_file.name}: {e}")
                self.errors.append(f"Config parse: {yaml_file.name}")
    
    def check_level_support(self):
        """检查级别支持"""
        print("\n--- 5. Level Support Check ---\n")
        
        try:
            from src.methods import create_method
            
            methods_levels = {
                "lapeigvals": "sample",
                "haloscope": "sample",
                "hsdmvaf": "sample",  # 也支持token
                "hypergraph": "both",  # 支持token和sample
            }
            
            for method_name, expected_level in methods_levels.items():
                try:
                    method = create_method(method_name)
                    supports_token = getattr(method, 'supports_token_level', False)
                    
                    if expected_level == "both" and supports_token:
                        check_pass(f"'{method_name}' supports both sample and token level")
                    elif expected_level == "sample" and not supports_token:
                        check_pass(f"'{method_name}' supports sample level only")
                    else:
                        check_warn(f"'{method_name}' level support may not match expected")
                        
                except Exception as e:
                    check_fail(f"Level check for '{method_name}' failed: {e}")
                    self.errors.append(f"Level check: {method_name}")
                    
        except Exception as e:
            check_fail(f"Level support check failed: {e}")
            self.errors.append(f"Level support: {e}")
    
    def check_haloscope_fix(self):
        """检查 HaloScope 修复"""
        print("\n--- 6. HaloScope Fix Check ---\n")
        
        try:
            from src.methods.haloscope import HaloScopeDetector
            import numpy as np
            import torch
            
            # 测试 predict_proba 的 squeeze 问题修复
            config = {
                'detection': {'layer_selection': 'middle'},
                'feature_config': {'token_selection': 'last'},
                'svd_config': {'n_components': 10, 'k': 5},
                'mlp_config': {'hidden_dim': 1024, 'epochs': 1, 'batch_size': 32},
            }
            
            detector = HaloScopeDetector(config)
            
            # 模拟训练
            np.random.seed(42)
            fake_features = np.random.randn(100, 256).astype(np.float32)
            fake_labels = np.random.randint(0, 2, 100).astype(np.float32)
            
            detector.train_classifier(fake_features, fake_labels)
            
            # 测试单样本预测
            single_sample = fake_features[0:1]
            try:
                proba = detector.predict_proba(single_sample)
                
                # 检查返回值是否为1D数组
                if isinstance(proba, np.ndarray) and proba.ndim == 1:
                    check_pass(f"HaloScope predict_proba returns 1D array: shape={proba.shape}")
                    self.passes.append("HaloScope squeeze fix")
                    
                    # 测试索引是否正常工作
                    _ = proba[0]
                    check_pass("HaloScope proba[0] indexing works")
                else:
                    check_fail(f"HaloScope predict_proba returns wrong shape: {proba.shape if hasattr(proba, 'shape') else type(proba)}")
                    self.errors.append("HaloScope shape issue")
                    
            except Exception as e:
                check_fail(f"HaloScope predict_proba failed: {e}")
                self.errors.append(f"HaloScope predict: {e}")
                
        except Exception as e:
            check_fail(f"HaloScope fix check failed: {e}")
            self.errors.append(f"HaloScope: {e}")
    
    def check_extractor_fix(self):
        """检查 Extractor 截断警告修复"""
        print("\n--- 7. Extractor Fix Check ---\n")
        
        try:
            # 检查代码中是否有正确的警告
            extractor_path = PROJECT_ROOT / "src" / "features" / "extractor.py"
            
            with open(extractor_path) as f:
                content = f.read()
            
            # 检查关键修复是否存在
            checks = [
                ("response_len = 0" in content and "logger.warning" in content, 
                 "Truncation warning for response_len=0"),
                ("original_total_len" in content,
                 "Original length tracking"),
            ]
            
            for condition, description in checks:
                if condition:
                    check_pass(f"Extractor fix: {description}")
                    self.passes.append(f"Extractor: {description}")
                else:
                    check_warn(f"Extractor fix may be missing: {description}")
                    self.warnings.append(f"Extractor: {description}")
                    
        except Exception as e:
            check_fail(f"Extractor fix check failed: {e}")
            self.errors.append(f"Extractor: {e}")
    
    def check_config_hierarchy(self):
        """检查配置层级"""
        print("\n--- 8. Config Hierarchy Check ---\n")
        
        import yaml
        
        try:
            # 读取所有配置文件中的 max_length
            configs = {}
            
            # params.yaml
            params_path = PROJECT_ROOT / "params.yaml"
            if params_path.exists():
                with open(params_path) as f:
                    data = yaml.safe_load(f)
                configs["params.yaml"] = data.get("features", {}).get("max_length")
            
            # features/default.yaml
            default_path = PROJECT_ROOT / "config" / "features" / "default.yaml"
            if default_path.exists():
                with open(default_path) as f:
                    data = yaml.safe_load(f)
                configs["features/default.yaml"] = data.get("max_length")
            
            # features/haloscope.yaml
            haloscope_path = PROJECT_ROOT / "config" / "features" / "haloscope.yaml"
            if haloscope_path.exists():
                with open(haloscope_path) as f:
                    data = yaml.safe_load(f)
                configs["features/haloscope.yaml"] = data.get("max_length")
            
            # 检查一致性
            values = [v for v in configs.values() if v is not None]
            
            if len(set(values)) == 1:
                check_pass(f"max_length is consistent across configs: {values[0]}")
                self.passes.append("Config consistency")
            else:
                check_warn(f"max_length varies across configs: {configs}")
                self.warnings.append(f"Config inconsistency: {configs}")
            
            # 打印配置层级
            print("\nConfig hierarchy (highest priority first):")
            print("  1. Command line (--config.features.max_length=X)")
            print(f"  2. params.yaml: max_length={configs.get('params.yaml')}")
            print(f"  3. features/haloscope.yaml: max_length={configs.get('features/haloscope.yaml')}")
            print(f"  4. features/default.yaml: max_length={configs.get('features/default.yaml')}")
            
        except Exception as e:
            check_fail(f"Config hierarchy check failed: {e}")
            self.errors.append(f"Config hierarchy: {e}")
    
    def print_summary(self):
        """打印总结"""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        print(f"\n{GREEN}Passes: {len(self.passes)}{RESET}")
        print(f"{YELLOW}Warnings: {len(self.warnings)}{RESET}")
        print(f"{RED}Errors: {len(self.errors)}{RESET}")
        
        if self.errors:
            print(f"\n{RED}ERRORS:{RESET}")
            for err in self.errors:
                print(f"  - {err}")
        
        if self.warnings:
            print(f"\n{YELLOW}WARNINGS:{RESET}")
            for warn in self.warnings:
                print(f"  - {warn}")
        
        print("\n" + "=" * 70)
        if not self.errors:
            print(f"{GREEN}All critical checks passed!{RESET}")
        else:
            print(f"{RED}Some checks failed. Please fix the errors above.{RESET}")
        print("=" * 70 + "\n")
        
        return len(self.errors) == 0


if __name__ == "__main__":
    validator = ProjectValidator()
    success = validator.run_all_checks()
    sys.exit(0 if success else 1)
