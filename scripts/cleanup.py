#!/usr/bin/env python3
"""Cleanup script for removing unused files and directories.

运行此脚本以清理项目中的冗余文件：
- .ipynb_checkpoints 目录
- __pycache__ 目录
- 空文件

Usage:
    python scripts/cleanup.py --dry-run   # 预览将要删除的文件
    python scripts/cleanup.py             # 实际执行删除
"""
import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List


def find_checkpoints(root: Path) -> List[Path]:
    """Find all .ipynb_checkpoints directories."""
    return list(root.rglob(".ipynb_checkpoints"))


def find_pycache(root: Path) -> List[Path]:
    """Find all __pycache__ directories."""
    return list(root.rglob("__pycache__"))


def find_empty_files(root: Path) -> List[Path]:
    """Find empty files (except __init__.py)."""
    empty_files = []
    for f in root.rglob("*"):
        if f.is_file() and f.stat().st_size == 0:
            if f.name != "__init__.py":
                empty_files.append(f)
    return empty_files


def cleanup(root: Path, dry_run: bool = True) -> None:
    """Clean up unused files and directories."""
    root = Path(root)
    
    # Find items to delete
    checkpoints = find_checkpoints(root)
    pycaches = find_pycache(root)
    empty_files = find_empty_files(root)
    
    print(f"Found {len(checkpoints)} .ipynb_checkpoints directories")
    print(f"Found {len(pycaches)} __pycache__ directories")
    print(f"Found {len(empty_files)} empty files")
    
    if dry_run:
        print("\n[DRY RUN] Would delete:")
        for p in checkpoints:
            print(f"  [DIR] {p}")
        for p in pycaches:
            print(f"  [DIR] {p}")
        for f in empty_files:
            print(f"  [FILE] {f}")
        return
    
    # Actually delete
    deleted_dirs = 0
    deleted_files = 0
    
    for p in checkpoints:
        try:
            shutil.rmtree(p)
            deleted_dirs += 1
            print(f"Deleted: {p}")
        except Exception as e:
            print(f"Failed to delete {p}: {e}")
    
    for p in pycaches:
        try:
            shutil.rmtree(p)
            deleted_dirs += 1
            print(f"Deleted: {p}")
        except Exception as e:
            print(f"Failed to delete {p}: {e}")
    
    for f in empty_files:
        try:
            f.unlink()
            deleted_files += 1
            print(f"Deleted: {f}")
        except Exception as e:
            print(f"Failed to delete {f}: {e}")
    
    print(f"\nDeleted {deleted_dirs} directories and {deleted_files} files")


def main():
    parser = argparse.ArgumentParser(description="Clean up unused files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    parser.add_argument("--root", default=".", help="Root directory to clean")
    
    args = parser.parse_args()
    
    root = Path(args.root)
    if not root.exists():
        print(f"Directory not found: {root}")
        sys.exit(1)
    
    cleanup(root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
