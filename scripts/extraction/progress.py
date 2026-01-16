"""Progress tracking module for feature extraction.

提供简单的进度跟踪和时间预估功能。
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class ProgressTracker:
    """进度跟踪器，提供进度百分比、速率和预估剩余时间。
    
    Attributes:
        total: 总样本数
        current: 当前已处理数
        desc: 进度描述
        start_time: 开始时间
    """
    
    def __init__(self, total: int, desc: str = "Processing"):
        """初始化进度跟踪器。
        
        Args:
            total: 总样本数
            desc: 进度描述文本
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = datetime.now()
        self._last_log_time = self.start_time
        self._log_interval = 10  # 每处理N个样本记录一次
    
    def update(self, n: int = 1) -> None:
        """更新进度。
        
        Args:
            n: 本次处理的样本数
        """
        self.current += n
        
        # 计算统计信息
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / max(elapsed, 1)
        remaining = (self.total - self.current) / max(rate, 0.001)
        pct = 100 * self.current / max(self.total, 1)
        
        # 记录日志
        logger.info(
            f"[{self.desc}] {self.current}/{self.total} ({pct:.1f}%) "
            f"| {rate:.2f} samples/sec | ETA: {remaining:.0f}s"
        )
    
    def get_stats(self) -> dict:
        """获取当前统计信息。
        
        Returns:
            包含进度统计的字典
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / max(elapsed, 1)
        remaining = (self.total - self.current) / max(rate, 0.001)
        
        return {
            "current": self.current,
            "total": self.total,
            "percentage": 100 * self.current / max(self.total, 1),
            "elapsed_seconds": elapsed,
            "rate": rate,
            "eta_seconds": remaining,
        }
    
    def is_complete(self) -> bool:
        """检查是否完成。
        
        Returns:
            是否已处理完所有样本
        """
        return self.current >= self.total


class SilentProgressTracker(ProgressTracker):
    """静默进度跟踪器，只在特定间隔记录日志。"""
    
    def __init__(self, total: int, desc: str = "Processing", log_interval: int = 10):
        """初始化静默进度跟踪器。
        
        Args:
            total: 总样本数
            desc: 进度描述
            log_interval: 日志记录间隔
        """
        super().__init__(total, desc)
        self._log_interval = log_interval
    
    def update(self, n: int = 1) -> None:
        """更新进度（仅在间隔时记录日志）。
        
        Args:
            n: 本次处理的样本数
        """
        self.current += n
        
        # 只在特定间隔记录日志
        if self.current % self._log_interval == 0 or self.current == self.total:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / max(elapsed, 1)
            remaining = (self.total - self.current) / max(rate, 0.001)
            pct = 100 * self.current / max(self.total, 1)
            
            logger.info(
                f"[{self.desc}] {self.current}/{self.total} ({pct:.1f}%) "
                f"| {rate:.2f} samples/sec | ETA: {remaining:.0f}s"
            )
