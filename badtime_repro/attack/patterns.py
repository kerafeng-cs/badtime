"""Mẫu mục tiêu p (Bảng III: Hump, t_PIN = 53). Công thức số là ĐỀ XUẤT của dự án (giả định A06).

Bảng X của v2 dùng ba dạng Hump / Upward / Up-and-down, tương ứng cone / up_trend / up_and_down trong
configs/default_config.yaml của BackTime (xiaolin-cs/BackTime@9e0a1f9).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Sequence

import numpy as np

from ..common.errors import PlanError


class TargetPattern(ABC):
    name: str = ""

    @abstractmethod
    def shape(self, length: int) -> np.ndarray:
        """Đường cong chuẩn (trước khi nhân biên độ, cộng nền), dạng [length]."""

    def build(self, length: int, amplitude: float, baseline: float, n_targets: int) -> np.ndarray:
        """p = baseline + amplitude * shape, lặp cho từng kênh đích: [length, n_targets], thang chuẩn hóa."""
        curve = baseline + amplitude * self.shape(length)
        return np.repeat(curve[:, None], n_targets, axis=1)


class InterpolatedPattern(TargetPattern):
    """Nội suy tuyến tính các điểm neo (cách đều) lên `length` điểm."""

    def __init__(self, name: str, anchors: Sequence[float]):
        self.name = name
        self.anchors = np.asarray(anchors, dtype=np.float64)

    def shape(self, length: int) -> np.ndarray:
        grid = np.linspace(0.0, len(self.anchors) - 1.0, length)
        return np.interp(grid, np.arange(len(self.anchors)), self.anchors)


class HalfSinePattern(TargetPattern):
    """sin(pi * (k + 1) / (length + 1)), k in [0, length)."""

    name = "hump_sine"

    def shape(self, length: int) -> np.ndarray:
        k = np.arange(length, dtype=np.float64)
        return np.sin(np.pi * (k + 1.0) / (length + 1.0))


PATTERNS: Dict[str, TargetPattern] = {
    p.name: p
    for p in (
        InterpolatedPattern("hump_backtime_cone", [0.1278, 0.1748, 0.2787, 0.3853, 0.2787, 0.1748, 0.1278]),
        InterpolatedPattern("upward_backtime", [0.1405, 0.2064, 0.2705, 0.2835, 0.2949, 0.3495, 0.3567]),
        InterpolatedPattern("up_and_down_backtime", [0.10, 0.1687, 0.2851, 0.09601, -0.0230, -0.0916, -0.2402]),
        HalfSinePattern(),
    )
}


def get_pattern(name: str) -> TargetPattern:
    if name not in PATTERNS:
        raise PlanError(f"Dạng mục tiêu '{name}' chưa có công thức số.")

    return PATTERNS[name]
