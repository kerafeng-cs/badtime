"""Kênh đích V_t và kênh bị sửa V_p (Thuật toán 2).

Điểm ảnh hưởng do một `InfluenceScorer` cung cấp. BadTime dùng GAT (`attack.gat.GatInfluenceScorer`, cần torch);
`FixedInfluenceScorer` dùng cho kiểm thử và cho các biến thể bỏ thành phần (vd. w/o-GAT chọn ngẫu nhiên).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..common.errors import PlanError
from ..common.numeric import floor_count, round_half_up


@dataclass(frozen=True)
class InfluenceResult:
    scores: np.ndarray                                   # [C], Influence(i) của từng kênh
    history: List[Dict[str, Any]] = field(default_factory=list)


class InfluenceScorer(ABC):
    @abstractmethod
    def score(self, train_norm: np.ndarray, seq_len: int, target_idx: Sequence[int], seed: int) -> InfluenceResult:
        """train_norm: [n_rows, C] phần huấn luyện đã chuẩn hóa."""


class FixedInfluenceScorer(InfluenceScorer):
    def __init__(self, scores: Sequence[float]):
        self.scores = np.asarray(scores, dtype=np.float64)

    def score(self, train_norm: np.ndarray, seq_len: int, target_idx: Sequence[int], seed: int) -> InfluenceResult:
        if len(self.scores) != train_norm.shape[1]:
            raise PlanError(f"Cần {train_norm.shape[1]} điểm ảnh hưởng, nhận {len(self.scores)}.")

        return InfluenceResult(self.scores.copy())


class TargetChannelResolver:
    """Kênh đích phải được nêu tên. Tỉ lệ alpha_S chỉ dùng để kiểm tra nhất quán với Bảng III."""

    def __init__(self, names: Sequence[str], ratio: float):
        self.names = list(names)
        self.ratio = ratio

    def resolve(self, channels: Sequence[str]) -> Tuple[List[int], Optional[str]]:
        idx = []
        for name in self.names:
            if name not in channels:
                raise PlanError(f"Kênh đích '{name}' không có trong dữ liệu {list(channels)}.")

            idx.append(list(channels).index(name))

        expected = max(1, round_half_up(self.ratio * len(channels)))
        note = None
        if expected != len(idx):
            note = f"Số kênh đích {len(idx)} khác tỷ lệ cấu hình ({self.ratio} x {len(channels)} -> {expected})."

        return idx, note


class PoisonChannelSelector:
    """V_p = TopK(Influence, k = floor(alpha_P * |V|)), tối thiểu 1; hòa điểm thì theo chỉ số kênh."""

    def __init__(self, ratio: float, exclude_targets: bool):
        self.ratio = ratio
        self.exclude_targets = exclude_targets

    def count(self, n_channels: int) -> int:
        return max(1, floor_count(self.ratio * n_channels))

    def select(self, influence: Sequence[float], target_idx: Sequence[int]) -> List[int]:
        n_poison = self.count(len(influence))
        exclude = set(target_idx) if self.exclude_targets else set()
        order = sorted(range(len(influence)), key=lambda j: (-influence[j], j))
        order = [j for j in order if j not in exclude]
        if len(order) < n_poison:
            raise PlanError(f"Không đủ kênh ứng viên ({len(order)}) để chọn {n_poison} kênh bị sửa.")

        return order[:n_poison]
