"""Chọn mẫu bị tác động (Thuật toán 1, Công thức 2-3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np

from ..common.errors import PlanError
from ..common.numeric import floor_count
from ..data.windows import WindowSpec


@dataclass(frozen=True)
class SampleSelection:
    similar_pos: np.ndarray        # vị trí (trong mảng starts) của T_sim, tăng dần
    different_pos: np.ndarray      # vị trí của T_diff, tăng dần
    info: Dict[str, Any]

    @property
    def all_pos(self) -> np.ndarray:
        return np.sort(np.concatenate([self.similar_pos, self.different_pos]))


class HybridSampleSelector:
    """T_sim: k_sim cửa sổ có nhãn gần mục tiêu nhất; T_diff: k_diff cửa sổ xa nhất.

    k_sim = floor(gamma * alpha_T * |D|), k_diff = floor((1 - gamma) * alpha_T * |D|) (Thuật toán 1, dòng 7-8).
    Khoảng cách Euclid ở thang chuẩn hóa (giả định A12).
    """

    def __init__(self, poison_ratio: float, similar_ratio: float):
        self.poison_ratio = poison_ratio
        self.similar_ratio = similar_ratio

    @staticmethod
    def distances(norm: np.ndarray, starts: np.ndarray, window: WindowSpec,
                  target_idx: Sequence[int], pattern: np.ndarray) -> np.ndarray:
        """d(w) = || Y_w[0:t_pin, V_t] - p ||_F cho từng cửa sổ (Công thức 3)."""
        rows = window.label_rows(starts, pattern.shape[0])
        lab = norm[rows][:, :, list(target_idx)]
        diff = lab - pattern[None, :, :]
        return np.sqrt((diff * diff).sum(axis=(1, 2)))

    def select(self, distances: np.ndarray) -> SampleSelection:
        n = len(distances)
        k_sim = floor_count(self.similar_ratio * self.poison_ratio * n)
        k_diff = floor_count((1.0 - self.similar_ratio) * self.poison_ratio * n)
        if k_sim + k_diff > n:
            raise PlanError("Ngân sách mẫu vượt số cửa sổ huấn luyện.")

        order = np.lexsort((np.arange(n), distances))      # tăng dần; hòa thì theo chỉ số
        sim = order[:k_sim]
        diff = order[n - k_diff:][::-1] if k_diff > 0 else order[:0]
        if len(np.intersect1d(sim, diff)) > 0:
            raise PlanError("Nhóm tương tự và nhóm khác biệt giao nhau; ngân sách quá lớn.")

        info = {
            "n_train_windows": int(n),
            "k_total_requested": float(self.poison_ratio * n),
            "k_total": int(k_sim + k_diff),
            "k_similar": int(k_sim),
            "k_different": int(k_diff),
            "distance_similar_max": float(distances[sim].max()) if k_sim else None,
            "distance_different_min": float(distances[diff].min()) if k_diff else None,
        }
        return SampleSelection(np.sort(sim), np.sort(diff), info)
