"""Cửa sổ trượt.

Cửa sổ có đầu vào bắt đầu ở s: đầu vào [s, s + seq_len), nhãn [s + seq_len, s + seq_len + pred_len).
Cửa sổ thuộc phân vùng p khi TOÀN BỘ đoạn nhãn nằm trong [b_s, b_e) của p.
Nếu cho phép dùng lịch sử từ phân vùng trước (không áp dụng cho train): s >= max(0, b_s - seq_len); ngược lại s >= b_s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .splits import SplitBounds


@dataclass(frozen=True)
class WindowSpec:
    seq_len: int
    pred_len: int

    def starts(self, bounds: SplitBounds, split: str, allow_history: bool) -> np.ndarray:
        b_s, b_e = bounds[split]
        if split == "train" or not allow_history:
            first = b_s
        else:
            first = max(0, b_s - self.seq_len)

        last = b_e - self.seq_len - self.pred_len     # s + seq_len + pred_len <= b_e
        if last < first:
            return np.zeros(0, dtype=np.int64)

        return np.arange(first, last + 1, dtype=np.int64)

    def input_rows(self, starts: np.ndarray) -> np.ndarray:
        return np.asarray(starts, dtype=np.int64)[:, None] + np.arange(self.seq_len)[None, :]

    def label_rows(self, starts: np.ndarray, length: Optional[int] = None) -> np.ndarray:
        """Hàng của [0, length) bước đầu của nhãn (mặc định toàn bộ pred_len)."""
        n = self.pred_len if length is None else length
        return np.asarray(starts, dtype=np.int64)[:, None] + self.seq_len + np.arange(n)[None, :]
