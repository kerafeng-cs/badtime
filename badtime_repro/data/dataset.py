"""Dataset đã chuẩn bị: chuỗi gốc, chuỗi chuẩn hóa, đặc trưng thời gian, ranh giới và chỉ số cửa sổ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .scaling import StandardScaler
from .sources import DataQualityReport
from .splits import SplitBounds
from .windows import WindowSpec


@dataclass(frozen=True)
class PreparedDataset:
    report: Optional[DataQualityReport]
    raw: np.ndarray                 # [n_rows, C] đơn vị gốc
    norm: np.ndarray                # [n_rows, C] thang chuẩn hóa
    marks: np.ndarray               # [n_rows, n_time_features]
    scaler: StandardScaler
    bounds: SplitBounds
    starts: Dict[str, np.ndarray]   # split -> các s hợp lệ, tăng dần
    channels: List[str]
    window: WindowSpec

    @property
    def seq_len(self) -> int:
        return self.window.seq_len

    @property
    def pred_len(self) -> int:
        return self.window.pred_len

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def windows(self, split: str, starts: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x, y, x_mark) dạng [n, seq_len, C], [n, pred_len, C], [n, seq_len, F]; thang chuẩn hóa; là bản sao."""
        if starts is None:
            starts = self.starts[split]

        x_rows = self.window.input_rows(starts)
        y_rows = self.window.label_rows(starts)
        return self.norm[x_rows], self.norm[y_rows], self.marks[x_rows]

    def train_norm(self) -> np.ndarray:
        b_s, b_e = self.bounds.train
        return self.norm[b_s:b_e]

    def profile(self) -> Dict[str, Any]:
        return {
            "report": self.report.to_dict() if self.report is not None else None,
            "bounds": self.bounds.to_dict(),
            "window_counts": {k: int(len(v)) for k, v in self.starts.items()},
            "window_first_last_start": {k: [int(v[0]), int(v[-1])] for k, v in self.starts.items()},
            "scaler": self.scaler.to_dict(),
            "channels": list(self.channels),
        }
