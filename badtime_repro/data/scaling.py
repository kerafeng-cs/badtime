"""Chuẩn hóa từng kênh bằng thống kê của phần huấn luyện (ddof=0 như sklearn)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..common.errors import DataUnavailableError


class StandardScaler:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        if np.any(std == 0):
            raise DataUnavailableError("Có kênh hằng trong tập huấn luyện; không chuẩn hóa được.")

        return cls(mean, std)

    @classmethod
    def from_dict(cls, d: Dict[str, List[float]]) -> "StandardScaler":
        return cls(d["mean"], d["std"])

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray, channels: Optional[Sequence[int]] = None) -> np.ndarray:
        if channels is None:
            return values * self.std + self.mean

        idx = list(channels)
        return values * self.std[idx] + self.mean[idx]

    def to_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}
