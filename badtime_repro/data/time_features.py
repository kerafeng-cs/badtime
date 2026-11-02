"""Đặc trưng thời gian dạng 'timeF' của TSLib."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd


class TimeFeatureEncoder(ABC):
    n_features: int = 0

    @abstractmethod
    def encode(self, dates: pd.Series) -> np.ndarray:
        """Trả về [len(dates), n_features]."""


class HourlyTimeFeatures(TimeFeatureEncoder):
    """freq='h': HourOfDay, DayOfWeek, DayOfMonth, DayOfYear, mỗi đặc trưng trong [-0.5, 0.5]."""

    n_features = 4

    def encode(self, dates: pd.Series) -> np.ndarray:
        idx = pd.DatetimeIndex(dates)
        feats = [
            idx.hour / 23.0 - 0.5,
            idx.dayofweek / 6.0 - 0.5,
            (idx.day - 1) / 30.0 - 0.5,
            (idx.dayofyear - 1) / 365.0 - 0.5,
        ]
        return np.stack([np.asarray(f, dtype=np.float64) for f in feats], axis=1)


TIME_FEATURES: Dict[str, TimeFeatureEncoder] = {"h": HourlyTimeFeatures()}
