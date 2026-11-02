"""Dữ liệu và đối tượng dựng sẵn cho test."""

from __future__ import annotations

import os

import numpy as np

from badtime_repro.attack.plan import AttackPlan
from badtime_repro.config import load_config
from badtime_repro.data.dataset import PreparedDataset
from badtime_repro.data.scaling import StandardScaler
from badtime_repro.data.splits import RatioSplit
from badtime_repro.data.windows import WindowSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ETTH1_CSV = os.path.join(ROOT, "dataset", "ETTh1", "ETTh1.csv")
HAS_ETTH1 = os.path.isfile(ETTH1_CSV)


def synthetic_dataset(n: int = 40, seq: int = 4, pred: int = 3, channels: int = 3) -> PreparedDataset:
    raw = np.arange(n * channels, dtype=np.float64).reshape(n, channels)
    raw[:, 1] = np.sin(np.arange(n))
    bounds = RatioSplit().boundaries(n)
    scaler = StandardScaler.fit(raw[bounds.train[0]:bounds.train[1]])
    window = WindowSpec(seq, pred)
    starts = {k: window.starts(bounds, k, True) for k in bounds}
    return PreparedDataset(None, raw, scaler.transform(raw), np.zeros((n, 4)), scaler, bounds, starts,
                           [f"c{i}" for i in range(channels)], window)


def toy_plan(seq: int = 10, pred: int = 6, channels: int = 3, lam: float = 1.0) -> AttackPlan:
    return AttackPlan(
        channels=[f"c{i}" for i in range(channels)], target_idx=[channels - 1], poison_idx=[0, 1], seg_lens=[3, 2],
        lags=[4, 7], offsets=[seq - 4, seq - 7], pattern=np.full((4, 1), 5.0),
        poison_starts=np.array([0]), similar_starts=np.array([0]), different_starts=np.array([], dtype=np.int64),
        init_start=0, init_trigger=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        range_min=np.array([-1.0, -1.0]), range_max=np.array([1.0, 1.0]), interp_lambda=lam,
        seq_len=seq, pred_len=pred,
    )


def etth1_toy_plan(lam: float = 1.0) -> AttackPlan:
    """Kế hoạch có kích thước thật của slot PoC (96 -> 96, 7 kênh, 29 = 15 + 14)."""
    return AttackPlan(
        channels=["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"], target_idx=[6], poison_idx=[3, 1],
        seg_lens=[15, 14], lags=[59, 58], offsets=[37, 38], pattern=np.ones((53, 1)),
        poison_starts=np.array([0]), similar_starts=np.array([0]), different_starts=np.array([], dtype=np.int64),
        init_start=0, init_trigger=np.linspace(-1, 1, 29),
        range_min=np.array([-3.0, -3.0]), range_max=np.array([3.0, 3.0]), interp_lambda=lam,
        seq_len=96, pred_len=96,
    )


def default_config(*overrides: str):
    return load_config(overrides=list(overrides))
