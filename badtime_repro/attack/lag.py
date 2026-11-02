"""Vị trí trigger theo độ trễ (Thuật toán 3, Công thức 6-7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from ..common.errors import PlanError


@dataclass(frozen=True)
class LagEstimate:
    lag: int
    scores: Dict[int, float]        # delta -> tổng tương quan (chỉ các delta tính được)


class LagEstimator:
    """delta* = argmax_{delta in [L, t_IN]} sum_{v_j in V_t} sum_t
                Corr(X^{v_i}[t : t+t_IN-delta], X^{v_j}[t+delta : t+t_IN]).

    Quy tắc biên của dự án (A11): bỏ delta có t_IN - delta < min_corr_len; bỏ cửa sổ có phương sai ~0
    (đóng góp 0 vào tổng). Hòa điểm thì chọn delta nhỏ hơn.
    """

    def __init__(self, seq_len: int, min_corr_len: int):
        self.seq_len = seq_len
        self.min_corr_len = min_corr_len

    def lagged_corr_sum(self, a: np.ndarray, b: np.ndarray, starts: np.ndarray, lag: int) -> Tuple[float, int]:
        """Tổng theo cửa sổ của Corr(a[s : s+T-lag], b[s+lag : s+T]). Trả về (tổng, số cửa sổ hợp lệ)."""
        return self._lagged_corr_sum(_PairPrefix(a, b), starts, lag)

    def _lagged_corr_sum(self, pp: "_PairPrefix", starts: np.ndarray, lag: int) -> Tuple[float, int]:
        n = self.seq_len - lag
        if n < self.min_corr_len:
            return float("nan"), 0

        s = np.asarray(starts, dtype=np.int64)
        pa, paa, pb, pbb = pp.pa, pp.paa, pp.pb, pp.pbb
        pab = _prefix(pp.a[: len(pp.a) - lag] * pp.b[lag:])

        sum_x = pa[s + n] - pa[s]
        sum_xx = paa[s + n] - paa[s]
        sum_y = pb[s + lag + n] - pb[s + lag]
        sum_yy = pbb[s + lag + n] - pbb[s + lag]
        sum_xy = pab[s + n] - pab[s]

        mx = sum_x / n
        my = sum_y / n
        var_x = sum_xx / n - mx * mx
        var_y = sum_yy / n - my * my
        cov = sum_xy / n - mx * my

        valid = (var_x > 1e-10) & (var_y > 1e-10)
        if not np.any(valid):
            return float("nan"), 0

        corr = cov[valid] / np.sqrt(var_x[valid] * var_y[valid])
        return float(corr.sum()), int(valid.sum())

    def estimate(self, series: np.ndarray, channel: int, target_idx: Sequence[int],
                 starts: np.ndarray, min_lag: int) -> LagEstimate:
        lo = min_lag
        hi = self.seq_len - self.min_corr_len
        if hi < lo:
            raise PlanError(
                f"Không có độ trễ hợp lệ trong [{lo}, {self.seq_len}] với min_corr_len={self.min_corr_len}."
            )

        pairs = [_PairPrefix(series[:, channel], series[:, tch]) for tch in target_idx]   # tổng tiền tố tính một lần
        scores: Dict[int, float] = {}
        for lag in range(lo, hi + 1):
            total = 0.0
            ok = True
            for pp in pairs:
                value, n_valid = self._lagged_corr_sum(pp, starts, lag)
                if n_valid == 0 or not np.isfinite(value):
                    ok = False
                    break

                total += value

            if ok:
                scores[lag] = total

        if not scores:
            raise PlanError(f"Kênh {channel}: không có độ trễ nào đủ dữ liệu để tính tương quan.")

        best = max(scores, key=lambda d: (scores[d], -d))
        return LagEstimate(int(best), scores)


class _PairPrefix:
    """Tổng tiền tố của a, a^2, b, b^2 (không phụ thuộc độ trễ), tính một lần cho mỗi cặp kênh."""

    def __init__(self, a: np.ndarray, b: np.ndarray):
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.pa, self.paa = _prefix(self.a), _prefix(self.a * self.a)
        self.pb, self.pbb = _prefix(self.b), _prefix(self.b * self.b)


def _prefix(v: np.ndarray) -> np.ndarray:
    """P[k] = sum_{t < k} v[t]."""
    out = np.zeros(len(v) + 1)
    np.cumsum(v, out=out[1:])
    return out
