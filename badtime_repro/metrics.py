"""Hợp đồng phép đo (docs/06). Thuần numpy để tính lại được từ dự báo đã lưu.

Mọi sai số là trung bình theo từng giá trị hợp lệ (không lấy trung bình không trọng số giữa các lô).
- mae_c / mse_c        : dự báo từ đầu vào sạch vs nhãn gốc, mọi kênh, toàn khoảng dự báo
- mae_pa / mse_pa      : dự báo từ đầu vào có trigger vs mục tiêu, kênh đích, đoạn [0, t_pin)
- mae_pn / mse_pn      : dự báo từ đầu vào có trigger vs nhãn gốc, kênh ngoài đích, toàn khoảng dự báo
- mae_rest / mse_rest  : dự báo từ đầu vào có trigger vs nhãn gốc, kênh đích, đoạn [t_pin, pred_len)
Mỗi phép đo có hai thang: chuẩn hóa (*_norm) và đơn vị gốc (*_orig).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .attack.plan import AttackPlan
from .data.scaling import StandardScaler

Metrics = Dict[str, float]


@dataclass(frozen=True)
class AttackTarget:
    """Phần của kế hoạch tác động mà phép đo cần."""

    target_idx: Tuple[int, ...]
    pattern: np.ndarray            # [t_pin, n_target], thang chuẩn hóa

    @classmethod
    def of(cls, plan: AttackPlan) -> "AttackTarget":
        return cls(tuple(plan.target_idx), plan.pattern)


class MetricCalculator:
    def __init__(self, scaler: StandardScaler):
        self.scaler = scaler

    # ----------------------------------------------------------------------------- dự báo

    def forecast(self, pred_clean: np.ndarray, y: np.ndarray,
                 pred_trig: Optional[np.ndarray] = None, target: Optional[AttackTarget] = None) -> Metrics:
        """pred_*, y: [n, P, C] thang chuẩn hóa."""
        all_ch = list(range(y.shape[-1]))
        out: Metrics = {}
        self._pair(out, "c", pred_clean, y, all_ch, all_ch)
        if pred_trig is None:
            return out

        if target is None:
            raise ValueError("Cần AttackTarget khi có dự báo từ đầu vào có trigger.")

        tgt = list(target.target_idx)
        other = [c for c in all_ch if c not in tgt]
        t_pin = target.pattern.shape[0]
        pat = np.broadcast_to(target.pattern[None, :, :], (len(y), t_pin, len(tgt)))

        self._pair(out, "pa", pred_trig[:, :t_pin][:, :, tgt], pat, tgt, tgt)
        if other:
            self._pair(out, "pn", pred_trig[:, :, other], y[:, :, other], other, other)

        if t_pin < y.shape[1]:
            self._pair(out, "rest", pred_trig[:, t_pin:][:, :, tgt], y[:, t_pin:][:, :, tgt], tgt, tgt)

        return out

    def _pair(self, out: Metrics, tag: str, pred: np.ndarray, ref: np.ndarray,
              pred_ch: List[int], ref_ch: List[int]) -> None:
        mae, mse, cnt = _mae_mse(pred, ref)
        out[f"mae_{tag}_norm"], out[f"mse_{tag}_norm"] = mae, mse
        mae, mse, _ = _mae_mse(self.scaler.inverse(pred, pred_ch), self.scaler.inverse(ref, ref_ch))
        out[f"mae_{tag}_orig"], out[f"mse_{tag}_orig"] = mae, mse
        out[f"n_values_{tag}"] = cnt

    # ----------------------------------------------------------------------------- mức sửa đầu vào

    @staticmethod
    def input_change(x_clean: np.ndarray, x_trig: np.ndarray, plan: AttackPlan) -> Dict[str, float]:
        """Số ô thay đổi, độ lớn thay đổi, bước nhảy tại biên nối đoạn (thang chuẩn hóa)."""
        diff = np.abs(np.asarray(x_trig) - np.asarray(x_clean))
        changed = diff > 0
        jumps = []
        for (t_s, t_e), ch, off in plan.segments():
            length = t_e - t_s
            if off > 0:
                jumps.append(np.abs(x_trig[:, off, ch] - x_trig[:, off - 1, ch]))

            if off + length < x_trig.shape[1]:
                jumps.append(np.abs(x_trig[:, off + length, ch] - x_trig[:, off + length - 1, ch]))

        jumps_all = np.concatenate(jumps) if jumps else np.zeros(1)
        return {
            "changed_cells_per_window_mean": float(changed.sum(axis=(1, 2)).mean()),
            "registered_cells_per_window": int(plan.trigger_len),
            "abs_change_mean_over_registered_cells": float(diff.sum() / (len(x_clean) * plan.trigger_len)),
            "abs_change_max": float(diff.max()),
            "boundary_jump_mean": float(jumps_all.mean()),
            "boundary_jump_max": float(jumps_all.max()),
        }


def _mae_mse(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.abs(d).mean()), float((d * d).mean()), int(d.size)

