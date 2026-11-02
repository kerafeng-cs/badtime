"""Bản tham chiếu numpy của phép gắn trigger (Công thức 12-13) và sửa nhãn (Công thức 14).

Bản torch (`attack.trigger.Trigger`) được đối chiếu với bản này sau mỗi lần huấn luyện.
"""

from __future__ import annotations

import numpy as np

from .plan import AttackPlan


class ReferenceInjector:
    def __init__(self, plan: AttackPlan):
        self.plan = plan

    def inject(self, x: np.ndarray, trigger: np.ndarray) -> np.ndarray:
        """x: [n, T, C]. Trả về bản sao có trigger; không sửa x."""
        out = np.array(x, dtype=np.float64, copy=True)
        lam = self.plan.interp_lambda
        for (t_s, t_e), ch, off in self.plan.segments():
            length = t_e - t_s
            seg = out[:, off:off + length, ch]
            out[:, off:off + length, ch] = (1.0 - lam) * seg + lam * np.asarray(trigger)[None, t_s:t_e]

        return out

    def poison_labels(self, y: np.ndarray) -> np.ndarray:
        """y: [n, P, C]. Thay [0, t_pin) của kênh đích bằng p; phần còn lại giữ nguyên."""
        out = np.array(y, dtype=np.float64, copy=True)
        for k, ch in enumerate(self.plan.target_idx):
            out[:, :self.plan.t_pin, ch] = self.plan.pattern[None, :, k]

        return out

    def input_mask(self) -> np.ndarray:
        """Mặt nạ [T, C] các ô đầu vào nhận thay đổi."""
        mask = np.zeros((self.plan.seq_len, len(self.plan.channels)), dtype=bool)
        for (t_s, t_e), ch, off in self.plan.segments():
            mask[off:off + (t_e - t_s), ch] = True

        return mask
