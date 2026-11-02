"""Trigger có thể tối ưu (torch): gắn vào đầu vào (Công thức 12-13), sửa nhãn (Công thức 14),
và hai hàm phạt riêng của trigger (Công thức 21-22)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .plan import AttackPlan


class Trigger(nn.Module):
    """Trigger phẳng tau dài sum(seg_lens), cắt thành từng đoạn theo thứ tự plan.poison_idx."""

    def __init__(self, plan: AttackPlan, init: Optional[np.ndarray] = None):
        super().__init__()
        init = plan.init_trigger if init is None else init
        self.tau = nn.Parameter(torch.as_tensor(np.asarray(init), dtype=torch.float32).clone())
        self.seg_bounds = list(plan.seg_bounds)
        self.poison_idx = list(plan.poison_idx)
        self.offsets = list(plan.offsets)
        self.lam = float(plan.interp_lambda)
        self.target_idx = list(plan.target_idx)
        self.t_pin = int(plan.t_pin)
        self.register_buffer("pattern", torch.as_tensor(plan.pattern, dtype=torch.float32))
        self.register_buffer("range_min", torch.as_tensor(plan.range_min, dtype=torch.float32))
        self.register_buffer("range_max", torch.as_tensor(plan.range_max, dtype=torch.float32))

    def inject(self, x: torch.Tensor, detach: bool = False) -> torch.Tensor:
        """x: [B, T, C]. Trả về bản sao có trigger; không sửa x. detach=True khi mô hình đang được cập nhật."""
        tau = self.tau.detach() if detach else self.tau
        out = x.clone()
        for (t_s, t_e), ch, off in zip(self.seg_bounds, self.poison_idx, self.offsets):
            length = t_e - t_s
            seg = x[:, off:off + length, ch]
            out[:, off:off + length, ch] = (1.0 - self.lam) * seg + self.lam * tau[t_s:t_e].unsqueeze(0)

        return out

    def poison_labels(self, y: torch.Tensor) -> torch.Tensor:
        out = y.clone()
        for k, ch in enumerate(self.target_idx):
            out[:, :self.t_pin, ch] = self.pattern[:, k].unsqueeze(0)

        return out

    def smooth_loss(self, eps: float) -> torch.Tensor:
        """Công thức 21: sum [max(|tau_i - tau_{i-1}| - eps, 0)]^2 / t_TGR, chỉ trong từng đoạn."""
        total = self.tau.new_zeros(())
        for t_s, t_e in self.seg_bounds:
            if t_e - t_s < 2:
                continue

            d = self.tau[t_s + 1:t_e] - self.tau[t_s:t_e - 1]
            total = total + F.relu(d.abs() - eps).pow(2).sum()

        return total / self.tau.numel()

    def range_loss(self) -> torch.Tensor:
        """Công thức 22: phạt phần vượt [min, max] của kênh trên tập huấn luyện."""
        total = self.tau.new_zeros(())
        for k, (t_s, t_e) in enumerate(self.seg_bounds):
            seg = self.tau[t_s:t_e]
            total = total + F.relu(self.range_min[k] - seg).sum() + F.relu(seg - self.range_max[k]).sum()

        return total / self.tau.numel()

    def numpy(self) -> np.ndarray:
        return self.tau.detach().cpu().numpy().astype(np.float64)
