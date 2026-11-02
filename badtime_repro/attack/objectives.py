"""Hàm mục tiêu (torch).

    SquaredPlusAbsolute       ||d||_2^2 + ||d||_1 với phép tổng hợp 'sum' (theo mẫu) hoặc 'mean' (theo phần tử)
    CleanObjective            loss của nhánh sạch: 'mse' (TSLib) hoặc 'mse_mae' (Công thức 16)
    BackdoorModelObjective    Công thức 15-18: a1 L_c + a2 L_p^{V_t} + a3 L_p^{~V_t}
    TriggerObjective          Công thức 20: b1 L_p^{V_t} + b2 L_p^{~V_t} + b3 L_smooth + b4 L_range
Mỗi objective trả về (tổng, {tên_thành_phần: giá_trị}).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .trigger import Trigger

LossParts = Dict[str, torch.Tensor]


class SquaredPlusAbsolute:
    def __init__(self, reduction: str):
        if reduction not in ("sum", "mean"):
            raise ValueError(f"reduction không hợp lệ: {reduction}")

        self.reduction = reduction

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d = pred - target
        if self.reduction == "mean":
            return d.pow(2).mean() + d.abs().mean()

        return (d.pow(2).sum() + d.abs().sum()) / pred.shape[0]


class CleanObjective:
    KINDS = ("mse", "mse_mae")

    def __init__(self, kind: str, reduction: str):
        if kind not in self.KINDS:
            raise ValueError(f"clean_loss không hợp lệ: {kind}")

        self.kind = kind
        self.sq_abs = SquaredPlusAbsolute(reduction)

    def __call__(self, pred: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, LossParts]:
        loss = F.mse_loss(pred, y) if self.kind == "mse" else self.sq_abs(pred, y)
        return loss, {"clean": loss}


class _PoisonTerms:
    """L_p^{V_t} (Công thức 17) và L_p^{~V_t} (Công thức 18) trên các mẫu bị chọn."""

    def __init__(self, n_channels: int, target_idx: Sequence[int], reduction: str):
        self.target: List[int] = list(target_idx)
        self.other: List[int] = [c for c in range(n_channels) if c not in self.target]
        self.sq_abs = SquaredPlusAbsolute(reduction)

    def __call__(self, pred_p: torch.Tensor, y_poisoned_p: torch.Tensor,
                 y_orig_p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        l_target = self.sq_abs(pred_p[:, :, self.target], y_poisoned_p[:, :, self.target])
        if self.other:
            l_other = self.sq_abs(pred_p[:, :, self.other], y_orig_p[:, :, self.other])
        else:
            l_other = pred_p.new_zeros(())

        return l_target, l_other


class BackdoorModelObjective:
    """Nhóm rỗng trong lô bị bỏ qua (không chia cho 0, không lấy thêm mẫu)."""

    def __init__(self, n_channels: int, target_idx: Sequence[int], weights: Sequence[float], reduction: str):
        self.a1, self.a2, self.a3 = (float(w) for w in weights)
        self.sq_abs = SquaredPlusAbsolute(reduction)
        self.poison_terms = _PoisonTerms(n_channels, target_idx, reduction)

    def __call__(self, pred: torch.Tensor, y: torch.Tensor, y_poisoned: torch.Tensor,
                 is_poison: torch.Tensor) -> Tuple[torch.Tensor, LossParts]:
        parts: LossParts = {}
        total = pred.new_zeros(())
        clean = ~is_poison
        if bool(clean.any()):
            parts["clean"] = self.sq_abs(pred[clean], y[clean])
            total = total + self.a1 * parts["clean"]

        if bool(is_poison.any()):
            l_t, l_o = self.poison_terms(pred[is_poison], y_poisoned[is_poison], y[is_poison])
            parts["poison_target"] = l_t
            parts["poison_other"] = l_o
            total = total + self.a2 * l_t + self.a3 * l_o

        return total, parts


class TriggerObjective:
    def __init__(self, n_channels: int, target_idx: Sequence[int], weights: Sequence[float],
                 reduction: str, smooth_eps: float):
        self.b1, self.b2, self.b3, self.b4 = (float(w) for w in weights)
        self.smooth_eps = float(smooth_eps)
        self.poison_terms = _PoisonTerms(n_channels, target_idx, reduction)

    def __call__(self, pred_p: torch.Tensor, y_poisoned_p: torch.Tensor, y_orig_p: torch.Tensor,
                 trigger: Trigger) -> Tuple[torch.Tensor, LossParts]:
        l_t, l_o = self.poison_terms(pred_p, y_poisoned_p, y_orig_p)
        l_s = trigger.smooth_loss(self.smooth_eps)
        l_r = trigger.range_loss()
        total = self.b1 * l_t + self.b2 * l_o + self.b3 * l_s + self.b4 * l_r
        return total, {"poison_target": l_t, "poison_other": l_o, "smooth": l_s, "range": l_r}
