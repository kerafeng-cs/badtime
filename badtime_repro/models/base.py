"""Giao diện mô hình dự báo."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..config.schema import ExperimentConfig


class Forecaster(nn.Module):
    """Bọc mô hình: chỉ nhận lịch sử x [B, seq_len, C] và đặc trưng thời gian của lịch sử [B, seq_len, F].

    Đầu vào bộ giải mã (x_dec, x_mark_dec) luôn là None, nên nhãn tương lai không thể lọt vào mô hình.
    Trả về dự báo [B, pred_len, C].
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        return self.inner(x, x_mark, None, None)


class ModelFactory(ABC):
    @abstractmethod
    def build(self, cfg: ExperimentConfig, n_channels: int) -> Forecaster:
        ...
