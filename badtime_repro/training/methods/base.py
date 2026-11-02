"""Giao diện phương pháp huấn luyện. Trainer chỉ làm việc với giao diện này.

Vòng đời do Trainer gọi, theo thứ tự:
    prepare()                    trước khi khởi tạo mô hình (lập kế hoạch; dùng RNG riêng)
    bind(model)                  sau khi có mô hình (tạo optimizer phụ)
    mỗi epoch:
        batch_loss(model, batch) cho từng lô của pha cập nhật mô hình
        after_model_epoch(...)   việc làm thêm sau pha mô hình (vd. cập nhật trigger)
    make_selector()              quy tắc chọn cặp trạng thái
    finalize(results)            hiện vật bổ sung sau đánh giá cuối
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch

from ...attack.objectives import LossParts
from ...attack.trigger import Trigger
from ...metrics import AttackTarget, Metrics
from ..context import RunContext
from ..selection import StateSelector
from ..tensors import Batch


class TrainingMethod(ABC):
    name: str = ""

    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.config = ctx.config

    # ----------------------------------------------------------------------------- vòng đời

    def prepare(self) -> None:
        """Mặc định không cần chuẩn bị."""

    def bind(self, model: torch.nn.Module) -> None:
        """Mặc định không có optimizer phụ."""

    @abstractmethod
    def batch_loss(self, model: torch.nn.Module, batch: Batch) -> Tuple[torch.Tensor, LossParts]:
        ...

    def after_model_epoch(self, epoch: int, model: torch.nn.Module) -> Optional[Dict[str, float]]:
        """epoch đếm từ 0. Trả về bản ghi để đưa vào history, hoặc None."""
        return None

    @abstractmethod
    def make_selector(self) -> StateSelector:
        ...

    def finalize(self, results: Dict[str, Any]) -> None:
        """Bổ sung hiện vật vào metrics.json sau đánh giá cuối."""

    # ----------------------------------------------------------------------------- trigger (nếu có)

    @property
    def trigger(self) -> Optional[Trigger]:
        return None

    @property
    def attack_target(self) -> Optional[AttackTarget]:
        return None

    def plan_dict(self) -> Optional[Dict[str, Any]]:
        return None

    def trigger_state(self) -> Optional[torch.Tensor]:
        trigger = self.trigger
        return None if trigger is None else trigger.tau.detach().cpu().clone()

    def load_trigger_state(self, tau: Optional[torch.Tensor]) -> None:
        trigger = self.trigger
        if trigger is not None and tau is not None:
            with torch.no_grad():
                trigger.tau.copy_(tau.to(trigger.tau.device))

    # ----------------------------------------------------------------------------- hiển thị

    def epoch_summary(self, val: Metrics) -> str:
        return f"val MAE_c {val['mae_c_norm']:.4f}"
