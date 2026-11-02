"""Chọn cặp trạng thái (mô hình, trigger) cuối cùng từ các epoch."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..metrics import Metrics


@dataclass(frozen=True)
class StateSnapshot:
    epoch: int                          # đếm từ 1
    val: Metrics
    val_loss_mse: float
    model_state: Dict[str, Any]         # bản sao CPU của state_dict
    tau: Optional[Any]                  # bản sao CPU của trigger, hoặc None


Selection = Tuple[StateSnapshot, Dict[str, Any]]


class StateSelector(ABC):
    @abstractmethod
    def observe(self, snapshot: StateSnapshot) -> bool:
        """Ghi nhận trạng thái cuối epoch. Trả về True nếu nên dừng huấn luyện."""

    @abstractmethod
    def select(self) -> Selection:
        ...


class EarlyStoppingSelector(StateSelector):
    """Như TSLib: giữ trạng thái có MSE val nhỏ nhất; dừng khi `patience` epoch liên tiếp không cải thiện."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best: Optional[StateSnapshot] = None
        self.bad_epochs = 0

    def observe(self, snapshot: StateSnapshot) -> bool:
        if self.best is None or snapshot.val_loss_mse < self.best.val_loss_mse:
            self.best = snapshot
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def select(self) -> Selection:
        return self.best, {"rule": "min val MSE (TSLib early stopping)", "chosen_epoch": self.best.epoch}


class CleanConstrainedAttackSelector(StateSelector):
    """Ứng viên: MAE_c(val) <= (1 + tol) * min MAE_c(val); trong đó chọn MAE_pa(val) nhỏ nhất,
    hòa thì epoch sớm hơn (A17)."""

    def __init__(self, tolerance: float):
        self.tolerance = tolerance
        self.snapshots: List[StateSnapshot] = []

    def observe(self, snapshot: StateSnapshot) -> bool:
        self.snapshots.append(snapshot)
        return False

    def select(self) -> Selection:
        best_c = min(s.val["mae_c_norm"] for s in self.snapshots)
        limit = (1.0 + self.tolerance) * best_c
        feasible = [s for s in self.snapshots if s.val["mae_c_norm"] <= limit]
        chosen = min(feasible, key=lambda s: (s.val["mae_pa_norm"], s.epoch))
        return chosen, {
            "rule": f"MAE_c(val) <= (1 + {self.tolerance}) * min MAE_c(val); argmin MAE_pa(val)",
            "min_val_mae_c": best_c,
            "limit": limit,
            "feasible_epochs": [s.epoch for s in feasible],
            "chosen_epoch": chosen.epoch,
        }
