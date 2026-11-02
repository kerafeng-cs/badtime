"""Lịch learning rate theo epoch (epoch đếm từ 0)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type


class LearningRateSchedule(ABC):
    def __init__(self, base_lr: float):
        self.base_lr = base_lr

    @abstractmethod
    def lr(self, epoch: int) -> float:
        ...


class HalvingSchedule(LearningRateSchedule):
    """TSLib 'type1': lr = base * 0.5^epoch."""

    def lr(self, epoch: int) -> float:
        return self.base_lr * (0.5 ** epoch)


class ConstantSchedule(LearningRateSchedule):
    def lr(self, epoch: int) -> float:
        return self.base_lr


SCHEDULES: Dict[str, Type[LearningRateSchedule]] = {"type1": HalvingSchedule, "constant": ConstantSchedule}
