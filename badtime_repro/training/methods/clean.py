"""Huấn luyện sạch (Benign)."""

from __future__ import annotations

from typing import Tuple

import torch

from ...attack.objectives import CleanObjective, LossParts
from ..context import RunContext
from ..selection import EarlyStoppingSelector, StateSelector
from ..tensors import Batch
from .base import TrainingMethod


class CleanMethod(TrainingMethod):
    name = "clean"

    def __init__(self, ctx: RunContext):
        super().__init__(ctx)
        self.objective = CleanObjective(self.config.train.clean_loss, self.config.attack.loss_reduction)

    def batch_loss(self, model: torch.nn.Module, batch: Batch) -> Tuple[torch.Tensor, LossParts]:
        return self.objective(model(batch.x, batch.m), batch.y)

    def make_selector(self) -> StateSelector:
        return EarlyStoppingSelector(self.config.train.early_stopping_patience)
