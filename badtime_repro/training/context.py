"""Mọi thứ một lần chạy cần, gom vào một chỗ để truyền cho Trainer và TrainingMethod."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch

from ..config.schema import ExperimentConfig
from ..data.dataset import PreparedDataset
from ..runs.store import RunStore
from .tensors import SplitTensors


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SeedPlan:
    """Mỗi nguồn ngẫu nhiên có seed riêng suy ra từ seed của lần chạy, để các nhánh khớp nhau khi so sánh."""

    MODEL_OFFSET = 0          # khởi tạo mô hình + thứ tự lô
    INFLUENCE_OFFSET = 1000   # huấn luyện GAT
    TRIGGER_OFFSET = 2000     # thứ tự mẫu khi cập nhật trigger

    def __init__(self, seed: int):
        self.seed = seed

    @property
    def model(self) -> int:
        return self.seed + self.MODEL_OFFSET

    @property
    def influence(self) -> int:
        return self.seed + self.INFLUENCE_OFFSET

    @property
    def trigger(self) -> int:
        return self.seed + self.TRIGGER_OFFSET

    @staticmethod
    def generator(seed: int) -> torch.Generator:
        gen = torch.Generator()
        gen.manual_seed(seed)
        return gen


@dataclass
class RunContext:
    config: ExperimentConfig
    dataset: PreparedDataset
    device: torch.device
    seed: int
    store: RunStore
    splits: Dict[str, SplitTensors] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.splits:
            self.splits = {name: SplitTensors.from_dataset(self.dataset, name) for name in ("train", "val", "test")}

    @property
    def seeds(self) -> SeedPlan:
        return SeedPlan(self.seed)

    @property
    def train(self) -> SplitTensors:
        return self.splits["train"]

    @property
    def val(self) -> SplitTensors:
        return self.splits["val"]

    @property
    def test(self) -> SplitTensors:
        return self.splits["test"]
