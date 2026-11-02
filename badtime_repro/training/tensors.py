"""Tensor của một phân vùng (CPU, float32). Lô được chuyển lên thiết bị khi dùng."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data.dataset import PreparedDataset


def as_float_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)


@dataclass(frozen=True)
class SplitTensors:
    name: str
    starts: np.ndarray        # [n] s của từng cửa sổ, tăng dần
    x: torch.Tensor           # [n, seq_len, C]
    y: torch.Tensor           # [n, pred_len, C]
    m: torch.Tensor           # [n, seq_len, F]

    @classmethod
    def from_dataset(cls, dataset: PreparedDataset, split: str) -> "SplitTensors":
        x, y, m = dataset.windows(split)
        return cls(split, dataset.starts[split], as_float_tensor(x), as_float_tensor(y), as_float_tensor(m))

    def __len__(self) -> int:
        return len(self.starts)

    def y_numpy(self) -> np.ndarray:
        return self.y.numpy().astype(np.float64)


@dataclass(frozen=True)
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    m: torch.Tensor
    idx: torch.Tensor         # vị trí của các mẫu trong SplitTensors (CPU, long)
