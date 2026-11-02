"""Dự báo và đánh giá một phân vùng.

TimesNet chọn chu kỳ bằng FFT lấy trung bình trên cả lô, nên dự báo phụ thuộc thành phần lô (A21):
luôn dự báo theo thứ tự thời gian, không xáo, cùng kích thước lô cho mọi nhánh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..attack.trigger import Trigger
from ..data.scaling import StandardScaler
from ..metrics import AttackTarget, MetricCalculator, Metrics
from .tensors import SplitTensors


class Predictor:
    def __init__(self, device: torch.device, batch_size: int):
        self.device = device
        self.batch_size = batch_size

    def predict(self, model: torch.nn.Module, split: SplitTensors, trigger: Optional[Trigger] = None) -> np.ndarray:
        """Trả về [n, pred_len, C] float64 (giá trị float32 của mô hình, đổi kiểu không mất mát)."""
        model.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(split), self.batch_size):
                e = s + self.batch_size
                xb = split.x[s:e].to(self.device)
                if trigger is not None:
                    xb = trigger.inject(xb, detach=True)

                outs.append(model(xb, split.m[s:e].to(self.device)).cpu())

        return torch.cat(outs).numpy().astype(np.float64)


@dataclass(frozen=True)
class EvaluationResult:
    metrics: Metrics
    pred_clean: np.ndarray
    pred_trig: Optional[np.ndarray]


class Evaluator:
    def __init__(self, scaler: StandardScaler, predictor: Predictor):
        self.calculator = MetricCalculator(scaler)
        self.predictor = predictor

    def evaluate(self, model: torch.nn.Module, split: SplitTensors,
                 trigger: Optional[Trigger] = None, target: Optional[AttackTarget] = None) -> EvaluationResult:
        y = split.y_numpy()
        pred_clean = self.predictor.predict(model, split)
        if trigger is None:
            return EvaluationResult(self.calculator.forecast(pred_clean, y), pred_clean, None)

        pred_trig = self.predictor.predict(model, split, trigger)
        return EvaluationResult(self.calculator.forecast(pred_clean, y, pred_trig, target), pred_clean, pred_trig)
