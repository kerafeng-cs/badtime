"""Mặt tiền (facade) của ứng dụng: kiểm tra tổ hợp -> chuẩn bị dữ liệu -> huấn luyện một lần chạy.

`resolve` không cần torch (dùng cho lệnh `check`); `train` mới import tầng huấn luyện.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Tuple

from .config.schema import ExperimentConfig
from .data.dataset import PreparedDataset
from .data.pipeline import DataPipeline
from .runs.store import RunStore
from .support import SupportChecker

log = logging.getLogger(__name__)


@dataclass
class Experiment:
    project_root: str
    checker: SupportChecker = field(default_factory=SupportChecker)

    def resolve(self, cfg: ExperimentConfig) -> Tuple[ExperimentConfig, PreparedDataset]:
        """Ném ConfigInvalidError / UnsupportedSelectionError / DataUnavailableError nếu không chạy được."""
        report = self.checker.check(cfg)
        resolved = report.raise_if_blocked()
        for message in report.messages:
            log.info(message)

        return resolved, DataPipeline(self.project_root).prepare(resolved)

    def train(self, cfg: ExperimentConfig, method: str, seed: int) -> RunStore:
        import torch

        from .training.context import RunContext, resolve_device
        from .training.methods import TRAINING_METHODS
        from .training.trainer import Trainer

        resolved, dataset = self.resolve(cfg.replace(method=method))
        if resolved.train.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)

        store = RunStore.create(self.project_root, resolved, resolved.method, seed)
        store.mark_running(resolved.method, seed)
        log.info("Thư mục lần chạy: %s", store.path)
        try:
            ctx = RunContext(resolved, dataset, resolve_device(resolved.train.device), seed, store)
            method_impl = TRAINING_METHODS.resolve(resolved.method)(ctx)
            outcome = Trainer(ctx, method_impl).run()
        except BaseException as exc:
            store.mark_failed(exc)
            raise

        store.mark_completed(outcome.selected_epoch)
        return store
