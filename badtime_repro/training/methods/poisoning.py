"""Khung chung cho các phương pháp đầu độc dữ liệu có trigger tối ưu được và tối ưu xen kẽ mô hình/trigger.

Lớp con chỉ cần cung cấp:
    build_plan()               -> AttackPlan (chọn mẫu, kênh, vị trí, trigger khởi tạo)
    make_model_objective()     -> objective của pha mô hình
    make_trigger_objective()   -> objective của pha trigger
BadTime là một lớp con; một bản chuyển đổi BackTime về sau cũng có thể là lớp con.
"""

from __future__ import annotations

import hashlib
import logging
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ...attack.injection import ReferenceInjector
from ...attack.objectives import BackdoorModelObjective, LossParts, TriggerObjective
from ...attack.plan import AttackPlan
from ...attack.trigger import Trigger
from ...common.errors import IntegrityError
from ...metrics import AttackTarget, MetricCalculator, Metrics
from ...runs.store import RunFiles
from ..context import RunContext
from ..selection import CleanConstrainedAttackSelector, StateSelector
from ..tensors import Batch
from .base import TrainingMethod

log = logging.getLogger(__name__)


def params_digest(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for p in module.parameters():
        h.update(p.detach().cpu().numpy().tobytes())

    return h.hexdigest()


class PoisoningMethod(TrainingMethod):
    def __init__(self, ctx: RunContext):
        super().__init__(ctx)
        self.plan: Optional[AttackPlan] = None
        self._trigger: Optional[Trigger] = None
        self.flags = torch.zeros(len(ctx.train), dtype=torch.bool)    # flags[i]: mẫu i của train bị chọn
        self.poison_pos = torch.zeros(0, dtype=torch.long)
        self.model_objective: Optional[BackdoorModelObjective] = None
        self.trigger_objective: Optional[TriggerObjective] = None
        self.trigger_optimizer: Optional[torch.optim.Optimizer] = None
        self.trigger_generator: Optional[torch.Generator] = None

    # ----------------------------------------------------------------------------- điểm mở rộng

    @abstractmethod
    def build_plan(self) -> AttackPlan:
        ...

    @abstractmethod
    def make_model_objective(self) -> BackdoorModelObjective:
        ...

    @abstractmethod
    def make_trigger_objective(self) -> TriggerObjective:
        ...

    # ----------------------------------------------------------------------------- vòng đời

    def prepare(self) -> None:
        ctx = self.ctx
        self.plan = self.build_plan()
        ctx.store.write_json(RunFiles.PLAN, self.plan.to_dict())
        log.info(
            "Kênh bị sửa: %s, đoạn %s, độ trễ %s, số cửa sổ bị chọn %d",
            [ctx.dataset.channels[i] for i in self.plan.poison_idx], self.plan.seg_lens, self.plan.lags,
            len(self.plan.poison_starts),
        )

        pos = np.searchsorted(ctx.train.starts, self.plan.poison_starts)
        if not np.array_equal(ctx.train.starts[pos], self.plan.poison_starts):
            raise IntegrityError("Cửa sổ bị chọn không thuộc tập huấn luyện.")

        self.poison_pos = torch.as_tensor(pos, dtype=torch.long)
        self.flags[self.poison_pos] = True
        self._trigger = Trigger(self.plan).to(ctx.device)
        self.model_objective = self.make_model_objective()
        self.trigger_objective = self.make_trigger_objective()

    def bind(self, model: torch.nn.Module) -> None:
        lr = self.config.attack.trigger_learning_rate
        self.trigger_optimizer = torch.optim.Adam(self._trigger.parameters(), lr=lr)
        self.trigger_generator = self.ctx.seeds.generator(self.ctx.seeds.trigger)

    def batch_loss(self, model: torch.nn.Module, batch: Batch) -> Tuple[torch.Tensor, LossParts]:
        """Pha mô hình: trigger cố định (detach), mẫu bị chọn được gắn trigger và sửa nhãn (Công thức 11-14)."""
        is_poison = self.flags[batch.idx].to(self.ctx.device)
        x, y_poisoned = batch.x, batch.y
        if bool(is_poison.any()):
            x = x.clone()
            x[is_poison] = self._trigger.inject(x[is_poison], detach=True)
            y_poisoned = batch.y.clone()
            y_poisoned[is_poison] = self._trigger.poison_labels(batch.y[is_poison])

        return self.model_objective(model(x, batch.m), batch.y, y_poisoned, is_poison)

    def after_model_epoch(self, epoch: int, model: torch.nn.Module) -> Optional[Dict[str, float]]:
        if (epoch + 1) % self.config.attack.trigger_update_every != 0 or len(self.poison_pos) == 0:
            return None

        return self._update_trigger(model)

    def _update_trigger(self, model: torch.nn.Module) -> Dict[str, float]:
        """Pha trigger: cố định mô hình (eval, không cập nhật tham số), duyệt các mẫu bị chọn."""
        att = self.config.attack
        ctx = self.ctx
        model.eval()
        saved = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)

        before = params_digest(model)
        sums: Dict[str, float] = {}
        n_seen = 0
        for _ in range(att.trigger_passes_per_update):
            perm = self.poison_pos[torch.randperm(len(self.poison_pos), generator=self.trigger_generator)]
            for s in range(0, len(perm), att.trigger_batch_size):
                idx = perm[s:s + att.trigger_batch_size]
                x = ctx.train.x[idx].to(ctx.device)
                y = ctx.train.y[idx].to(ctx.device)
                m = ctx.train.m[idx].to(ctx.device)
                pred = model(self._trigger.inject(x), m)
                loss, parts = self.trigger_objective(pred, self._trigger.poison_labels(y), y, self._trigger)
                self.trigger_optimizer.zero_grad()
                loss.backward()
                self.trigger_optimizer.step()

                n_seen += len(idx)
                _accumulate(sums, "total", loss, len(idx))
                for key, value in parts.items():
                    _accumulate(sums, key, value, len(idx))

        for p, flag in zip(model.parameters(), saved):
            p.requires_grad_(flag)

        if params_digest(model) != before:
            raise IntegrityError("Tham số mô hình thay đổi trong giai đoạn cập nhật trigger.")

        return {k: v / max(n_seen, 1) for k, v in sums.items()}

    def make_selector(self) -> StateSelector:
        return CleanConstrainedAttackSelector(self.config.attack.state_selection.clean_tolerance)

    def finalize(self, results: Dict[str, Any]) -> None:
        """Đối chiếu phép gắn torch với bản numpy tham chiếu, rồi đo mức sửa đầu vào trên test."""
        test = self.ctx.test
        tau = self._trigger.numpy()
        x_test = test.x.numpy().astype(np.float64)
        x_trig_ref = ReferenceInjector(self.plan).inject(x_test, tau)
        x_trig_torch = self._trigger.inject(test.x.to(self.ctx.device), detach=True).cpu().numpy()
        if not np.allclose(x_trig_ref, x_trig_torch, atol=1e-5):
            raise IntegrityError("Gắn trigger bằng torch và numpy không khớp.")

        results["input_change_test"] = MetricCalculator.input_change(x_test, x_trig_ref, self.plan)
        results["trigger_final"] = tau.tolist()

    # ----------------------------------------------------------------------------- trigger

    @property
    def trigger(self) -> Optional[Trigger]:
        return self._trigger

    @property
    def attack_target(self) -> Optional[AttackTarget]:
        return None if self.plan is None else AttackTarget.of(self.plan)

    def plan_dict(self) -> Optional[Dict[str, Any]]:
        return None if self.plan is None else self.plan.to_dict()

    def epoch_summary(self, val: Metrics) -> str:
        return (f"{super().epoch_summary(val)} MAE_pa {val['mae_pa_norm']:.4f} "
                f"MAE_pn {val.get('mae_pn_norm', float('nan')):.4f}")


def _accumulate(sums: Dict[str, float], key: str, value: torch.Tensor, n: int) -> None:
    sums[key] = sums.get(key, 0.0) + float(value) * n
