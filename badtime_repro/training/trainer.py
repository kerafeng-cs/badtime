"""Vòng huấn luyện chung cho mọi TrainingMethod.

Thứ tự dùng RNG được cố định để nhánh sạch và BadTime đã vô hiệu hóa (0% mẫu) có cùng quỹ đạo huấn luyện:
    method.prepare()  (RNG riêng của khâu lập kế hoạch)
    seed_everything(seed) -> khởi tạo mô hình
    torch.Generator(seed) -> thứ tự lô
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from ..assumptions import ASSUMPTIONS
from ..catalog import CATALOG
from ..common.errors import IntegrityError
from ..models import build_model
from ..runs.environment import peak_memory_mb, software_versions
from ..runs.store import RunFiles, StatePair
from .context import RunContext, seed_everything
from .inference import Evaluator, Predictor
from .methods.base import TrainingMethod
from .schedule import SCHEDULES
from .selection import StateSnapshot
from .tensors import Batch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingOutcome:
    selected_epoch: int
    metrics: Dict[str, Any]


class CostClock:
    """Cộng dồn thời gian theo hạng mục."""

    def __init__(self) -> None:
        self.costs: Dict[str, float] = {}
        self._t0 = time.time()

    def add(self, key: str, since: float) -> None:
        self.costs[key] = self.costs.get(key, 0.0) + (time.time() - since)

    def total(self) -> float:
        return time.time() - self._t0


class Trainer:
    def __init__(self, ctx: RunContext, method: TrainingMethod):
        self.ctx = ctx
        self.method = method
        self.config = ctx.config
        self.clock = CostClock()
        self.evaluator = Evaluator(ctx.dataset.scaler, Predictor(ctx.device, self.config.train.eval_batch_size))
        self.history: List[Dict[str, Any]] = []

    # ----------------------------------------------------------------------------- điều phối

    def run(self) -> TrainingOutcome:
        ctx, method, tr = self.ctx, self.method, self.config.train
        ctx.store.write_json(RunFiles.PROFILE, self._profile())

        t = time.time()
        method.prepare()
        self.clock.add("planning_sec", t)

        seed_everything(ctx.seeds.model)
        model = build_model(self.config, ctx.dataset.n_channels).to(ctx.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=tr.learning_rate)
        schedule = SCHEDULES[tr.lradj](tr.learning_rate)
        batch_order = ctx.seeds.generator(ctx.seeds.model)
        method.bind(model)
        selector = method.make_selector()

        for epoch in range(tr.epochs):
            snapshot = self._run_epoch(epoch, model, optimizer, schedule.lr(epoch), batch_order)
            if selector.observe(snapshot):
                log.info("Dừng sớm.")
                break

        chosen, selection = selector.select()
        model.load_state_dict(chosen.model_state)
        method.load_trigger_state(chosen.tau)
        ctx.store.save_state_pair(StatePair(
            epoch=chosen.epoch, model_state=chosen.model_state, tau=chosen.tau, plan=method.plan_dict(),
            method=method.name, seed=ctx.seed,
        ))

        results = self._final_evaluation(model, selection)
        log.info("Chọn epoch %d. Test: %s", chosen.epoch, ", ".join(
            f"{k}={v:.4f}" for k, v in results["test"].items() if k.startswith("mae") and k.endswith("norm")
        ))
        return TrainingOutcome(chosen.epoch, results)

    # ----------------------------------------------------------------------------- một epoch

    def _run_epoch(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   lr: float, batch_order: torch.Generator) -> StateSnapshot:
        ctx, method = self.ctx, self.method
        for group in optimizer.param_groups:
            group["lr"] = lr

        rec: Dict[str, Any] = {"epoch": epoch + 1, "lr": lr}

        t = time.time()
        tau_before = method.trigger_state()
        rec["train"] = self._model_phase(model, optimizer, batch_order)
        tau_after = method.trigger_state()
        if tau_before is not None and not torch.equal(tau_before, tau_after):
            raise IntegrityError("Trigger thay đổi trong giai đoạn cập nhật mô hình.")

        self.clock.add("model_train_sec", t)

        t = time.time()
        extra = method.after_model_epoch(epoch, model)
        if extra is not None:
            rec["trigger_update"] = extra
            self.clock.add("trigger_update_sec", t)

        t = time.time()
        val = self.evaluator.evaluate(model, ctx.val, method.trigger, method.attack_target)
        rec["val"] = val.metrics
        rec["val_loss_mse"] = float(np.mean((val.pred_clean - ctx.val.y_numpy()) ** 2))
        self.clock.add("eval_sec", t)

        log.info("epoch %d: train %.4f | %s", epoch + 1, rec["train"]["total"], method.epoch_summary(val.metrics))
        self.history.append(rec)
        ctx.store.write_json(RunFiles.HISTORY, self.history)

        return StateSnapshot(
            epoch=epoch + 1,
            val=val.metrics,
            val_loss_mse=rec["val_loss_mse"],
            model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            tau=method.trigger_state(),
        )

    def _model_phase(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                     batch_order: torch.Generator) -> Dict[str, float]:
        ctx = self.ctx
        train = ctx.train
        bs = self.config.train.batch_size
        model.train()
        perm = torch.randperm(len(train), generator=batch_order)
        sums: Dict[str, float] = {}
        n_seen = 0
        for s in range(0, len(train), bs):
            idx = perm[s:s + bs]
            batch = Batch(train.x[idx].to(ctx.device), train.y[idx].to(ctx.device), train.m[idx].to(ctx.device), idx)
            loss, parts = self.method.batch_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_seen += len(idx)
            sums["total"] = sums.get("total", 0.0) + float(loss) * len(idx)
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value) * len(idx)

        return {k: v / n_seen for k, v in sums.items()}

    # ----------------------------------------------------------------------------- kết thúc

    def _final_evaluation(self, model: torch.nn.Module, selection: Dict[str, Any]) -> Dict[str, Any]:
        ctx, method = self.ctx, self.method
        t = time.time()
        results: Dict[str, Any] = {"selection": selection}
        for split in (ctx.val, ctx.test):
            res = self.evaluator.evaluate(model, split, method.trigger, method.attack_target)
            results[split.name] = res.metrics
            arrays = {"pred_clean": res.pred_clean.astype(np.float32), "y": split.y.numpy(), "starts": split.starts}
            if res.pred_trig is not None:
                arrays["pred_trig"] = res.pred_trig.astype(np.float32)

            ctx.store.save_predictions(split.name, **arrays)

        method.finalize(results)
        self.clock.add("final_eval_sec", t)

        costs = dict(self.clock.costs)
        costs["total_sec"] = self.clock.total()
        costs["peak_memory_mb"] = peak_memory_mb(ctx.device)
        results["costs"] = costs
        ctx.store.write_json(RunFiles.METRICS, results)
        return results

    def _profile(self) -> Dict[str, Any]:
        ctx = self.ctx
        return {
            "method": self.method.name,
            "seed": ctx.seed,
            "device": str(ctx.device),
            "config": self.config.to_dict(),
            "data": ctx.dataset.profile(),
            "model_source": CATALOG.models[self.config.model].source,
            "versions": software_versions(),
            "assumptions": [a.to_dict() for a in ASSUMPTIONS],
        }
