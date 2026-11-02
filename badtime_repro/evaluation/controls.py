"""Đối chứng bốn điều kiện: {mô hình sạch, mô hình bị tấn công} x {đầu vào sạch, đầu vào có trigger}.

Cùng một trigger đã khóa, cùng cửa sổ, cùng mục tiêu cho cả hai mô hình (docs/03 mục 'Đánh giá theo cặp').
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from ..attack.plan import AttackPlan
from ..common.errors import ConfigInvalidError, DataUnavailableError
from ..common.jsonio import write_json
from ..experiment import Experiment
from ..metrics import AttackTarget, MetricCalculator
from ..runs.store import RunFiles, RunReader

log = logging.getLogger(__name__)

CONDITIONS = (
    ("clean_model", "clean_input"),
    ("clean_model", "trigger_input"),
    ("attacked_model", "clean_input"),
    ("attacked_model", "trigger_input"),
)


class ControlExperiment:
    def __init__(self, project_root: str):
        self.experiment = Experiment(project_root)

    def run(self, clean_run: str, attack_run: str) -> Dict[str, Any]:
        import torch

        from ..attack.trigger import Trigger
        from ..models import build_model
        from ..training.context import resolve_device
        from ..training.inference import Predictor
        from ..training.tensors import SplitTensors

        clean, attack = RunReader(clean_run), RunReader(attack_run)
        self._check_compatible(clean, attack)

        cfg, dataset = self.experiment.resolve(attack.config())
        if dataset.report.sha256 != attack.profile()["data"]["report"]["sha256"]:
            raise DataUnavailableError("Dữ liệu hiện tại khác dữ liệu lúc huấn luyện (sha256).")

        device = resolve_device(cfg.train.device)
        pairs = {"clean_model": clean.state_pair(), "attacked_model": attack.state_pair()}
        readers = {"clean_model": clean, "attacked_model": attack}
        models = {}
        for name, pair in pairs.items():
            model = build_model(readers[name].config(), dataset.n_channels).to(device)
            model.load_state_dict(pair.model_state)
            models[name] = model

        att_pair = pairs["attacked_model"]
        plan = AttackPlan.from_dict(att_pair.plan)
        trigger = Trigger(plan, init=att_pair.tau.numpy()).to(device)
        target = AttackTarget.of(plan)
        predictor = Predictor(device, cfg.train.eval_batch_size)
        calculator = MetricCalculator(dataset.scaler)

        out: Dict[str, Any] = {
            "clean_run": clean.path,
            "attack_run": attack.path,
            "attack_epoch": att_pair.epoch,
            "note": "Cùng trigger đã khóa, cùng cửa sổ, cùng mục tiêu cho cả hai mô hình.",
        }
        with torch.no_grad():
            for split_name in ("val", "test"):
                split = SplitTensors.from_dataset(dataset, split_name)
                y = split.y_numpy()
                res = {}
                for model_name, input_name in CONDITIONS:
                    use_trigger = trigger if input_name == "trigger_input" else None
                    pred = predictor.predict(models[model_name], split, use_trigger)
                    res[f"{model_name}|{input_name}"] = calculator.forecast(pred, y, pred, target)

                out[split_name] = res

        out_path = attack.file(RunFiles.controls(clean.name))
        write_json(out_path, out)
        log.info("Đã ghi %s", out_path)
        for cond, met in out["test"].items():
            log.info("test %-32s MAE_all %.4f  MAE_pa %.4f  MAE_pn %.4f",
                     cond, met["mae_c_norm"], met["mae_pa_norm"], met["mae_pn_norm"])

        return out

    @staticmethod
    def _check_compatible(clean: RunReader, attack: RunReader) -> None:
        p_clean, p_att = clean.profile(), attack.profile()
        if p_clean["method"] != "clean":
            raise ConfigInvalidError(f"--clean-run phải là lần chạy clean, nhận '{p_clean['method']}'.")

        if not attack.has(RunFiles.PLAN):
            raise ConfigInvalidError(f"--attack-run phải là lần chạy có kế hoạch tác động, nhận '{p_att['method']}'.")

        for reader in (clean, attack):
            if not reader.is_completed():
                raise ConfigInvalidError(f"Lần chạy {reader.path} chưa hoàn thành.")

        pairs: Tuple[Tuple[str, Any, Any], ...] = (
            ("sha256", p_clean["data"]["report"]["sha256"], p_att["data"]["report"]["sha256"]),
            ("n_rows", p_clean["data"]["report"]["n_rows"], p_att["data"]["report"]["n_rows"]),
            ("bounds", p_clean["data"]["bounds"], p_att["data"]["bounds"]),
            ("timesnet", p_clean["config"]["timesnet"], p_att["config"]["timesnet"]),
            ("seq_len", p_clean["config"]["seq_len"], p_att["config"]["seq_len"]),
            ("pred_len", p_clean["config"]["pred_len"], p_att["config"]["pred_len"]),
            ("eval_batch_size", p_clean["config"]["train"]["eval_batch_size"],
             p_att["config"]["train"]["eval_batch_size"]),
        )
        for key, a, b in pairs:
            if a != b:
                raise ConfigInvalidError(f"Hai lần chạy khác nhau ở '{key}'; không đối chứng được.")
