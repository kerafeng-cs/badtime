"""Tính lại chỉ số từ dự báo đã lưu và so với metrics.json (docs/06: kiểm tra tính lại chỉ số)."""

from __future__ import annotations

from typing import Any, Dict

from ..data.scaling import StandardScaler
from ..metrics import AttackTarget, MetricCalculator
from ..runs.store import RunReader


class MetricRecomputer:
    def __init__(self, tolerance: float = 1e-9):
        self.tolerance = tolerance

    def run(self, run_dir: str) -> Dict[str, Dict[str, Any]]:
        reader = RunReader(run_dir)
        saved = reader.metrics()
        calculator = MetricCalculator(StandardScaler.from_dict(reader.profile()["data"]["scaler"]))
        plan = reader.plan()
        target = None if plan is None else AttackTarget.of(plan)

        report = {}
        for split in ("val", "test"):
            z = reader.predictions(split)
            pred_trig = z["pred_trig"].astype("float64") if target is not None else None
            met = calculator.forecast(z["pred_clean"].astype("float64"), z["y"].astype("float64"), pred_trig, target)
            missing = sorted(set(met) - set(saved[split]))
            diffs = {k: abs(met[k] - saved[split][k]) for k in met if k in saved[split]}
            worst = max(diffs.values()) if diffs else float("inf")
            report[split] = {
                "max_abs_diff": worst,
                "n_metrics": len(diffs),
                "missing_in_saved": missing,
                "ok": bool(diffs) and not missing and worst <= self.tolerance,
            }

        return report
