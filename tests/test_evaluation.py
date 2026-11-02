"""Phép đo, thư mục lần chạy, tính lại chỉ số và báo cáo (không cần torch)."""

import shutil
import tempfile

import numpy as np
import pytest

from badtime_repro.attack.injection import ReferenceInjector
from badtime_repro.common.errors import DataUnavailableError
from badtime_repro.common.jsonio import write_json
from badtime_repro.data.scaling import StandardScaler
from badtime_repro.metrics import AttackTarget, MetricCalculator
from badtime_repro.evaluation.recompute import MetricRecomputer
from badtime_repro.evaluation.report import MarkdownReportRenderer, ReportBuilder
from badtime_repro.runs.store import RunFiles, RunIndex, RunStore
from helpers import default_config, etth1_toy_plan, toy_plan


def test_metrics_hand_example():
    calc = MetricCalculator(StandardScaler([0.0, 10.0], [1.0, 2.0]))
    y = np.zeros((1, 2, 2))
    pc = np.array([[[1.0, 0.0], [0.0, -1.0]]])
    pt = np.array([[[3.0, 2.0], [1.0, 2.0]]])
    m = calc.forecast(pc, y, pt, AttackTarget((1,), np.array([[1.0]])))   # t_pin = 1, kênh đích 1
    assert m["mae_c_norm"] == pytest.approx(0.5)
    assert m["mae_c_orig"] == pytest.approx((1 + 0 + 0 + 2) / 4)
    assert m["mae_pa_norm"] == pytest.approx(1.0)       # |2 - 1|
    assert m["mae_pa_orig"] == pytest.approx(2.0)
    assert m["mae_pn_norm"] == pytest.approx(2.0)       # kênh 0: |3|, |1|
    assert m["mae_rest_norm"] == pytest.approx(2.0)     # kênh 1, bước 1: |2|


def test_input_change_counts():
    plan = toy_plan()
    x = np.zeros((4, 10, 3))
    st = MetricCalculator.input_change(x, ReferenceInjector(plan).inject(x, plan.init_trigger), plan)
    assert st["changed_cells_per_window_mean"] == 5
    assert st["registered_cells_per_window"] == 5


def _fake_run(root, method, seed, rng, plan=None):
    """Một lần chạy giả hoàn chỉnh: profile, (plan), preds, metrics đúng như Trainer ghi."""
    cfg = default_config(f"method={method}")
    store = RunStore.create(root, cfg, method, seed)
    store.mark_running(method, seed)
    scaler = StandardScaler(rng.standard_normal(7), rng.random(7) + 0.5)
    store.write_json(RunFiles.PROFILE, {"method": method, "seed": seed, "config": cfg.to_dict(),
                                        "data": {"scaler": scaler.to_dict()}})
    calc = MetricCalculator(scaler)
    target = None
    if plan is not None:
        store.write_json(RunFiles.PLAN, plan.to_dict())
        target = AttackTarget.of(plan)

    metrics = {}
    for split in ("val", "test"):
        y = rng.standard_normal((20, 96, 7)).astype(np.float32)
        pc = rng.standard_normal((20, 96, 7)).astype(np.float32)
        pt = rng.standard_normal((20, 96, 7)).astype(np.float32) if plan is not None else None
        metrics[split] = calc.forecast(pc.astype(np.float64), y.astype(np.float64),
                                       None if pt is None else pt.astype(np.float64), target)
        arrays = {"pred_clean": pc, "y": y, "starts": np.arange(20)}
        if pt is not None:
            arrays["pred_trig"] = pt

        store.save_predictions(split, **arrays)

    store.write_json(RunFiles.METRICS, metrics)
    store.mark_completed(1)
    return store


def test_run_store_recompute_and_report():
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(0)
        plan = etth1_toy_plan()
        clean = _fake_run(tmp, "clean", 1, rng)
        bad = _fake_run(tmp, "badtime", 1, rng, plan)
        _fake_run(tmp, "badtime", 2, rng, plan)

        for run in (clean, bad):
            res = MetricRecomputer().run(run.path)
            assert all(v["ok"] for v in res.values()), res

        slot_dir = f"{tmp}/runs/ETTh1_TimesNet_96_96"
        index = RunIndex(slot_dir)
        assert len(index.completed("badtime")) == 2
        assert index.latest("badtime", 1).path == bad.path

        ctl = {s: {"clean_model|trigger_input": {"mae_pa_norm": 0.8, "mae_pn_norm": 0.43}} for s in ("val", "test")}
        write_json(bad.file(RunFiles.controls(clean.name)), ctl)

        builder = ReportBuilder(slot_dir, "ETTh1_TimesNet_96_96")
        report = builder.build()
        row = report["rows"]["test"]
        assert row["BadTime"]["n_runs"] == 2 and row["BadTime"]["mae_pa"]["paper"] == 0.376
        assert row["Benign"]["mae_pa"]["runs"] == [0.8]
        assert row["Benign"]["mae_pa"]["warn_over_5pct"] is False       # |0.8 - 0.781| / 0.781 < 5%
        text = MarkdownReportRenderer().render(report)
        assert "| BadTime | mae_pa | 0.376 |" in text
        builder.write(report)
    finally:
        shutil.rmtree(tmp)


def test_report_without_runs_is_data_error():
    tmp = tempfile.mkdtemp()
    try:
        with pytest.raises(DataUnavailableError):
            ReportBuilder(f"{tmp}/runs/x", "x").build()
    finally:
        shutil.rmtree(tmp)


def test_failed_run_is_not_listed():
    tmp = tempfile.mkdtemp()
    try:
        store = RunStore.create(tmp, default_config(), "clean", 5)
        store.mark_running("clean", 5)
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            store.mark_failed(exc)

        assert store.status()["state"] == "failed" and "boom" in store.status()["error"]
        assert RunIndex(f"{tmp}/runs/ETTh1_TimesNet_96_96").completed("clean") == []
    finally:
        shutil.rmtree(tmp)
