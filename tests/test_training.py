"""Huấn luyện thật trên ETTh1 (CPU, mô hình nhỏ, 2 epoch). Chỉ chạy khi BADTIME_SLOW=1.

CPU để so sánh chính xác giữa hai nhánh (cuDNN có thể không tất định). Không dùng để so số với bài báo.
"""

import os
import shutil
import tempfile

import pytest

torch = pytest.importorskip("torch")

from badtime_repro.evaluation.controls import ControlExperiment  # noqa: E402
from badtime_repro.evaluation.recompute import MetricRecomputer  # noqa: E402
from badtime_repro.experiment import Experiment  # noqa: E402
from badtime_repro.runs.store import RunFiles  # noqa: E402
from helpers import HAS_ETTH1, ROOT, default_config  # noqa: E402

SLOW = os.environ.get("BADTIME_SLOW") == "1" and HAS_ETTH1

SMALL = [
    "train.epochs=2", "train.device=cpu", "attack.gat.epochs=1",
    "timesnet.d_model=8", "timesnet.d_ff=8", "timesnet.e_layers=1", "timesnet.top_k=3",
]


def _project_copy():
    tmp = tempfile.mkdtemp()
    shutil.copytree(os.path.join(ROOT, "dataset"), os.path.join(tmp, "dataset"))
    return tmp


@pytest.mark.skipif(not SLOW, reason="đặt BADTIME_SLOW=1 để chạy huấn luyện thật")
def test_pass_through_equals_clean_and_artifacts():
    """BadTime với 0% mẫu bị chọn phải cho cùng lịch sử huấn luyện với nhánh sạch cùng loss."""
    tmp = _project_copy()
    try:
        cfg = default_config(*SMALL, "train.clean_loss=mse_mae", "train.early_stopping_patience=99",
                             "attack.poison_ratio=0.0")
        exp = Experiment(tmp)
        clean = exp.train(cfg, "clean", 7)
        bad = exp.train(cfg, "badtime", 7)

        for a, b in zip(clean.history(), bad.history()):
            assert a["train"]["total"] == pytest.approx(b["train"]["total"], rel=1e-6)
            assert a["val"]["mae_c_norm"] == pytest.approx(b["val"]["mae_c_norm"], rel=1e-6)

        for run in (clean, bad):
            assert all(v["ok"] for v in MetricRecomputer().run(run.path).values())

        pair = bad.state_pair()
        assert pair.tau is not None and pair.plan is not None
        assert bad.metrics()["selection"]["chosen_epoch"] == pair.epoch
        assert bad.status()["state"] == "completed"
    finally:
        shutil.rmtree(tmp)


@pytest.mark.skipif(not SLOW, reason="đặt BADTIME_SLOW=1 để chạy huấn luyện thật")
def test_badtime_short_run_and_controls():
    tmp = _project_copy()
    try:
        cfg = default_config(*SMALL)
        exp = Experiment(tmp)
        clean = exp.train(cfg, "clean", 3)
        bad = exp.train(cfg, "badtime", 3)

        hist = bad.history()
        assert "trigger_update" in hist[1] and "trigger_update" not in hist[0]
        assert all(v["ok"] for v in MetricRecomputer().run(bad.path).values())
        assert bad.metrics()["input_change_test"]["changed_cells_per_window_mean"] <= 29
        assert bad.has(RunFiles.PLAN)

        ctl = ControlExperiment(tmp).run(clean.path, bad.path)
        assert set(ctl["test"]) == {
            "clean_model|clean_input", "clean_model|trigger_input",
            "attacked_model|clean_input", "attacked_model|trigger_input",
        }
        # Cùng mô hình sạch, đầu vào sạch -> khớp chỉ số đã lưu của nhánh sạch.
        saved = clean.metrics()["test"]["mae_c_norm"]
        assert ctl["test"]["clean_model|clean_input"]["mae_c_norm"] == pytest.approx(saved, rel=1e-9)
        assert bad.controls_files()
    finally:
        shutil.rmtree(tmp)
