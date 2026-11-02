"""Phân chia, cửa sổ, chuẩn hóa, nguồn dữ liệu (docs/03, docs/06)."""

import os
import shutil
import tempfile

import numpy as np
import pytest

from badtime_repro.catalog import CATALOG
from badtime_repro.common.errors import DataUnavailableError
from badtime_repro.data.pipeline import DataPipeline
from badtime_repro.data.scaling import StandardScaler
from badtime_repro.data.splits import RatioSplit, TslibEttHourSplit
from badtime_repro.data.windows import WindowSpec
from helpers import ETTH1_CSV, HAS_ETTH1, ROOT, default_config, synthetic_dataset


def test_split_ratio_boundaries():
    assert RatioSplit().boundaries(17420).to_dict() == {
        "train": [0, 10452], "val": [10452, 13936], "test": [13936, 17420]}
    assert TslibEttHourSplit().boundaries(17420).to_dict() == {
        "train": [0, 8640], "val": [8640, 11520], "test": [11520, 14400]}
    with pytest.raises(DataUnavailableError):
        TslibEttHourSplit().boundaries(1000)


def test_window_boundaries_small_example():
    bounds = RatioSplit().boundaries(20)        # train [0,12) val [12,16) test [16,20)
    w = WindowSpec(3, 2)
    assert w.starts(bounds, "train", True).tolist() == list(range(0, 8))    # s + 3 + 2 <= 12
    assert w.starts(bounds, "val", True).tolist() == [9, 10, 11]            # nhãn [12,14) ... [14,16)
    assert w.starts(bounds, "val", False).tolist() == []


def test_windows_exact_rows_and_labels_inside_partition():
    ds = synthetic_dataset()
    x, y, _ = ds.windows("val")
    s = ds.starts["val"][0]
    assert np.allclose(x[0], ds.norm[s:s + 4])
    assert np.allclose(y[0], ds.norm[s + 4:s + 7])
    v_s, v_e = ds.bounds.val
    for st in ds.starts["val"]:
        assert v_s <= st + 4 and st + 7 <= v_e


def test_scaler_uses_train_only():
    ds = synthetic_dataset()
    raw = ds.raw.copy()
    raw[30:] += 1000.0
    b_s, b_e = ds.bounds.train
    sc = StandardScaler.fit(raw[b_s:b_e])
    assert np.allclose(sc.mean, ds.scaler.mean) and np.allclose(sc.std, ds.scaler.std)


def test_scaler_rejects_constant_channel_and_round_trips():
    with pytest.raises(DataUnavailableError):
        StandardScaler.fit(np.ones((10, 2)))

    sc = StandardScaler([1.0, 2.0], [3.0, 4.0])
    v = np.array([[0.5, -1.0]])
    assert np.allclose(sc.inverse(sc.transform(v)), v)
    assert StandardScaler.from_dict(sc.to_dict()).to_dict() == sc.to_dict()


def test_windows_are_independent_copies():
    ds = synthetic_dataset()
    x, _, _ = ds.windows("train")
    before = ds.norm.copy()
    x[0, -1, 0] = 999.0
    assert np.array_equal(ds.norm, before)
    assert x[1, -2, 0] != 999.0          # cửa sổ kế tiếp dùng chung hàng đó không bị đổi


def test_invalid_csv_is_reported_not_fixed():
    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, "dataset", "ETTh1")
        os.makedirs(d)
        with open(os.path.join(d, "ETTh1.csv"), "w") as f:
            f.write("date,a\n2016-07-01 00:00:00,1\n")

        with pytest.raises(DataUnavailableError) as exc:
            DataPipeline(tmp).prepare(default_config())

        assert "Cột không khớp" in str(exc.value)
    finally:
        shutil.rmtree(tmp)


@pytest.mark.skipif(not HAS_ETTH1, reason="chưa có dataset/ETTh1/ETTh1.csv")
def test_etth1_profile():
    ds = DataPipeline(ROOT).prepare(default_config())
    assert ds.report.n_rows == 17420 and ds.report.n_missing == 0
    assert ds.report.channels == list(CATALOG.datasets["ETTh1"].expected_columns[1:])
    assert {k: len(v) for k, v in ds.starts.items()} == {"train": 10261, "val": 3389, "test": 3389}
    assert ds.report.path == os.path.abspath(ETTH1_CSV)
