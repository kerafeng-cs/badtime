"""Các khối thiết kế BadTime, phần numpy (Thuật toán 1-4, Công thức 12-14)."""

import numpy as np
import pytest

from badtime_repro.attack.budget import BUDGET_PROTOCOLS, get_budget_protocol
from badtime_repro.attack.channels import FixedInfluenceScorer, PoisonChannelSelector, TargetChannelResolver
from badtime_repro.attack.injection import ReferenceInjector
from badtime_repro.attack.lag import LagEstimator
from badtime_repro.attack.patterns import get_pattern
from badtime_repro.attack.plan import AttackPlan, AttackPlanner
from badtime_repro.attack.sample_selection import HybridSampleSelector
from badtime_repro.common.errors import PlanError
from badtime_repro.support import SupportChecker
from badtime_repro.data.pipeline import DataPipeline
from helpers import HAS_ETTH1, ROOT, default_config, toy_plan

INFLUENCE = [0.1, 0.2, 0.05, 0.3, 0.1, 0.1, 0.15]


# ------------------------------------------------------------------ ngân sách, mục tiêu, kênh

def test_budget_protocols():
    assert get_budget_protocol("keep_total").split(29, 2) == [15, 14]
    assert get_budget_protocol("equal_segments").split(29, 2) == [14, 14]
    assert get_budget_protocol("ceil_segments").split(29, 2) == [15, 15]
    for protocol in BUDGET_PROTOCOLS.values():
        with pytest.raises(PlanError):
            protocol.split(1, 2)


def test_patterns():
    p = get_pattern("hump_backtime_cone").build(53, 1.0, 0.0, 1)
    assert p.shape == (53, 1)
    assert p[0, 0] == pytest.approx(0.1278) and p[-1, 0] == pytest.approx(0.1278)
    assert p[26, 0] == pytest.approx(0.3853)
    q = get_pattern("hump_sine").build(53, 2.0, 0.5, 2)
    assert q.shape == (53, 2) and q.max() <= 2.5 and q.min() > 0.5
    with pytest.raises(PlanError):
        get_pattern("star")


def test_channel_resolution_and_selection():
    idx, note = TargetChannelResolver(["OT"], 0.1).resolve(["a", "b", "c", "d", "e", "f", "OT"])
    assert idx == [6] and note is None
    with pytest.raises(PlanError):
        TargetChannelResolver(["XX"], 0.1).resolve(["a", "OT"])

    sel = PoisonChannelSelector(0.3, exclude_targets=False)
    assert sel.count(7) == 2
    assert sel.select(INFLUENCE, [6]) == [3, 1]
    assert PoisonChannelSelector(0.3, exclude_targets=True).select([0, 0, 0, 0, 0, 0, 1.0], [6]) == [0, 1]


# ------------------------------------------------------------------ Thuật toán 1

def test_sample_selection_groups():
    d = np.array([5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.0])
    sel = HybridSampleSelector(0.5, 0.4).select(d)      # k_sim = floor(2.0) = 2, k_diff = floor(3.0) = 3
    assert sel.info["k_similar"] == 2 and sel.info["k_different"] == 3
    assert set(sel.similar_pos.tolist()) == {9, 1}
    assert set(sel.different_pos.tolist()) == {2, 6, 4}
    assert len(sel.all_pos) == 5


# ------------------------------------------------------------------ Thuật toán 3

def test_lag_recovers_shift():
    rng = np.random.default_rng(0)
    n = 3000
    a = rng.standard_normal(n)
    series = np.zeros((n, 2))
    series[:, 0] = a
    series[20:, 1] = a[:-20]           # b(t) = a(t - 20): a dẫn trước 20 bước
    est = LagEstimator(96, 8).estimate(series, 0, [1], np.arange(0, n - 96), 10)
    assert est.lag == 20


def test_lag_skips_constant_segments():
    series = np.zeros((500, 2))
    series[:, 0] = np.arange(500)
    with pytest.raises(PlanError):
        LagEstimator(96, 8).estimate(series, 0, [1], np.arange(0, 300), 10)


# ------------------------------------------------------------------ Công thức 12-14

def test_inject_exact_cells_and_no_side_effects():
    plan = toy_plan()
    inj = ReferenceInjector(plan)
    x = np.zeros((2, 10, 3))
    x_t = inj.inject(x, plan.init_trigger)
    assert np.all(x == 0)
    assert inj.input_mask().sum() == 5
    assert np.array_equal(x_t[0] != 0, inj.input_mask())
    assert x_t[0, 6:9, 0].tolist() == [1.0, 2.0, 3.0]
    assert x_t[0, 3:5, 1].tolist() == [4.0, 5.0]


def test_inject_disabled_is_identity():
    plan = toy_plan(lam=0.0)
    x = np.random.default_rng(1).standard_normal((3, 10, 3))
    assert np.array_equal(ReferenceInjector(plan).inject(x, plan.init_trigger), x)


def test_poison_labels_only_target_segment():
    plan = toy_plan()
    y = np.zeros((2, 6, 3))
    yp = ReferenceInjector(plan).poison_labels(y)
    assert np.all(yp[:, :4, 2] == 5.0)
    assert np.all(yp[:, 4:, 2] == 0.0) and np.all(yp[:, :, :2] == 0.0)
    assert np.all(y == 0)


def test_plan_validation_and_round_trip():
    plan = toy_plan()
    again = AttackPlan.from_dict(plan.to_dict())
    assert again.to_dict() == plan.to_dict()
    plan.offsets = [9, 0]                 # đoạn [9, 12) vượt ra ngoài đầu vào [0, 10)
    with pytest.raises(PlanError):
        plan.validate()


# ------------------------------------------------------------------ dữ liệu thật

@pytest.mark.skipif(not HAS_ETTH1, reason="chưa có dataset/ETTh1/ETTh1.csv")
def test_real_plan_consistency():
    cfg = SupportChecker().check(default_config("method=badtime")).raise_if_blocked()
    ds = DataPipeline(ROOT).prepare(cfg)
    plan = AttackPlanner(cfg.attack, FixedInfluenceScorer(INFLUENCE)).plan(ds, seed=0)

    assert plan.target_idx == [6] and plan.poison_idx == [3, 1]
    assert plan.seg_lens == [15, 14] and plan.trigger_len == 29
    assert len(plan.poison_starts) == 1025                    # floor(0.03 x 10261) + floor(0.07 x 10261)
    assert len(plan.similar_starts) == 307 and len(plan.different_starts) == 718
    assert all(15 <= lag <= 96 for lag in plan.lags)          # [ceil(29/2), t_IN]
    assert set(plan.poison_starts.tolist()) <= set(ds.starts["train"].tolist())

    # Trigger khởi tạo = đúng đoạn đầu vào của cửa sổ có nhãn gần mục tiêu nhất (Công thức 9)
    x, _, _ = ds.windows("train", np.array([plan.init_start]))
    for (t_s, t_e), ch, off in plan.segments():
        assert np.allclose(plan.init_trigger[t_s:t_e], x[0, off:off + (t_e - t_s), ch])

    # Mọi dữ liệu dùng để lựa chọn thuộc phần huấn luyện
    b_e = ds.bounds.train[1]
    assert plan.poison_starts.max() + 96 + 96 <= b_e
    assert plan.init_start + 96 + 96 <= b_e
