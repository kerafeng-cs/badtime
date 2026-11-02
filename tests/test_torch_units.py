"""Thành phần torch, không huấn luyện thật. Bỏ qua nếu chưa cài torch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from badtime_repro.attack.gat import GatInfluenceScorer, GraphAttentionNextStep  # noqa: E402
from badtime_repro.attack.injection import ReferenceInjector  # noqa: E402
from badtime_repro.attack.objectives import (  # noqa: E402
    BackdoorModelObjective,
    CleanObjective,
    SquaredPlusAbsolute,
    TriggerObjective,
)
from badtime_repro.attack.trigger import Trigger  # noqa: E402
from badtime_repro.catalog import CATALOG  # noqa: E402
from badtime_repro.config import GatConfig  # noqa: E402
from badtime_repro.models import MODEL_FACTORIES, build_model  # noqa: E402
from badtime_repro.training.methods import TRAINING_METHODS  # noqa: E402
from badtime_repro.training.selection import (  # noqa: E402
    CleanConstrainedAttackSelector,
    EarlyStoppingSelector,
    StateSnapshot,
)
from helpers import default_config, etth1_toy_plan  # noqa: E402


def test_registries_resolve():
    for name in TRAINING_METHODS:
        assert TRAINING_METHODS.resolve(name).name == name

    for name in MODEL_FACTORIES:
        assert CATALOG.models[name].status.runnable
        MODEL_FACTORIES.resolve(name)


def test_clean_loss_names_match_config():
    from badtime_repro.support import CLEAN_LOSSES

    assert set(CLEAN_LOSSES) == set(CleanObjective.KINDS)
    with pytest.raises(ValueError):
        CleanObjective("l1", "sum")


def test_inject_matches_reference_and_keeps_input():
    plan = etth1_toy_plan(lam=0.7)
    x = np.random.default_rng(0).standard_normal((4, 96, 7))
    trig = Trigger(plan)
    xt = torch.as_tensor(x, dtype=torch.float32)
    before = xt.clone()
    out = trig.inject(xt).detach().numpy()
    assert torch.equal(xt, before)
    ref = ReferenceInjector(plan).inject(x.astype(np.float32).astype(np.float64), plan.init_trigger.astype(np.float32))
    assert np.allclose(out, ref, atol=1e-6)


def test_disabled_interpolation_is_identity():
    plan = etth1_toy_plan(lam=0.0)
    x = torch.randn(3, 96, 7)
    assert torch.equal(Trigger(plan).inject(x), x)


def test_poison_labels_torch():
    plan = etth1_toy_plan()
    y = torch.zeros(2, 96, 7)
    yp = Trigger(plan).poison_labels(y)
    assert torch.all(yp[:, :53, 6] == 1) and torch.all(yp[:, 53:, 6] == 0) and torch.all(yp[:, :, :6] == 0)
    assert torch.all(y == 0)


def test_objectives_and_empty_groups():
    pred, y = torch.randn(4, 96, 7), torch.randn(4, 96, 7)
    obj = BackdoorModelObjective(7, [6], (1, 1, 1), "sum")
    for flags in (torch.zeros(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool)):
        loss, _ = obj(pred, y, y, flags)
        assert torch.isfinite(loss)

    _, parts = obj(pred, y, y, torch.zeros(4, dtype=torch.bool))
    assert set(parts) == {"clean"}

    # 'sum' = tổng phần tử / số mẫu; 'mean' = trung bình phần tử
    d = pred - y
    assert float(SquaredPlusAbsolute("sum")(pred, y)) == pytest.approx(float((d.pow(2).sum() + d.abs().sum()) / 4))
    assert float(SquaredPlusAbsolute("mean")(pred, y)) == pytest.approx(float(d.pow(2).mean() + d.abs().mean()))

    # Nhánh sạch 'mse_mae' và BadTime khi lô không có mẫu bị chọn cho cùng giá trị loss
    clean_loss, _ = CleanObjective("mse_mae", "sum")(pred, y)
    bad_loss, _ = obj(pred, y, y, torch.zeros(4, dtype=torch.bool))
    assert torch.equal(clean_loss, bad_loss)


def test_smooth_and_range_loss():
    trig = Trigger(etth1_toy_plan(), init=np.zeros(29))
    assert float(trig.smooth_loss(0.8)) == 0.0 and float(trig.range_loss()) == 0.0
    with torch.no_grad():
        trig.tau[0] = 5.0                  # vượt max 3 và nhảy 5 > 0.8
    assert float(trig.range_loss()) == pytest.approx(2.0 / 29)
    assert float(trig.smooth_loss(0.8)) == pytest.approx((5.0 - 0.8) ** 2 / 29)


def _tiny_model():
    torch.manual_seed(0)
    return build_model(default_config("timesnet.d_model=8", "timesnet.d_ff=8", "timesnet.e_layers=1"), 7)


def test_model_shape_and_no_decoder_input():
    out = _tiny_model()(torch.randn(5, 96, 7), torch.randn(5, 96, 4))
    assert out.shape == (5, 96, 7)


def test_trigger_gradient_with_frozen_model():
    model = _tiny_model()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    trig = Trigger(etth1_toy_plan())
    x, y = torch.randn(8, 96, 7), torch.randn(8, 96, 7)
    pred = model(trig.inject(x), torch.zeros(8, 96, 4))
    loss, _ = TriggerObjective(7, [6], (1, 1, 1, 1), "sum", 0.8)(pred, trig.poison_labels(y), y, trig)
    loss.backward()
    assert trig.tau.grad is not None and torch.any(trig.tau.grad != 0)
    assert all(p.grad is None for p in model.parameters())


def test_gat_attention_and_influence():
    net = GraphAttentionNextStep(96, 8, 4, 0.2, True)
    alpha, _ = net.attention(torch.randn(3, 7, 96))
    assert alpha.shape == (3, 4, 7, 7)
    assert torch.allclose(alpha.sum(-1), torch.ones(3, 4, 7))

    series = np.random.default_rng(0).standard_normal((400, 7))
    cfg = GatConfig(hidden_dim=8, heads=4, epochs=1, batch_size=64)
    res = GatInfluenceScorer(cfg, torch.device("cpu")).score(series, 96, [6], seed=0)
    assert res.scores.shape == (7,) and abs(res.scores.sum() - 1.0) < 1e-5   # một kênh đích: tổng attention = 1
    assert len(res.history) == 1


def _snap(epoch, mae_c, mae_pa, mse):
    return StateSnapshot(epoch, {"mae_c_norm": mae_c, "mae_pa_norm": mae_pa}, mse, {}, None)


def test_state_selectors():
    es = EarlyStoppingSelector(patience=2)
    stops = [es.observe(_snap(e, 0, 0, mse)) for e, mse in enumerate([0.5, 0.4, 0.45, 0.46], start=1)]
    assert stops == [False, False, False, True] and es.select()[0].epoch == 2

    cc = CleanConstrainedAttackSelector(tolerance=0.05)
    for e, (c, pa) in enumerate([(0.40, 0.9), (0.41, 0.5), (0.45, 0.3)], start=1):
        assert cc.observe(_snap(e, c, pa, 0)) is False

    chosen, info = cc.select()
    assert chosen.epoch == 2 and info["feasible_epochs"] == [1, 2]     # 0.45 > 1.05 * 0.40
