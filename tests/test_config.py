"""Schema cấu hình, bộ nạp và kiểm tra hỗ trợ (docs/07)."""

import json
import os
import shutil
import tempfile

import pytest

from badtime_repro.catalog import CATALOG
from badtime_repro.common.errors import ConfigInvalidError, DataUnavailableError, UnsupportedSelectionError
from badtime_repro.config import ExperimentConfig, load_config
from badtime_repro.experiment import Experiment
from badtime_repro.support import SupportChecker, SupportVerdict
from helpers import ROOT, default_config


def check(*overrides):
    return SupportChecker().check(default_config(*overrides))


# ------------------------------------------------------------------ schema

def test_default_config_matches_shipped_file():
    with open(os.path.join(ROOT, "configs", "etth1_timesnet_96.json"), encoding="utf-8") as f:
        assert ExperimentConfig().to_dict() == json.load(f)


def test_round_trip():
    cfg = default_config("attack.pattern.amplitude=2", "train.epochs=3")
    again = ExperimentConfig.from_dict(cfg.to_dict())
    assert again == cfg
    assert again.attack.pattern.amplitude == 2.0 and isinstance(again.attack.pattern.amplitude, float)


def test_unknown_key_is_rejected_with_path():
    with pytest.raises(ConfigInvalidError) as exc:
        default_config("attack.gatt.epochs=3")

    assert "attack.gatt" in str(exc.value)


def test_wrong_type_and_length_rejected():
    for ov, needle in (("train.epochs=ten", "train.epochs"),
                       ("train.deterministic=1", "train.deterministic"),
                       ("attack.model_loss_weights=[1,2]", "model_loss_weights")):
        with pytest.raises(ConfigInvalidError) as exc:
            default_config(ov)

        assert needle in str(exc.value), ov


def test_all_errors_reported_together():
    with pytest.raises(ConfigInvalidError) as exc:
        default_config("train.epochs=ten", "foo=1")

    assert len(exc.value.messages) == 3   # tiêu đề + 2 lỗi


def test_config_is_immutable():
    cfg = default_config()
    with pytest.raises(Exception):
        cfg.dataset = "ETTh2"


# ------------------------------------------------------------------ hỗ trợ

def test_poc_slot_ok():
    assert check().verdict is SupportVerdict.OK


def test_case_insensitive_then_unsupported():
    r = check("dataset=ETTH2")
    assert r.verdict is SupportVerdict.UNSUPPORTED
    assert r.config.dataset == "ETTh2"
    assert "dataset/ETTh2/" in r.messages[0]


def test_unknown_name_is_invalid_not_unsupported():
    assert check("dataset=ETTh9").verdict is SupportVerdict.INVALID


def test_out_of_scope_selections_rejected():
    for ov in ("model=Autoformer", "pred_len=720", "pred_len=192", "method=backtime"):
        assert check(ov).verdict is SupportVerdict.UNSUPPORTED, ov


def test_raise_if_blocked_maps_to_error_types():
    with pytest.raises(UnsupportedSelectionError):
        check("dataset=ETTh2").raise_if_blocked()

    with pytest.raises(ConfigInvalidError):
        check("dataset=Foo").raise_if_blocked()


def test_semantic_contradictions_invalid():
    for ov in ("attack.target_channels=[]", "attack.poison_ratio=1.5", "attack.budget_protocol=foo",
               "attack.pattern.shape=star", "data.split_protocol=random", "train.lradj=cosine"):
        assert check(ov).verdict is SupportVerdict.INVALID, ov


def test_exit_codes():
    assert ConfigInvalidError.exit_code == 2
    assert UnsupportedSelectionError.exit_code == 3
    assert DataUnavailableError.exit_code == 4


# ------------------------------------------------------------------ dữ liệu theo cấu hình

def test_existing_folder_does_not_make_dataset_supported():
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "dataset", "ETTh2"))
        with pytest.raises(UnsupportedSelectionError):
            Experiment(tmp).resolve(default_config("dataset=ETTh2"))
    finally:
        shutil.rmtree(tmp)


def test_missing_data_creates_folder_and_reports():
    tmp = tempfile.mkdtemp()
    try:
        with pytest.raises(DataUnavailableError) as exc:
            Experiment(tmp).resolve(default_config())

        assert os.path.isdir(os.path.join(tmp, "dataset", "ETTh1"))
        assert not os.path.isdir(os.path.join(tmp, "dataset", "ETTh2"))
        assert "Đã chuẩn bị thư mục dataset/ETTh1/" in exc.value.messages[0]
    finally:
        shutil.rmtree(tmp)


def test_load_config_from_file_and_override_order():
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "c.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"train": {"epochs": 4}}, f)

        cfg = load_config(path, ["train.epochs=5"])
        assert cfg.train.epochs == 5 and cfg.train.batch_size == 32
    finally:
        shutil.rmtree(tmp)


def test_catalog_poc_slot():
    assert CATALOG.slot_supported("ETTh1", "TimesNet")
    assert CATALOG.canonical("etth1", CATALOG.datasets) == "ETTh1"
    assert CATALOG.canonical(3, CATALOG.datasets) is None


def test_more_semantic_checks():
    for ov in ("train.device=gpu", "attack.trigger_total_len=200", 'attack.target_channels=["OT","OT"]',
               "attack.model_loss_weights=[1,-1,1]", "attack.lag.min_corr_len=1", "timesnet.dropout=1.5",
               "timesnet.embed=fixed", "attack.gat.heads=0", "train.learning_rate=0"):
        assert check(ov).verdict is SupportVerdict.INVALID, ov

    assert check("train.device=cuda:1").verdict is SupportVerdict.OK


def test_errors_are_collected_not_first_only():
    r = check("train.device=gpu", "attack.gat.heads=0")
    assert len(r.messages) == 3       # tiêu đề + 2 lỗi


def test_unverified_components_are_announced():
    notes = " ".join(check().messages)
    assert "TimesNet" in notes and "clean" in notes      # chưa chạy kiểm tra torch -> chưa kiểm chứng


def test_schedule_names_match_implementation():
    from badtime_repro.support import LR_SCHEDULES
    from badtime_repro.training.schedule import SCHEDULES

    assert set(LR_SCHEDULES) == set(SCHEDULES)
