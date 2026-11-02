"""Cài đặt mô hình. Tên khớp với `catalog.CATALOG.models`; chỉ import torch khi resolve."""

from typing import TYPE_CHECKING

from ..common.registry import LazyRegistry

if TYPE_CHECKING:
    from ..config.schema import ExperimentConfig
    from .base import Forecaster

MODEL_FACTORIES = LazyRegistry("mô hình")
MODEL_FACTORIES.register("TimesNet", "badtime_repro.models.timesnet:TimesNetFactory")


def build_model(cfg: "ExperimentConfig", n_channels: int) -> "Forecaster":
    return MODEL_FACTORIES.resolve(cfg.model)().build(cfg, n_channels)
