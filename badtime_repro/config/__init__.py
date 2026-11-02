"""Cấu hình: schema có kiểu và bộ nạp JSON + ghi đè. Kiểm tra hỗ trợ nằm ở `badtime_repro.support`."""

from .loader import ConfigLoader, load_config
from .schema import (
    AttackConfig,
    DataConfig,
    ExperimentConfig,
    GatConfig,
    LagConfig,
    PatternConfig,
    StateSelectionConfig,
    TimesNetConfig,
    TrainConfig,
)

__all__ = [
    "AttackConfig",
    "ConfigLoader",
    "DataConfig",
    "ExperimentConfig",
    "GatConfig",
    "LagConfig",
    "PatternConfig",
    "StateSelectionConfig",
    "TimesNetConfig",
    "TrainConfig",
    "load_config",
]
