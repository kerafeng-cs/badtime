"""TimesNet từ Time-Series-Library (vendored, commit ghim trong TSLIB_COMMIT)."""

from __future__ import annotations

from types import SimpleNamespace

from ..catalog import CATALOG
from ..config.schema import ExperimentConfig
from .base import Forecaster, ModelFactory
from .tslib.timesnet import Model as TslibTimesNet

TSLIB_COMMIT = "4e938a1767106324dd753b2a44832bf870a0252e"


class TimesNetFactory(ModelFactory):
    def build(self, cfg: ExperimentConfig, n_channels: int) -> Forecaster:
        tn = cfg.timesnet
        args = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=cfg.seq_len,
            label_len=tn.label_len,
            pred_len=cfg.pred_len,
            e_layers=tn.e_layers,
            enc_in=n_channels,
            c_out=n_channels,
            d_model=tn.d_model,
            d_ff=tn.d_ff,
            top_k=tn.top_k,
            num_kernels=tn.num_kernels,
            dropout=tn.dropout,
            embed=tn.embed,
            freq=CATALOG.datasets[cfg.dataset].freq,     # chọn số đặc trưng thời gian của nhúng timeF
        )
        return Forecaster(TslibTimesNet(args))
