"""Schema cấu hình có kiểu. Cấu trúc trùng khớp với configs/*.json.

- Mỗi mục cấu hình là một dataclass bất biến (frozen).
- `from_dict` kiểm tra nghiêm: khóa lạ, sai kiểu, sai độ dài bộ trọng số đều bị báo lỗi kèm đường dẫn khóa.
  Mọi lỗi được gom và báo cùng lúc.
- Kiểm tra ngữ nghĩa (tỉ lệ trong [0, 1], tên giao thức hợp lệ...) nằm ở `badtime_repro.support`, không nằm ở đây.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from ..common.errors import ConfigInvalidError

S = TypeVar("S", bound="ConfigSection")


class ConfigSection:
    """Lớp cơ sở của mọi mục cấu hình: dựng từ dict có kiểm tra và xuất ngược ra dict."""

    @classmethod
    def from_dict(cls: Type[S], data: Any) -> S:
        errors: List[str] = []
        obj = _build(cls, data, "", errors)
        if errors:
            raise ConfigInvalidError(["Cấu hình không hợp lệ:"] + errors)

        return obj

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: _export(getattr(self, f.name)) for f in fields(self)}

    def replace(self: S, **changes: Any) -> S:
        return dataclasses.replace(self, **changes)


# ---------------------------------------------------------------------------------------------
# Các mục cấu hình
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig(ConfigSection):
    # "ratio_6_2_2": tỉ lệ 6/2/2 trên toàn chuỗi (nguồn: v2 Bảng XI).
    # "tslib_ett_hour": ranh giới 12/4/4 tháng của Time-Series-Library (cấu hình đối chiếu riêng).
    split_protocol: str = "ratio_6_2_2"
    # Cho cửa sổ đầu vào của val/test dùng lịch sử thuộc phân vùng trước (docs/03 mục 3).
    allow_history_from_previous_split: bool = True


@dataclass(frozen=True)
class TimesNetConfig(ConfigSection):
    label_len: int = 48
    d_model: int = 16
    d_ff: int = 32
    e_layers: int = 2
    top_k: int = 5
    num_kernels: int = 6
    dropout: float = 0.1
    embed: str = "timeF"


@dataclass(frozen=True)
class TrainConfig(ConfigSection):
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    lradj: str = "type1"
    # "mse" = loss mặc định TSLib cho Benign; "mse_mae" = (||.||_2^2 + ||.||_1) như Công thức 16,
    # cùng phép tổng hợp attack.loss_reduction (dùng cho kiểm tra đường xử lý).
    clean_loss: str = "mse"
    early_stopping_patience: int = 3
    # TimesNet chọn chu kỳ bằng FFT trung bình trên CẢ LÔ -> dự báo phụ thuộc thành phần lô.
    # Giữ 32 như bộ nạp test của TSLib; không đổi giữa các nhánh.
    eval_batch_size: int = 32
    device: str = "auto"
    deterministic: bool = False


@dataclass(frozen=True)
class PatternConfig(ConfigSection):
    # Bảng III: Hump, t_PIN = 53 (= mu * t_OUT = 0.55 * 96). Công thức số là đề xuất (attack.patterns).
    shape: str = "hump_backtime_cone"
    length: int = 53
    amplitude: float = 1.0
    baseline: float = 0.0
    scale: str = "normalized"


@dataclass(frozen=True)
class LagConfig(ConfigSection):
    min_corr_len: int = 8


@dataclass(frozen=True)
class GatConfig(ConfigSection):
    hidden_dim: int = 64
    heads: int = 4
    epochs: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 256
    self_loops: bool = True
    leaky_slope: float = 0.2


@dataclass(frozen=True)
class StateSelectionConfig(ConfigSection):
    clean_tolerance: float = 0.05


@dataclass(frozen=True)
class AttackConfig(ConfigSection):
    target_channels: Tuple[str, ...] = ("OT",)
    target_channel_ratio: float = 0.1
    pattern: PatternConfig = field(default_factory=PatternConfig)
    poison_ratio: float = 0.1
    similar_ratio: float = 0.3
    poison_channel_ratio: float = 0.3
    exclude_target_from_poison_channels: bool = False
    trigger_total_len: int = 29
    # t_TGR / |V_p| = 14.5 không nguyên. Ba giao thức có nhãn riêng:
    # "keep_total": giữ tổng 29, các đoạn chênh nhau tối đa 1 bước (15 + 14)
    # "equal_segments": các đoạn bằng nhau floor(29/2)=14, tổng thực tế 28
    # "ceil_segments": các đoạn bằng nhau ceil(29/2)=15 (= L của Thuật toán 3), tổng thực tế 30
    budget_protocol: str = "keep_total"
    interp_lambda: float = 1.0
    trigger_update_every: int = 2
    lag: LagConfig = field(default_factory=LagConfig)
    gat: GatConfig = field(default_factory=GatConfig)
    model_loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    trigger_loss_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    # "sum": (||.||_2^2 + ||.||_1) cộng trên mọi phần tử của mẫu rồi chia số mẫu, đúng cách viết Công thức 16-18.
    # "mean": trung bình theo phần tử trong mỗi nhóm (cấu hình đối chiếu).
    loss_reduction: str = "sum"
    smooth_eps: float = 0.8
    trigger_learning_rate: float = 0.01
    trigger_passes_per_update: int = 1
    trigger_batch_size: int = 32
    state_selection: StateSelectionConfig = field(default_factory=StateSelectionConfig)


@dataclass(frozen=True)
class ExperimentConfig(ConfigSection):
    dataset: str = "ETTh1"
    model: str = "TimesNet"
    method: str = "clean"
    seq_len: int = 96
    pred_len: int = 96
    runs_root: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    timesnet: TimesNetConfig = field(default_factory=TimesNetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)

    @property
    def slot(self) -> str:
        return f"{self.dataset}_{self.model}_{self.seq_len}_{self.pred_len}"


# ---------------------------------------------------------------------------------------------
# Dựng / xuất
# ---------------------------------------------------------------------------------------------

def _build(cls: type, data: Any, path: str, errors: List[str]) -> Optional[Any]:
    where = path.rstrip(".") or "cấu hình gốc"
    if not isinstance(data, dict):
        errors.append(f"{where} phải là object, nhận {type(data).__name__}.")
        return None

    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    for key in sorted(set(data) - known):
        errors.append(f"Khóa '{path}{key}' không có trong schema.")

    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(hints[f.name], data[f.name], f"{path}{f.name}", errors)

    if any(v is _INVALID for v in kwargs.values()):
        return None

    return cls(**kwargs)


_INVALID = object()


def _coerce(tp: Any, value: Any, path: str, errors: List[str]) -> Any:
    if dataclasses.is_dataclass(tp):
        obj = _build(tp, value, path + ".", errors)
        return _INVALID if obj is None else obj

    origin = typing.get_origin(tp)
    if origin is tuple:
        args = typing.get_args(tp)
        if not isinstance(value, (list, tuple)):
            errors.append(f"'{path}' phải là danh sách.")
            return _INVALID

        variadic = len(args) == 2 and args[1] is Ellipsis
        if not variadic and len(value) != len(args):
            errors.append(f"'{path}' cần đúng {len(args)} phần tử, nhận {len(value)}.")
            return _INVALID

        item_types = [args[0]] * len(value) if variadic else list(args)
        items = [_coerce(t, v, f"{path}[{i}]", errors) for i, (t, v) in enumerate(zip(item_types, value))]
        return _INVALID if any(v is _INVALID for v in items) else tuple(items)

    if tp is bool:
        if isinstance(value, bool):
            return value

    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value

    elif tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

    elif tp is str:
        if isinstance(value, str):
            return value

    else:
        raise TypeError(f"Schema dùng kiểu chưa hỗ trợ: {tp}")

    errors.append(f"'{path}' phải có kiểu {tp.__name__}, nhận {value!r}.")
    return _INVALID


def _export(value: Any) -> Any:
    if isinstance(value, ConfigSection):
        return value.to_dict()

    if isinstance(value, tuple):
        return [_export(v) for v in value]

    return value
