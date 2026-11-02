"""Danh mục lựa chọn ở mức thiết kế và ma trận hỗ trợ (docs/07-configuration-support.md).

Danh mục chỉ ghi lựa chọn nào tồn tại và đang ở trạng thái nào. Phần cài đặt nằm ở các registry riêng
(`data.sources.DATA_SOURCES`, `models.MODEL_FACTORIES`, `training.methods.TRAINING_METHODS`). `tests/test_catalog.py`
buộc mọi mục có thể chạy phải có cài đặt tương ứng.

Thêm một tên vào danh mục KHÔNG làm lựa chọn đó chạy được.

Trạng thái phản ánh bằng chứng hiện có, không phải mong muốn: một thành phần chỉ lên SUPPORTED khi các kiểm tra
tương ứng đã thực sự chạy đạt. Phần torch (TimesNet, clean, badtime) chưa được chạy ở môi trường viết mã, nên còn
IMPLEMENTED; nâng lên SUPPORTED sau khi tests/test_torch_units.py và tests/test_training.py chạy đạt.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Mapping, Optional, Tuple, TypeVar


class SupportStatus(str, Enum):
    DESCRIBED = "described_not_implemented"     # Có mô tả, chưa triển khai
    IMPLEMENTED = "implemented_not_verified"    # Đã triển khai, chưa kiểm chứng
    SUPPORTED = "supported_in_scope"            # Đã hỗ trợ trong phạm vi

    @property
    def runnable(self) -> bool:
        return self is not SupportStatus.DESCRIBED


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    folder: str                       # tương đối với gốc dự án, dùng "/"
    status: SupportStatus
    source: str = ""                  # khóa trong data.sources.DATA_SOURCES
    file_name: str = ""
    expected_columns: Tuple[str, ...] = ()
    freq: str = ""
    note: str = ""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    status: SupportStatus
    source: str = ""
    note: str = ""


@dataclass(frozen=True)
class MethodSpec:
    name: str
    status: SupportStatus
    note: str = ""


ETT_COLUMNS = ("date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")

_D, _I, _S = SupportStatus.DESCRIBED, SupportStatus.IMPLEMENTED, SupportStatus.SUPPORTED

T = TypeVar("T")


@dataclass(frozen=True)
class Catalog:
    datasets: Mapping[str, DatasetSpec]
    models: Mapping[str, ModelSpec]
    methods: Mapping[str, MethodSpec]
    seq_lens: Mapping[int, SupportStatus]
    pred_lens: Mapping[int, SupportStatus]
    supported_slots: FrozenSet[Tuple[str, str]]

    @staticmethod
    def canonical(name: object, table: Mapping[str, T]) -> Optional[str]:
        """Nhận diện tên không phân biệt hoa/thường. Trả về tên chuẩn hoặc None."""
        if not isinstance(name, str):
            return None

        lowered = name.strip().lower()
        for key in table:
            if key.lower() == lowered:
                return key

        return None

    def slot_supported(self, dataset: str, model: str) -> bool:
        return (dataset, model) in self.supported_slots


def _datasets() -> Dict[str, DatasetSpec]:
    return {
        "ETTh1": DatasetSpec("ETTh1", "dataset/ETTh1", _S, "ett_csv", "ETTh1.csv", ETT_COLUMNS, "h", "Slot PoC"),
        "ETTh2": DatasetSpec("ETTh2", "dataset/ETTh2", _D, "ett_csv", "ETTh2.csv", ETT_COLUMNS, "h"),
        "ETTm1": DatasetSpec("ETTm1", "dataset/ETTm1", _D, "ett_csv", "ETTm1.csv", ETT_COLUMNS, "t"),
        "ETTm2": DatasetSpec("ETTm2", "dataset/ETTm2", _D, "ett_csv", "ETTm2.csv", ETT_COLUMNS, "t"),
        "Electricity": DatasetSpec("Electricity", "dataset/Electricity", _D),
        "Weather": DatasetSpec("Weather", "dataset/Weather", _D),
        "Exchange": DatasetSpec("Exchange", "dataset/Exchange", _D),
    }


def _models() -> Dict[str, ModelSpec]:
    return {
        "Autoformer": ModelSpec("Autoformer", _D),
        "Mamba": ModelSpec("Mamba", _D),
        "TimesNet": ModelSpec(
            "TimesNet", _I, source="thuml/Time-Series-Library@4e938a1767106324dd753b2a44832bf870a0252e",
            note="vendored, chưa chạy kiểm tra torch",
        ),
        "iTransformer": ModelSpec("iTransformer", _D),
        "TSMixer": ModelSpec("TSMixer", _D),
        "TimeMixer": ModelSpec("TimeMixer", _D),
    }


def _methods() -> Dict[str, MethodSpec]:
    return {
        "clean": MethodSpec("clean", _I, "Huấn luyện sạch (Benign), chưa chạy kiểm tra torch"),
        "badtime": MethodSpec("badtime", _I, "BadTime tái lập độc lập, còn giả định của dự án"),
        "backtime": MethodSpec(
            "backtime", _D, "Đối chứng BackTime: chưa chuyển mã xiaolin-cs/BackTime sang giao thức này"
        ),
    }


CATALOG = Catalog(
    datasets=_datasets(),
    models=_models(),
    methods=_methods(),
    seq_lens={96: _S},
    pred_lens={96: _S, 192: _D, 336: _D, 720: _D},
    # Ô duy nhất của ma trận dataset x mô hình thuộc phạm vi PoC.
    supported_slots=frozenset({("ETTh1", "TimesNet")}),
)
