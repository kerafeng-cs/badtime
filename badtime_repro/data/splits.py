"""Giao thức phân chia theo thời gian. Mọi khoảng là half-open [s, e) theo chỉ số hàng của chuỗi gốc."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

from ..common.errors import DataUnavailableError

SPLIT_NAMES = ("train", "val", "test")

Span = Tuple[int, int]


@dataclass(frozen=True)
class SplitBounds:
    train: Span
    val: Span
    test: Span

    def __getitem__(self, name: str) -> Span:
        if name not in SPLIT_NAMES:
            raise KeyError(name)

        return getattr(self, name)

    def __iter__(self) -> Iterator[str]:
        return iter(SPLIT_NAMES)

    def items(self) -> Iterator[Tuple[str, Span]]:
        return ((name, self[name]) for name in SPLIT_NAMES)

    def to_dict(self) -> Dict[str, List[int]]:
        return {name: list(span) for name, span in self.items()}

    def validate(self, n_rows: int) -> None:
        s_prev = 0
        for name, (s, e) in self.items():
            if not (s_prev <= s <= e <= n_rows):
                raise DataUnavailableError(f"Ranh giới {name} [{s}, {e}) không hợp lệ với {n_rows} hàng.")

            s_prev = e


class SplitProtocol(ABC):
    name: str = ""

    @abstractmethod
    def boundaries(self, n_rows: int) -> SplitBounds:
        ...


class RatioSplit(SplitProtocol):
    """train = int(0.6 N), test = int(0.2 N), val = phần còn lại (như Dataset_Custom của TSLib)."""

    name = "ratio_6_2_2"

    def __init__(self, train_ratio: float = 0.6, test_ratio: float = 0.2):
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

    def boundaries(self, n_rows: int) -> SplitBounds:
        n_train = int(n_rows * self.train_ratio)
        n_test = int(n_rows * self.test_ratio)
        n_val = n_rows - n_train - n_test
        bounds = SplitBounds(
            train=(0, n_train),
            val=(n_train, n_train + n_val),
            test=(n_train + n_val, n_rows),
        )
        bounds.validate(n_rows)
        return bounds


class TslibEttHourSplit(SplitProtocol):
    """Ranh giới cố định 12/4/4 tháng của Dataset_ETT_hour trong TSLib (bỏ phần đuôi)."""

    name = "tslib_ett_hour"
    month = 30 * 24

    def boundaries(self, n_rows: int) -> SplitBounds:
        e_train = 12 * self.month
        e_val = e_train + 4 * self.month
        e_test = e_val + 4 * self.month
        if e_test > n_rows:
            raise DataUnavailableError(f"tslib_ett_hour cần ít nhất {e_test} hàng, có {n_rows}.")

        return SplitBounds(train=(0, e_train), val=(e_train, e_val), test=(e_val, e_test))


SPLIT_PROTOCOLS: Dict[str, SplitProtocol] = {p.name: p for p in (RatioSplit(), TslibEttHourSplit())}


def get_split_protocol(name: str) -> SplitProtocol:
    if name not in SPLIT_PROTOCOLS:
        raise DataUnavailableError(f"split_protocol không hợp lệ: {name}")

    return SPLIT_PROTOCOLS[name]
