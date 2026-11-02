"""Chia tổng độ dài trigger t_TGR cho |V_p| kênh.

v2 ghi độ dài đoạn t_TGR/|V_p| (= 14.5, không nguyên): giả định A08.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from ..common.errors import PlanError
from ..common.numeric import ceil_div


class BudgetProtocol(ABC):
    name: str = ""

    def split(self, total_len: int, n_channels: int) -> List[int]:
        if n_channels < 1:
            raise PlanError("Cần ít nhất 1 kênh bị sửa.")

        if n_channels > total_len:
            raise PlanError(f"Ngân sách trigger {total_len} nhỏ hơn số kênh bị sửa {n_channels}.")

        return self._split(total_len, n_channels)

    @abstractmethod
    def _split(self, total_len: int, n_channels: int) -> List[int]:
        ...


class KeepTotal(BudgetProtocol):
    """Giữ đúng tổng; các đoạn chênh nhau tối đa 1 bước, đoạn dài hơn đứng trước (29 -> 15 + 14)."""

    name = "keep_total"

    def _split(self, total_len: int, n_channels: int) -> List[int]:
        base = total_len // n_channels
        extra = total_len - base * n_channels
        return [base + 1 if i < extra else base for i in range(n_channels)]


class EqualSegments(BudgetProtocol):
    """Các đoạn bằng nhau floor(t_TGR / |V_p|); tổng thực tế có thể nhỏ hơn (29 -> 14 + 14)."""

    name = "equal_segments"

    def _split(self, total_len: int, n_channels: int) -> List[int]:
        return [total_len // n_channels] * n_channels


class CeilSegments(BudgetProtocol):
    """Các đoạn bằng nhau ceil(t_TGR / |V_p|) = L của Thuật toán 3; tổng thực tế có thể lớn hơn (29 -> 15 + 15)."""

    name = "ceil_segments"

    def _split(self, total_len: int, n_channels: int) -> List[int]:
        return [ceil_div(total_len, n_channels)] * n_channels


BUDGET_PROTOCOLS: Dict[str, BudgetProtocol] = {p.name: p for p in (KeepTotal(), EqualSegments(), CeilSegments())}


def get_budget_protocol(name: str) -> BudgetProtocol:
    if name not in BUDGET_PROTOCOLS:
        raise PlanError(f"budget_protocol không hợp lệ: {name}")

    return BUDGET_PROTOCOLS[name]
