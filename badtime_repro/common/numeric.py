"""Quy tắc làm tròn dùng chung. Mọi phép đếm từ tỉ lệ đi qua đây để có một quy ước duy nhất."""

from __future__ import annotations

import math

# Bù sai số dấu phẩy động, ví dụ 0.3 * 0.1 * 10000 = 299.99999999999994 phải cho floor = 300.
_EPS = 1e-9


def floor_count(value: float) -> int:
    return int(math.floor(value + _EPS))


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)
