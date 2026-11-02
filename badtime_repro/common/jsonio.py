"""Đọc/ghi JSON UTF-8, chấp nhận kiểu numpy."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def _default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if hasattr(obj, "to_dict"):
        return obj.to_dict()

    raise TypeError(f"Không tuần tự hóa được {type(obj)}")


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
