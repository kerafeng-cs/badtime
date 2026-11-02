"""Registry nạp lười: ánh xạ tên -> "module:thuộc_tính", chỉ import khi cần.

Dùng cho các phần cài đặt phụ thuộc torch (mô hình, phương pháp huấn luyện, bộ chấm điểm kênh), để các lệnh
không cần torch (check, kiểm tra numpy) vẫn chạy được.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterator


class LazyRegistry:
    def __init__(self, kind: str):
        self.kind = kind
        self._targets: Dict[str, str] = {}

    def register(self, name: str, target: str) -> None:
        if name in self._targets:
            raise ValueError(f"{self.kind} '{name}' đã được đăng ký.")

        if ":" not in target:
            raise ValueError(f"Đích phải có dạng 'module:thuộc_tính', nhận '{target}'.")

        self._targets[name] = target

    def __contains__(self, name: str) -> bool:
        return name in self._targets

    def __iter__(self) -> Iterator[str]:
        return iter(self._targets)

    def target(self, name: str) -> str:
        return self._targets[name]

    def resolve(self, name: str) -> Any:
        if name not in self._targets:
            raise KeyError(f"Chưa có cài đặt cho {self.kind} '{name}'.")

        module_name, attr = self._targets[name].split(":", 1)
        return getattr(importlib.import_module(module_name), attr)
