"""Phân cấp lỗi của dự án. Mỗi loại lỗi gắn một mã thoát CLI cố định.

    BadTimeError                    1  lỗi chung / vi phạm bất biến nội bộ
    ├── ConfigInvalidError          2  tên không có trong danh mục, khóa lạ, giá trị mâu thuẫn
    │   └── PlanError               2  cấu hình hợp lệ nhưng không lập được kế hoạch tác động trên dữ liệu này
    ├── UnsupportedSelectionError   3  hợp lệ về thiết kế nhưng chưa triển khai (dừng trước huấn luyện)
    ├── DataUnavailableError        4  thiếu dữ liệu hoặc dữ liệu không qua kiểm tra
    └── IntegrityError              1  bất biến thí nghiệm bị vi phạm (vd. trigger đổi trong pha cập nhật mô hình)
"""

from __future__ import annotations

from typing import Iterable, Union


class BadTimeError(Exception):
    exit_code = 1

    def __init__(self, messages: Union[str, Iterable[str]]):
        self.messages = [messages] if isinstance(messages, str) else list(messages)
        super().__init__("\n".join(self.messages))


class ConfigInvalidError(BadTimeError):
    exit_code = 2


class PlanError(ConfigInvalidError):
    exit_code = 2


class UnsupportedSelectionError(BadTimeError):
    exit_code = 3


class DataUnavailableError(BadTimeError):
    exit_code = 4


class IntegrityError(BadTimeError):
    exit_code = 1
