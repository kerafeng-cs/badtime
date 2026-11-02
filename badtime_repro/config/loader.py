"""Nạp cấu hình: mặc định của schema <- file JSON <- các ghi đè `--set khoa.con=giatri` (theo thứ tự đó)."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, Iterable, Optional, Tuple

from ..common.errors import ConfigInvalidError
from .schema import ExperimentConfig


class ConfigLoader:
    def load(self, path: Optional[str] = None, overrides: Iterable[str] = ()) -> ExperimentConfig:
        raw = ExperimentConfig().to_dict()
        if path is not None:
            with open(path, "r", encoding="utf-8") as f:
                raw = self.merge(raw, json.load(f))

        for text in overrides:
            key, value = self.parse_override(text)
            self.apply_override(raw, key, value)

        return ExperimentConfig.from_dict(raw)

    @staticmethod
    def merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = ConfigLoader.merge(out[key], value)
            else:
                out[key] = copy.deepcopy(value)

        return out

    @staticmethod
    def parse_override(text: str) -> Tuple[str, Any]:
        """'a.b=3' -> ('a.b', 3). Giá trị được đọc như JSON nếu được, ngược lại là chuỗi."""
        if "=" not in text:
            raise ConfigInvalidError(f"Ghi đè phải có dạng khoa=giatri: {text}")

        key, raw = text.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw

        return key.strip(), value

    @staticmethod
    def apply_override(raw: Dict[str, Any], key: str, value: Any) -> None:
        """Khóa chưa tồn tại vẫn được ghi vào; schema sẽ báo 'không có trong schema' kèm đường dẫn."""
        parts = key.split(".")
        node = raw
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}

            node = node[part]

        node[parts[-1]] = value


def load_config(path: Optional[str] = None, overrides: Iterable[str] = ()) -> ExperimentConfig:
    return ConfigLoader().load(path, overrides)
