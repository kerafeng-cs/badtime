"""Nguồn dữ liệu: đọc file và kiểm tra chất lượng. Không sửa dữ liệu."""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..catalog import DatasetSpec
from ..common.errors import DataUnavailableError


@dataclass(frozen=True)
class DataQualityReport:
    path: str
    sha256: str
    n_rows: int
    columns: List[str]
    channels: List[str]
    start: str
    end: str
    freq: str
    n_missing: int
    n_duplicate_timestamps: int
    n_irregular_steps: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawSeries:
    dates: pd.Series
    values: np.ndarray          # [n_rows, n_channels], float64, đơn vị gốc
    report: DataQualityReport


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


class TimeSeriesSource(ABC):
    def __init__(self, spec: DatasetSpec):
        self.spec = spec

    def file_path(self, dataset_dir: str) -> str:
        return os.path.join(dataset_dir, self.spec.file_name)

    @abstractmethod
    def load(self, path: str) -> RawSeries:
        """Đọc và kiểm tra; ném DataUnavailableError nếu dữ liệu không hợp lệ."""


class EttCsvSource(TimeSeriesSource):
    """CSV dạng ETT: cột 'date' rồi các kênh số, bước thời gian đều."""

    STEPS = {"h": pd.Timedelta(hours=1), "t": pd.Timedelta(minutes=15)}

    def load(self, path: str) -> RawSeries:
        if not os.path.isfile(path):
            raise DataUnavailableError(f"Thiếu file dữ liệu: {path}")

        df = pd.read_csv(path)
        columns = list(df.columns)
        if tuple(columns) != tuple(self.spec.expected_columns):
            raise DataUnavailableError(
                f"Cột không khớp. Kỳ vọng {list(self.spec.expected_columns)}, nhận {columns}"
            )

        dates = pd.to_datetime(df["date"])
        values = df[columns[1:]].to_numpy(dtype=np.float64)
        n_missing, n_dup, n_irregular = self._quality(dates, values)

        report = DataQualityReport(
            path=os.path.abspath(path),
            sha256=sha256_of(path),
            n_rows=int(len(df)),
            columns=columns,
            channels=columns[1:],
            start=str(dates.iloc[0]),
            end=str(dates.iloc[-1]),
            freq=self.spec.freq,
            n_missing=n_missing,
            n_duplicate_timestamps=n_dup,
            n_irregular_steps=n_irregular,
        )
        self._raise_on_problems(report)
        return RawSeries(dates=dates, values=values, report=report)

    def _quality(self, dates: pd.Series, values: np.ndarray) -> Tuple[int, int, int]:
        n_missing = int(np.isnan(values).sum()) + int(dates.isna().sum())
        n_dup = int(dates.duplicated().sum())
        diffs = dates.diff().iloc[1:]
        n_irregular = int((diffs != self.STEPS[self.spec.freq]).sum())
        return n_missing, n_dup, n_irregular

    @staticmethod
    def _raise_on_problems(report: DataQualityReport) -> None:
        problems = []
        if report.n_missing:
            problems.append(f"{report.n_missing} giá trị thiếu")

        if report.n_duplicate_timestamps:
            problems.append(f"{report.n_duplicate_timestamps} mốc thời gian trùng")

        if report.n_irregular_steps:
            problems.append(f"{report.n_irregular_steps} bước thời gian không đều")

        if problems:
            raise DataUnavailableError("Dữ liệu không hợp lệ: " + "; ".join(problems) + ". Không tự sửa dữ liệu.")


DATA_SOURCES: Dict[str, Callable[[DatasetSpec], TimeSeriesSource]] = {"ett_csv": EttCsvSource}
