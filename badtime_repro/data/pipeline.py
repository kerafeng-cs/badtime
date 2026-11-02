"""Định vị thư mục dataset -> đọc & kiểm tra -> phân chia -> chuẩn hóa bằng train -> chỉ số cửa sổ."""

from __future__ import annotations

import os
from typing import Tuple

from ..catalog import CATALOG, Catalog, DatasetSpec
from ..common.errors import DataUnavailableError
from ..config.schema import ExperimentConfig
from .dataset import PreparedDataset
from .scaling import StandardScaler
from .sources import DATA_SOURCES, RawSeries, TimeSeriesSource
from .splits import SPLIT_NAMES, get_split_protocol
from .time_features import TIME_FEATURES, TimeFeatureEncoder
from .windows import WindowSpec


class DataPipeline:
    def __init__(self, project_root: str, catalog: Catalog = CATALOG):
        self.project_root = project_root
        self.catalog = catalog

    def dataset_dir(self, spec: DatasetSpec) -> str:
        return os.path.join(self.project_root, *spec.folder.split("/"))

    def locate(self, spec: DatasetSpec) -> Tuple[TimeSeriesSource, str]:
        """Tạo thư mục chuẩn nếu thiếu (giữ nguyên nội dung sẵn có). Thiếu file -> DataUnavailableError."""
        directory = self.dataset_dir(spec)
        created = not os.path.isdir(directory)
        os.makedirs(directory, exist_ok=True)

        source = DATA_SOURCES[spec.source](spec)
        path = source.file_path(directory)
        if not os.path.isfile(path):
            prefix = "Đã chuẩn bị" if created else "Đã có"
            raise DataUnavailableError(
                f"{prefix} thư mục {spec.folder}/. Cần bổ sung dữ liệu {spec.name} hợp lệ trước khi chạy "
                f"(file kỳ vọng: {spec.folder}/{spec.file_name})."
            )

        return source, path

    def prepare(self, cfg: ExperimentConfig) -> PreparedDataset:
        """`cfg` phải đã qua SupportChecker (tên chuẩn hóa, tổ hợp được hỗ trợ)."""
        spec = self.catalog.datasets[cfg.dataset]
        if spec.freq not in TIME_FEATURES:
            raise DataUnavailableError(
                f"Chưa có bộ mã hóa đặc trưng thời gian cho tần suất '{spec.freq}' của {spec.name}."
            )

        source, path = self.locate(spec)
        try:
            raw = source.load(path)
        except DataUnavailableError as exc:
            raise DataUnavailableError(
                [f"Dữ liệu {spec.name} không hợp lệ tại {spec.folder}/: {m}" for m in exc.messages]
            )

        return self.build(raw, cfg, TIME_FEATURES[spec.freq])

    @staticmethod
    def build(raw: RawSeries, cfg: ExperimentConfig, time_features: TimeFeatureEncoder) -> PreparedDataset:
        """Phân chia -> chuẩn hóa bằng thống kê của train -> đặc trưng thời gian -> chỉ số cửa sổ."""
        values = raw.values
        bounds = get_split_protocol(cfg.data.split_protocol).boundaries(len(values))
        b_s, b_e = bounds.train
        scaler = StandardScaler.fit(values[b_s:b_e])
        window = WindowSpec(cfg.seq_len, cfg.pred_len)

        starts = {
            split: window.starts(bounds, split, cfg.data.allow_history_from_previous_split)
            for split in SPLIT_NAMES
        }
        for split, arr in starts.items():
            if len(arr) == 0:
                raise DataUnavailableError(f"Phân vùng {split} không có cửa sổ hợp lệ.")

        return PreparedDataset(
            report=raw.report,
            raw=values,
            norm=scaler.transform(values),
            marks=time_features.encode(raw.dates),
            scaler=scaler,
            bounds=bounds,
            starts=starts,
            channels=list(raw.report.channels),
            window=window,
        )
