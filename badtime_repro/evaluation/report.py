"""Tổng hợp các lần chạy của một slot và so với số của bài báo (docs/06 'Quy tắc báo cáo')."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..common.errors import DataUnavailableError
from ..common.jsonio import read_json, write_json
from ..runs.store import RunIndex

METRICS = ("mae_c", "mae_pa", "mae_pn")


@dataclass(frozen=True)
class PaperReference:
    slot: str
    source: str
    rows: Mapping[str, Mapping[str, float]]


# Số của bài báo v2 (Bảng V, horizon 96). Thang chưa xác nhận (A20).
BADTIME_V2_TABLE_V = PaperReference(
    slot="ETTh1_TimesNet_96_96",
    source="BadTime v2: Bảng V (horizon 96), trùng cột TimesNet/ETTh1 của Bảng IV",
    rows={
        "Benign": {"mae_c": 0.424, "mae_pa": 0.781, "mae_pn": 0.432},
        "BackTime": {"mae_c": 0.448, "mae_pa": 0.750, "mae_pn": 0.439},
        "BadTime": {"mae_c": 0.421, "mae_pa": 0.376, "mae_pn": 0.429},
    },
)


@dataclass(frozen=True)
class MetricSummary:
    runs: List[float]
    paper: Optional[float]
    warn_rel: float

    @property
    def mean(self) -> float:
        return float(np.mean(self.runs))

    @property
    def std(self) -> float:
        return float(np.std(self.runs, ddof=1)) if len(self.runs) > 1 else float("nan")

    @property
    def abs_diff(self) -> Optional[float]:
        return None if self.paper is None else self.mean - self.paper

    @property
    def rel_diff(self) -> Optional[float]:
        if self.paper is None or self.paper == 0:
            return None

        return (self.mean - self.paper) / self.paper

    @property
    def warn(self) -> bool:
        return self.rel_diff is not None and abs(self.rel_diff) > self.warn_rel

    def to_dict(self) -> Dict[str, Any]:
        return {"runs": self.runs, "mean": self.mean, "std": self.std, "paper": self.paper,
                "abs_diff": self.abs_diff, "rel_diff": self.rel_diff, "warn_over_5pct": self.warn}


@dataclass
class ReportBuilder:
    slot_dir: str
    slot: str
    reference: PaperReference = BADTIME_V2_TABLE_V
    warn_rel: float = 0.05
    splits: Tuple[str, ...] = field(default=("val", "test"))

    def build(self) -> Dict[str, Any]:
        index = RunIndex(self.slot_dir)
        bad_runs = index.completed("badtime")
        clean_runs = index.completed("clean")
        if not bad_runs and not clean_runs:
            raise DataUnavailableError(f"Chưa có lần chạy hoàn thành nào trong {self.slot_dir}.")

        refs = self.reference.rows if self.slot == self.reference.slot else {}

        report: Dict[str, Any] = {
            "slot": self.slot, "reference": self.reference.source, "warn_rel": self.warn_rel, "rows": {},
        }
        for split in self.splits:
            rows: Dict[str, Any] = {}

            badtime = {m: [r.metrics()[split][f"{m}_norm"] for r in bad_runs] for m in METRICS}
            if bad_runs:
                rows["BadTime"] = self._row(badtime, refs.get("BadTime", {}), len(bad_runs))

            # Benign MAE_pa / MAE_pn = mô hình sạch nhận trigger BadTime đã khóa (từ ControlExperiment).
            benign: Dict[str, List[float]] = {"mae_c": [r.metrics()[split]["mae_c_norm"] for r in clean_runs],
                                              "mae_pa": [], "mae_pn": []}
            for r in bad_runs:
                for f in r.controls_files():
                    ctl = read_json(f)[split]["clean_model|trigger_input"]
                    benign["mae_pa"].append(ctl["mae_pa_norm"])
                    benign["mae_pn"].append(ctl["mae_pn_norm"])

            if clean_runs:
                rows["Benign"] = self._row({k: v for k, v in benign.items() if v}, refs.get("Benign", {}),
                                           len(clean_runs))

            report["rows"][split] = rows

        return report

    def _row(self, values: Dict[str, List[float]], refs: Mapping[str, float], n_runs: int) -> Dict[str, Any]:
        row: Dict[str, Any] = {k: MetricSummary(v, refs.get(k), self.warn_rel).to_dict() for k, v in values.items()}
        row["n_runs"] = n_runs
        return row

    def write(self, report: Dict[str, Any]) -> str:
        write_json(os.path.join(self.slot_dir, "report.json"), report)
        path = os.path.join(self.slot_dir, "report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(MarkdownReportRenderer().render(report))

        return path


class MarkdownReportRenderer:
    def render(self, report: Dict[str, Any]) -> str:
        lines = [
            f"# Báo cáo {report['slot']}",
            "",
            f"Mốc nguồn: {report['reference']}. Thang so sánh: chuẩn hóa (giả định A20, chưa xác nhận).",
            "Benign MAE_pa / MAE_pn = mô hình sạch nhận trigger BadTime đã khóa (từ lệnh controls).",
            f"Cảnh báo kỹ thuật khi |sai lệch tương đối| > {int(report['warn_rel'] * 100)}%; "
            "không phải kiểm định thống kê.",
            "",
        ]
        for split, rows in report["rows"].items():
            lines += [
                f"## Tập {split}",
                "",
                "| Nhánh | Chỉ số | Bài báo | Các lần chạy | TB | Độ lệch chuẩn "
                "| Chênh tuyệt đối | Chênh tương đối | Cảnh báo |",
                "|---|---|---:|---|---:|---:|---:|---:|---|",
            ]
            for method, cols in rows.items():
                for metric, r in cols.items():
                    if metric == "n_runs":
                        continue

                    lines.append(self._line(method, metric, r))

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _line(method: str, metric: str, r: Dict[str, Any]) -> str:
        runs = ", ".join(f"{v:.4f}" for v in r["runs"])
        paper = f"{r['paper']:.3f}" if r["paper"] is not None else "—"
        absd = f"{r['abs_diff']:+.4f}" if r["abs_diff"] is not None else "—"
        reld = f"{r['rel_diff'] * 100:+.1f}%" if r["rel_diff"] is not None else "—"
        warn = "⚠" if r["warn_over_5pct"] else ""
        cells = [method, metric, paper, runs, f"{r['mean']:.4f}", f"{r['std']:.4f}", absd, reld, warn]
        return "| " + " | ".join(cells) + " |"
