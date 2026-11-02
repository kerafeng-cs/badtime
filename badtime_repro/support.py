"""Kiểm tra TOÀN BỘ tổ hợp trước khi chạy (docs/07). Không bao giờ tự chuyển sang slot PoC.

Nằm ở tầng ứng dụng (trên config, data, attack) vì cần biết tên các giao thức mà các tầng đó cài đặt.

Ba kết luận tách biệt:
- INVALID      : tên không có trong danh mục hoặc giá trị mâu thuẫn
- UNSUPPORTED  : hợp lệ về thiết kế nhưng chưa triển khai -> dừng trước huấn luyện
- OK           : được phép chạy (vẫn còn bước kiểm tra dữ liệu)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional, Tuple

from .attack.budget import BUDGET_PROTOCOLS
from .attack.patterns import PATTERNS
from .catalog import CATALOG, Catalog, SupportStatus
from .common.errors import ConfigInvalidError, UnsupportedSelectionError
from .config.schema import ExperimentConfig
from .data.splits import SPLIT_PROTOCOLS

# Tên hợp lệ của các lựa chọn mà phần cài đặt nằm ở tầng torch (không import được ở đây).
# tests/test_config.py đối chiếu các bộ tên này với phần cài đặt tương ứng.
CLEAN_LOSSES = ("mse", "mse_mae")
LOSS_REDUCTIONS = ("mean", "sum")
LR_SCHEDULES = ("type1", "constant")
TIMESNET_EMBEDS = ("timeF",)
PATTERN_SCALES = ("normalized",)


class SupportVerdict(str, Enum):
    OK = "ok"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


@dataclass(frozen=True)
class SupportReport:
    verdict: SupportVerdict
    messages: Tuple[str, ...] = ()
    config: Optional[ExperimentConfig] = None     # cấu hình với tên đã chuẩn hóa (None nếu INVALID ở bước tên)

    @property
    def ok(self) -> bool:
        return self.verdict is SupportVerdict.OK

    def raise_if_blocked(self) -> ExperimentConfig:
        if self.verdict is SupportVerdict.INVALID:
            raise ConfigInvalidError(self.messages)

        if self.verdict is SupportVerdict.UNSUPPORTED:
            raise UnsupportedSelectionError(self.messages)

        return self.config


@dataclass
class SupportChecker:
    catalog: Catalog = field(default_factory=lambda: CATALOG)

    def check(self, cfg: ExperimentConfig) -> SupportReport:
        cat = self.catalog
        dataset = cat.canonical(cfg.dataset, cat.datasets)
        model = cat.canonical(cfg.model, cat.models)
        method = cat.canonical(cfg.method, cat.methods)

        invalid = self._name_errors(cfg, dataset, model, method)
        if invalid:
            return SupportReport(SupportVerdict.INVALID, tuple(["Cấu hình không hợp lệ:"] + invalid))

        resolved = cfg.replace(dataset=dataset, model=model, method=method)
        blocked = self._support_errors(resolved)
        if blocked:
            return SupportReport(SupportVerdict.UNSUPPORTED, tuple(blocked), resolved)

        contradictions = self.consistency_errors(resolved)
        if contradictions:
            return SupportReport(SupportVerdict.INVALID, tuple(["Cấu hình không hợp lệ:"] + contradictions), resolved)

        notes = []
        for kind, spec in (("dataset", cat.datasets[dataset]), ("mô hình", cat.models[model]),
                           ("phương pháp", cat.methods[method])):
            if spec.status is SupportStatus.IMPLEMENTED:
                detail = f" ({spec.note})" if spec.note else ""
                notes.append(f"Lưu ý: {kind} {spec.name} ở trạng thái 'đã triển khai, chưa kiểm chứng'{detail}.")

        return SupportReport(SupportVerdict.OK, tuple(notes), resolved)

    # -----------------------------------------------------------------------------------------

    def _name_errors(self, cfg: ExperimentConfig, dataset: Optional[str], model: Optional[str],
                     method: Optional[str]) -> List[str]:
        cat = self.catalog
        errors = []
        if dataset is None:
            errors.append(f"Dataset '{cfg.dataset}' không có trong danh mục. Hợp lệ: {', '.join(cat.datasets)}.")

        if model is None:
            errors.append(f"Mô hình '{cfg.model}' không có trong danh mục. Hợp lệ: {', '.join(cat.models)}.")

        if method is None:
            errors.append(f"Phương pháp '{cfg.method}' không có trong danh mục. Hợp lệ: {', '.join(cat.methods)}.")

        if cfg.seq_len not in cat.seq_lens:
            errors.append(f"Độ dài lịch sử {cfg.seq_len} không có trong thiết kế. Hợp lệ: {list(cat.seq_lens)}.")

        if cfg.pred_len not in cat.pred_lens:
            errors.append(f"Khoảng dự báo {cfg.pred_len} không có trong thiết kế. Hợp lệ: {list(cat.pred_lens)}.")

        return errors

    def _support_errors(self, cfg: ExperimentConfig) -> List[str]:
        cat = self.catalog
        ds = cat.datasets[cfg.dataset]
        md = cat.models[cfg.model]
        mt = cat.methods[cfg.method]
        messages = []

        if not ds.status.runnable:
            messages.append(
                f"{ds.name} có trong thiết kế, nhưng chưa được hỗ trợ ở PoC. Thư mục dự kiến: {ds.folder}/. "
                "Phạm vi PoC là ETTh1 + TimesNet, dự báo 96 bước."
            )

        if not md.status.runnable:
            messages.append(f"{md.name} chưa được tích hợp cho PoC. Hiện chưa thể chạy tổ hợp {ds.name} + {md.name}.")

        if not messages and not cat.slot_supported(ds.name, md.name):
            messages.append(f"Tổ hợp {ds.name} + {md.name} chưa được kiểm tra; chưa hỗ trợ.")

        if not cat.seq_lens[cfg.seq_len].runnable:
            messages.append(f"Độ dài lịch sử {cfg.seq_len} chưa được hỗ trợ ở PoC.")

        if not cat.pred_lens[cfg.pred_len].runnable:
            messages.append(f"Khoảng dự báo {cfg.pred_len} nằm trong thiết kế mở rộng, chưa được hỗ trợ ở PoC.")

        if not mt.status.runnable:
            messages.append(f"Phương pháp {mt.name}: {mt.note}. Chưa thể chạy.")

        return messages

    @staticmethod
    def consistency_errors(cfg: ExperimentConfig) -> List[str]:
        """Kiểm tra ngữ nghĩa (schema đã đảm bảo đúng kiểu). Gom mọi lỗi, không dừng ở lỗi đầu tiên."""
        errors: List[str] = []
        tr, tn, att = cfg.train, cfg.timesnet, cfg.attack
        pat, gat = att.pattern, att.gat

        def one_of(path: str, value: str, allowed: Iterable[str]) -> None:
            allowed = list(allowed)
            if value not in allowed:
                errors.append(f"{path} = '{value}' không hợp lệ; phải thuộc {allowed}.")

        def at_least(path: str, value: float, low: float) -> None:
            if value < low:
                errors.append(f"{path} phải >= {low}, nhận {value}.")

        def within(path: str, value: float, low: float, high: float) -> None:
            if not (low <= value <= high):
                errors.append(f"{path} phải nằm trong [{low}, {high}], nhận {value}.")

        # Danh mục giá trị rời rạc
        one_of("data.split_protocol", cfg.data.split_protocol, SPLIT_PROTOCOLS)
        one_of("train.clean_loss", tr.clean_loss, CLEAN_LOSSES)
        one_of("train.lradj", tr.lradj, LR_SCHEDULES)
        one_of("timesnet.embed", tn.embed, TIMESNET_EMBEDS)
        one_of("attack.pattern.shape", pat.shape, PATTERNS)
        one_of("attack.pattern.scale", pat.scale, PATTERN_SCALES)
        one_of("attack.budget_protocol", att.budget_protocol, BUDGET_PROTOCOLS)
        one_of("attack.loss_reduction", att.loss_reduction, LOSS_REDUCTIONS)
        if not (tr.device in ("auto", "cpu") or tr.device == "cuda" or tr.device.startswith("cuda:")):
            errors.append(f"train.device = '{tr.device}' không hợp lệ; dùng 'auto', 'cpu', 'cuda' hoặc 'cuda:<i>'.")

        # Số nguyên dương
        for path, value in (
            ("train.epochs", tr.epochs), ("train.batch_size", tr.batch_size),
            ("train.eval_batch_size", tr.eval_batch_size),
            ("train.early_stopping_patience", tr.early_stopping_patience),
            ("timesnet.d_model", tn.d_model), ("timesnet.d_ff", tn.d_ff), ("timesnet.e_layers", tn.e_layers),
            ("timesnet.top_k", tn.top_k), ("timesnet.num_kernels", tn.num_kernels),
            ("attack.gat.hidden_dim", gat.hidden_dim), ("attack.gat.heads", gat.heads),
            ("attack.gat.epochs", gat.epochs), ("attack.gat.batch_size", gat.batch_size),
            ("attack.trigger_total_len", att.trigger_total_len),
            ("attack.trigger_update_every", att.trigger_update_every),
            ("attack.trigger_passes_per_update", att.trigger_passes_per_update),
            ("attack.trigger_batch_size", att.trigger_batch_size),
        ):
            at_least(path, value, 1)

        at_least("timesnet.label_len", tn.label_len, 0)
        at_least("attack.lag.min_corr_len", att.lag.min_corr_len, 2)      # tương quan cần ít nhất 2 điểm

        # Số thực
        for path, value in (("train.learning_rate", tr.learning_rate), ("attack.gat.learning_rate", gat.learning_rate),
                            ("attack.trigger_learning_rate", att.trigger_learning_rate)):
            if value <= 0:
                errors.append(f"{path} phải > 0, nhận {value}.")

        within("timesnet.dropout", tn.dropout, 0.0, 1.0)
        for key in ("poison_ratio", "similar_ratio", "poison_channel_ratio", "target_channel_ratio", "interp_lambda"):
            within(f"attack.{key}", getattr(att, key), 0.0, 1.0)

        at_least("attack.smooth_eps", att.smooth_eps, 0.0)
        at_least("attack.gat.leaky_slope", gat.leaky_slope, 0.0)
        at_least("attack.state_selection.clean_tolerance", att.state_selection.clean_tolerance, 0.0)
        for name, weights in (("model_loss_weights", att.model_loss_weights),
                              ("trigger_loss_weights", att.trigger_loss_weights)):
            if any(w < 0 for w in weights):
                errors.append(f"attack.{name} không được có trọng số âm.")

        # Quan hệ giữa các trường
        if not (1 <= pat.length <= cfg.pred_len):
            errors.append(f"attack.pattern.length = {pat.length} phải nằm trong [1, {cfg.pred_len}].")

        if att.trigger_total_len > cfg.seq_len:
            errors.append(f"attack.trigger_total_len = {att.trigger_total_len} vượt độ dài lịch sử {cfg.seq_len}.")

        if not att.target_channels:
            errors.append("attack.target_channels phải nêu rõ tên kênh đích; không suy từ thứ tự cột.")
        elif len(set(att.target_channels)) != len(att.target_channels):
            errors.append("attack.target_channels có tên kênh bị lặp.")

        return errors
