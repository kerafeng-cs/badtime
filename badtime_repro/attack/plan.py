"""Kế hoạch tác động của BadTime: kết quả của bốn khối thiết kế (Thuật toán 1-4), cố định suốt quá trình huấn luyện.

`AttackPlanner` điều phối các khối; mỗi khối là một lớp riêng có thể thay thế:
    TargetChannelResolver   -> V_t
    TargetPattern           -> p
    InfluenceScorer + PoisonChannelSelector -> V_p           (Thuật toán 2)
    BudgetProtocol          -> độ dài từng đoạn trigger
    LagEstimator            -> delta*, vị trí đoạn trong đầu vào (Thuật toán 3)
    HybridSampleSelector    -> T_poison                       (Thuật toán 1)
    khởi tạo từ t*          -> tau_0                          (Thuật toán 4)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np

from ..common.errors import PlanError
from ..common.numeric import ceil_div
from ..config.schema import AttackConfig
from ..data.dataset import PreparedDataset
from .budget import get_budget_protocol
from .channels import InfluenceResult, InfluenceScorer, PoisonChannelSelector, TargetChannelResolver
from .lag import LagEstimator
from .patterns import get_pattern
from .sample_selection import HybridSampleSelector


@dataclass
class AttackPlan:
    channels: List[str]
    target_idx: List[int]
    poison_idx: List[int]
    seg_lens: List[int]
    lags: List[int]
    offsets: List[int]              # đoạn i chiếm [offsets[i], offsets[i] + seg_lens[i]) trong cửa sổ đầu vào
    pattern: np.ndarray             # [t_pin, n_target], thang chuẩn hóa
    poison_starts: np.ndarray       # s tuyệt đối của các cửa sổ huấn luyện bị chọn, tăng dần
    similar_starts: np.ndarray
    different_starts: np.ndarray
    init_start: int                 # s của cửa sổ có nhãn gần mục tiêu nhất (Công thức 8)
    init_trigger: np.ndarray        # [sum(seg_lens)] nối các đoạn theo thứ tự poison_idx (Công thức 10)
    range_min: np.ndarray           # [n_poison] min của kênh trên phần huấn luyện (thang chuẩn hóa)
    range_max: np.ndarray
    interp_lambda: float
    seq_len: int
    pred_len: int
    trace: Dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------------------- thuộc tính dẫn xuất

    @property
    def t_pin(self) -> int:
        return int(self.pattern.shape[0])

    @property
    def trigger_len(self) -> int:
        return int(sum(self.seg_lens))

    @property
    def seg_bounds(self) -> List[Tuple[int, int]]:
        """[s, e) của từng đoạn trong vector trigger phẳng."""
        out = []
        s = 0
        for length in self.seg_lens:
            out.append((s, s + length))
            s += length

        return out

    def segments(self) -> Iterator[Tuple[Tuple[int, int], int, int]]:
        """Lặp (đoạn [t_s, t_e) trong trigger, kênh, vị trí bắt đầu trong đầu vào)."""
        return zip(self.seg_bounds, self.poison_idx, self.offsets)

    # ----------------------------------------------------------------------------- bất biến

    def validate(self) -> None:
        n_ch = len(self.channels)
        for name, idx in (("target_idx", self.target_idx), ("poison_idx", self.poison_idx)):
            if not idx or any(not (0 <= i < n_ch) for i in idx):
                raise PlanError(f"{name} rỗng hoặc ngoài [0, {n_ch}).")

        if len(set(self.poison_idx)) != len(self.poison_idx) or len(set(self.target_idx)) != len(self.target_idx):
            raise PlanError("Kênh bị sửa hoặc kênh đích bị lặp.")

        if self.pattern.ndim != 2 or self.pattern.shape[1] != len(self.target_idx):
            raise PlanError("Mục tiêu phải có dạng [t_pin, số kênh đích].")

        if len(self.range_min) != len(self.poison_idx) or len(self.range_max) != len(self.poison_idx):
            raise PlanError("range_min/range_max phải có một giá trị cho mỗi kênh bị sửa.")

        if not (0.0 <= self.interp_lambda <= 1.0):
            raise PlanError("interp_lambda phải trong [0, 1].")

        starts = np.asarray(self.poison_starts)
        if len(starts) > 1 and np.any(np.diff(starts) <= 0):
            raise PlanError("poison_starts phải tăng ngặt (không lặp).")

        if len(np.intersect1d(self.similar_starts, self.different_starts)) > 0:
            raise PlanError("Nhóm tương tự và nhóm khác biệt giao nhau.")

        if not (len(self.poison_idx) == len(self.seg_lens) == len(self.lags) == len(self.offsets)):
            raise PlanError("Số kênh, số đoạn, số độ trễ không khớp.")

        for off, length in zip(self.offsets, self.seg_lens):
            if off < 0 or off + length > self.seq_len:
                raise PlanError(f"Đoạn trigger [{off}, {off + length}) nằm ngoài đầu vào [0, {self.seq_len}).")

        if self.t_pin > self.pred_len:
            raise PlanError("Đoạn mục tiêu dài hơn khoảng dự báo.")

        if len(self.init_trigger) != self.trigger_len:
            raise PlanError("Độ dài trigger khởi tạo không khớp ngân sách.")

    # ----------------------------------------------------------------------------- tuần tự hóa

    def to_dict(self) -> Dict[str, Any]:
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items()}
        d["channel_names"] = {
            "target": [self.channels[i] for i in self.target_idx],
            "poison": [self.channels[i] for i in self.poison_idx],
        }
        d["budget"] = {
            "n_poison_windows": int(len(self.poison_starts)),
            "n_poison_channels": len(self.poison_idx),
            "trigger_cells_per_window": self.trigger_len,
            "total_modified_input_cells": int(self.trigger_len * len(self.poison_starts)),
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AttackPlan":
        plan = cls(
            channels=list(d["channels"]),
            target_idx=[int(v) for v in d["target_idx"]],
            poison_idx=[int(v) for v in d["poison_idx"]],
            seg_lens=[int(v) for v in d["seg_lens"]],
            lags=[int(v) for v in d["lags"]],
            offsets=[int(v) for v in d["offsets"]],
            pattern=np.asarray(d["pattern"], dtype=np.float64),
            poison_starts=np.asarray(d["poison_starts"], dtype=np.int64),
            similar_starts=np.asarray(d["similar_starts"], dtype=np.int64),
            different_starts=np.asarray(d["different_starts"], dtype=np.int64),
            init_start=int(d["init_start"]),
            init_trigger=np.asarray(d["init_trigger"], dtype=np.float64),
            range_min=np.asarray(d["range_min"], dtype=np.float64),
            range_max=np.asarray(d["range_max"], dtype=np.float64),
            interp_lambda=float(d["interp_lambda"]),
            seq_len=int(d["seq_len"]),
            pred_len=int(d["pred_len"]),
            trace=dict(d.get("trace", {})),
        )
        plan.validate()
        return plan


class AttackPlanner:
    def __init__(self, config: AttackConfig, scorer: InfluenceScorer):
        self.config = config
        self.scorer = scorer
        self.targets = TargetChannelResolver(config.target_channels, config.target_channel_ratio)
        self.pattern = get_pattern(config.pattern.shape)
        self.channel_selector = PoisonChannelSelector(
            config.poison_channel_ratio, config.exclude_target_from_poison_channels
        )
        self.budget = get_budget_protocol(config.budget_protocol)
        self.sample_selector = HybridSampleSelector(config.poison_ratio, config.similar_ratio)

    def plan(self, dataset: PreparedDataset, seed: int) -> AttackPlan:
        target_idx, _ = self.targets.resolve(dataset.channels)
        influence = self.scorer.score(dataset.train_norm(), dataset.seq_len, target_idx, seed)
        return self.build(dataset, influence)

    def build(self, dataset: PreparedDataset, influence: InfluenceResult) -> AttackPlan:
        """Dựng kế hoạch từ điểm ảnh hưởng đã có (dùng trực tiếp trong kiểm thử và các biến thể bỏ thành phần)."""
        cfg = self.config
        seq_len = dataset.seq_len
        norm = dataset.norm
        b_s, b_e = dataset.bounds.train
        train_starts = dataset.starts["train"]

        # V_t và p
        target_idx, target_note = self.targets.resolve(dataset.channels)
        pattern = self.pattern.build(cfg.pattern.length, cfg.pattern.amplitude, cfg.pattern.baseline, len(target_idx))

        # V_p và ngân sách
        scores = np.asarray(influence.scores, dtype=np.float64)
        poison_idx = self.channel_selector.select(list(scores), target_idx)
        seg_lens = self.budget.split(cfg.trigger_total_len, len(poison_idx))

        # delta* (Thuật toán 3): L = ceil(t_TGR / |V_p|), và đoạn phải nằm trọn trong đầu vào
        min_lag = max(ceil_div(cfg.trigger_total_len, len(poison_idx)), max(seg_lens))
        lag_starts = np.arange(b_s, b_e - seq_len + 1)      # mọi cửa sổ đầu vào nằm trong phần huấn luyện
        estimator = LagEstimator(seq_len, cfg.lag.min_corr_len)
        lags: List[int] = []
        lag_scores: Dict[str, Dict[str, float]] = {}
        for ch in poison_idx:
            est = estimator.estimate(norm, ch, target_idx, lag_starts, min_lag)
            lags.append(est.lag)
            lag_scores[dataset.channels[ch]] = {str(k): float(v) for k, v in est.scores.items()}

        offsets = [seq_len - lag for lag in lags]

        # T_poison (Thuật toán 1)
        distances = self.sample_selector.distances(norm, train_starts, dataset.window, target_idx, pattern)
        selection = self.sample_selector.select(distances)

        # tau_0 (Thuật toán 4)
        best_pos = int(np.lexsort((np.arange(len(distances)), distances))[0])
        init_start = int(train_starts[best_pos])
        init_trigger = self._initial_trigger(norm, init_start + seq_len, poison_idx, seg_lens, lags, (b_s, b_e))

        train_rows = dataset.train_norm()
        plan = AttackPlan(
            channels=list(dataset.channels),
            target_idx=list(target_idx),
            poison_idx=list(poison_idx),
            seg_lens=list(seg_lens),
            lags=lags,
            offsets=offsets,
            pattern=pattern,
            poison_starts=train_starts[selection.all_pos],
            similar_starts=train_starts[selection.similar_pos],
            different_starts=train_starts[selection.different_pos],
            init_start=init_start,
            init_trigger=init_trigger,
            range_min=train_rows[:, poison_idx].min(axis=0),
            range_max=train_rows[:, poison_idx].max(axis=0),
            interp_lambda=float(cfg.interp_lambda),
            seq_len=seq_len,
            pred_len=dataset.pred_len,
            trace={
                "target_note": target_note,
                "influence": [float(v) for v in scores],
                "influence_history": list(influence.history),
                "lag_scores": lag_scores,
                "sample_selection": selection.info,
                "init_distance": float(distances[best_pos]),
                "min_lag": int(min_lag),
                "trigger_total_requested": int(cfg.trigger_total_len),
                "trigger_total_actual": int(sum(seg_lens)),
                "budget_protocol": self.budget.name,
                "pattern_shape": self.pattern.name,
            },
        )
        plan.validate()
        return plan

    @staticmethod
    def _initial_trigger(norm: np.ndarray, label_start: int, poison_idx: Sequence[int], seg_lens: Sequence[int],
                         lags: Sequence[int], train_span: Tuple[int, int]) -> np.ndarray:
        """TGR_i = X^{v_i}[t* - delta_i : t* - delta_i + len_i], với t* = đầu nhãn của cửa sổ gần mục tiêu nhất."""
        b_s, b_e = train_span
        pieces = []
        for ch, seg_len, lag in zip(poison_idx, seg_lens, lags):
            s = label_start - lag
            e = s + seg_len
            if s < b_s or e > b_e:
                raise PlanError("Đoạn khởi tạo trigger nằm ngoài phần huấn luyện.")

            pieces.append(norm[s:e, ch])

        return np.concatenate(pieces)
