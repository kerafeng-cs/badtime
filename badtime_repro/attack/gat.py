"""Chấm điểm ảnh hưởng kênh bằng GAT (Thuật toán 2, Công thức 4-5 của v2).

Nguồn:
- Đồ thị có hướng đầy đủ trên |V| kênh, láng giềng vào của j gồm cả chính j.
- Một tầng GAT, n_head = 4, chuỗi được z-score trên phần huấn luyện.
- Nhiệm vụ phụ trợ: dự báo bước kế tiếp trên các kênh đích V_t.
- alpha_ij lấy trung bình theo head; Influence(i) = sum_{j in V_t} mean alpha_ij.

ĐỀ XUẤT của dự án (A09), vì bài báo không nêu:
- Đặc trưng nút là cửa sổ seq_len bước của kênh (thay cho cả chuỗi H[i, :]), và nhãn là giá trị bước kế tiếp.
- Hidden 64 mỗi head; nối các head, ELU, Linear(heads * hidden -> 1); MSE; Adam lr 1e-3; 20 epoch; batch 256.
- Attention được lấy trung bình trên mọi cửa sổ huấn luyện.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.schema import GatConfig
from .channels import InfluenceResult, InfluenceScorer


class GraphAttentionNextStep(nn.Module):
    def __init__(self, in_len: int, hidden_dim: int, heads: int, leaky_slope: float, self_loops: bool):
        super().__init__()
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.w = nn.Linear(in_len, heads * hidden_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, hidden_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, hidden_dim))
        self.head = nn.Linear(heads * hidden_dim, 1)
        self.leaky_slope = leaky_slope
        self.self_loops = self_loops
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def attention(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """h: [B, N, L]. Trả về alpha [B, heads, N_dst(j), N_src(i)] (tổng theo i bằng 1) và W h [B, heads, N, H]."""
        b, n, _ = h.shape
        wh = self.w(h).view(b, n, self.heads, self.hidden_dim).permute(0, 2, 1, 3)
        src = (wh * self.a_src[None, :, None, :]).sum(-1)  # a^T phần W h_i (nguồn)
        dst = (wh * self.a_dst[None, :, None, :]).sum(-1)  # a^T phần W h_j (đích)
        e = F.leaky_relu(dst[:, :, :, None] + src[:, :, None, :], self.leaky_slope)
        if not self.self_loops:
            eye = torch.eye(n, dtype=torch.bool, device=h.device)
            e = e.masked_fill(eye[None, None], float("-inf"))

        return torch.softmax(e, dim=-1), wh

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha, wh = self.attention(h)
        z = torch.matmul(alpha, wh)                      # [B, heads, N, H]
        z = z.permute(0, 2, 1, 3).reshape(h.shape[0], h.shape[1], -1)
        return self.head(F.elu(z)).squeeze(-1), alpha   # [B, N]


class GatInfluenceScorer(InfluenceScorer):
    def __init__(self, config: GatConfig, device: torch.device):
        self.config = config
        self.device = device

    def score(self, train_norm: np.ndarray, seq_len: int, target_idx: Sequence[int], seed: int) -> InfluenceResult:
        cfg = self.config
        torch.manual_seed(seed)
        gen = torch.Generator()
        gen.manual_seed(seed)

        xs, ys = self._next_step_samples(train_norm, seq_len)
        n = xs.shape[0]
        net = GraphAttentionNextStep(seq_len, cfg.hidden_dim, cfg.heads, cfg.leaky_slope, cfg.self_loops)
        net = net.to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
        tgt = torch.as_tensor(list(target_idx), dtype=torch.long, device=self.device)

        history = []
        for epoch in range(cfg.epochs):
            net.train()
            perm = torch.randperm(n, generator=gen)
            total = 0.0
            for s in range(0, n, cfg.batch_size):
                idx = perm[s:s + cfg.batch_size]
                pred, _ = net(xs[idx].to(self.device))
                loss = F.mse_loss(pred[:, tgt], ys[idx].to(self.device)[:, tgt])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss) * len(idx)

            history.append({"epoch": epoch + 1, "train_mse_next_step": total / n})

        return InfluenceResult(self._influence(net, xs, tgt), history)

    @staticmethod
    def _next_step_samples(train_norm: np.ndarray, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Đầu vào [s, s + L) của từng kênh, nhãn là giá trị tại s + L. Trả về xs [n, C, L], ys [n, C]."""
        series = torch.as_tensor(train_norm, dtype=torch.float32)
        n = series.shape[0] - seq_len
        rows = torch.arange(n)[:, None] + torch.arange(seq_len)[None, :]
        xs = series[rows].permute(0, 2, 1).contiguous()
        ys = series[seq_len:seq_len + n]
        return xs, ys

    def _influence(self, net: GraphAttentionNextStep, xs: torch.Tensor, tgt: torch.Tensor) -> np.ndarray:
        net.eval()
        acc = torch.zeros(xs.shape[1], dtype=torch.float64)
        with torch.no_grad():
            for s in range(0, xs.shape[0], 1024):
                alpha, _ = net.attention(xs[s:s + 1024].to(self.device))
                mean_alpha = alpha.mean(dim=1)                 # trung bình theo head: [B, j, i]
                acc += mean_alpha[:, tgt, :].sum(dim=1).sum(dim=0).double().cpu()

        return (acc / xs.shape[0]).numpy()
