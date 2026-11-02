"""Tầng tấn công.

Các module numpy (patterns, budget, channels, sample_selection, lag, plan, injection) không phụ thuộc torch.
Các module torch (trigger, objectives, gat) chỉ được import khi huấn luyện.
"""
