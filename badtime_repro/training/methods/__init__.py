"""Cài đặt phương pháp huấn luyện. Tên khớp với `catalog.CATALOG.methods`.

Kế thừa:
    TrainingMethod (base)
    ├── CleanMethod                 Benign
    └── PoisoningMethod             khung đầu độc + tối ưu xen kẽ mô hình/trigger
        └── BadTimeMethod
"""

from ...common.registry import LazyRegistry

TRAINING_METHODS = LazyRegistry("phương pháp")
TRAINING_METHODS.register("clean", "badtime_repro.training.methods.clean:CleanMethod")
TRAINING_METHODS.register("badtime", "badtime_repro.training.methods.badtime:BadTimeMethod")
