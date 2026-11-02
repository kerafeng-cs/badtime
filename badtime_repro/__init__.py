"""Tái lập độc lập BadTime (arXiv 2508.04189v2), slot PoC ETTh1 + TimesNet, 96 -> 96.

Các tầng (một module chỉ import cùng tầng hoặc tầng thấp hơn; tests/test_architecture.py kiểm tra tự động):

    9  cli, __main__                         giao diện dòng lệnh
    8  evaluation (controls, report, ...)    đối chứng 4 điều kiện, tính lại chỉ số, báo cáo
    7  experiment                            mặt tiền: kiểm tra -> dữ liệu -> huấn luyện
    6  training (Trainer, TrainingMethod)    vòng huấn luyện chung + các phương pháp          [torch]
    5  models                                mô hình dự báo (TimesNet vendored)               [torch]
    4  metrics, runs, support                hợp đồng phép đo, hiện vật lần chạy, kiểm tra hỗ trợ
    3  attack (plan, ... | trigger, gat)     khối thiết kế BadTime (numpy) | phần torch
    2  data                                  nguồn, phân chia, chuẩn hóa, cửa sổ
    1  catalog, config                       danh mục, schema cấu hình
    0  common, assumptions                   lỗi, registry, làm tròn, JSON; sổ giả định
"""

__version__ = "0.2.0"
