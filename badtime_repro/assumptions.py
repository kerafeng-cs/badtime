"""Sổ giả định đang áp dụng trong mã. Được ghi vào hồ sơ (profile.json) của MỌI lần chạy.

status: "source" = có trong bài báo v2 (2508.04189v2.pdf); "proposal" = đề xuất của dự án;
"unknown" = bài báo không xác định, lựa chọn hiện tại ảnh hưởng trực tiếp con số.
`config` trỏ tới trường cấu hình điều khiển giả định đó (nếu có).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

STATUSES = ("source", "proposal", "unknown")


@dataclass(frozen=True)
class Assumption:
    id: str
    topic: str
    choice: str
    status: str
    config: Optional[str] = None
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"{self.id}: status không hợp lệ {self.status}")

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {k: v for k, v in asdict(self).items() if v is not None}


_RAW = [
    {
        "id": "A01",
        "topic": "Phiên bản thư viện nền",
        "choice": (
            "TimesNet vendored từ thuml/Time-Series-Library@4e938a17 (18/04/2026); siêu tham số theo "
            "scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh"
        ),
        "status": "proposal",
        "config": "timesnet",
        "note": "v2 mục V-A chỉ nói dùng kiến trúc và siêu tham số huấn luyện của TSLib, không nêu commit.",
    },
    {
        "id": "A02",
        "topic": "Phân chia dữ liệu",
        "choice": (
            "6/2/2 trên toàn chuỗi 17420 hàng: train [0, 10452), val [10452, 13936), test [13936, 17420). Ranh giới "
            "12/4/4 tháng của TSLib là cấu hình đối chiếu riêng."
        ),
        "status": "source",
        "config": "data.split_protocol",
        "note": (
            "v2 Bảng XI: ETTh1 17,420 mốc, tỉ lệ 6/2/2. Ranh giới cụ thể là do dự án tính (int(0.6N), int(0.2N))."
        ),
    },
    {
        "id": "A03",
        "topic": "Lịch sử từ phân vùng trước",
        "choice": (
            "Cửa sổ val/test được dùng tối đa seq_len bước lịch sử thuộc phân vùng trước; nhãn luôn nằm trọn trong "
            "phân vùng."
        ),
        "status": "proposal",
        "config": "data.allow_history_from_previous_split",
    },
    {
        "id": "A04",
        "topic": "Chuẩn hóa",
        "choice": (
            "Chuẩn hóa từng kênh bằng mean/std (ddof=0) của phần huấn luyện; báo cáo cả thang chuẩn hóa và đơn vị "
            "gốc."
        ),
        "status": "proposal",
    },
    {
        "id": "A05",
        "topic": "Kênh đích",
        "choice": "OT. alpha_S = 0.1 x 7 kênh = 0.7, làm tròn thành 1 kênh. Tên kênh được nêu tường minh.",
        "status": "unknown",
        "config": "attack.target_channels",
        "note": "v2 không nêu tên kênh đích của ETTh1.",
    },
    {
        "id": "A06",
        "topic": "Công thức số của Hump",
        "choice": (
            "Nội suy tuyến tính 7 điểm 'cone' của BackTime [0.1278 ... 0.3853 ... 0.1278] lên 53 điểm; giá trị tuyệt "
            "đối ở thang chuẩn hóa, amplitude 1, baseline 0."
        ),
        "status": "unknown",
        "config": "attack.pattern",
        "note": (
            "v2 Bảng X dùng Hump/Upward/Up-and-down, tương ứng cone/up_trend/up_and_down của BackTime, nhưng không "
            "cho số. Có thể chọn 'hump_sine' làm dạng thay thế. Biên độ ảnh hưởng trực tiếp MAE_pa."
        ),
    },
    {
        "id": "A07",
        "topic": "Làm tròn số mẫu và số kênh",
        "choice": (
            "k_sim = floor(gamma alpha_T |D|) = 307, k_diff = floor((1-gamma) alpha_T |D|) = 718; số kênh bị sửa "
            "floor(0.3 x 7) = 2."
        ),
        "status": "source",
        "config": "attack.poison_ratio / similar_ratio / poison_channel_ratio",
        "note": (
            "Floor theo Thuật toán 1 dòng 7-8. TopK alpha_P|V| (Thuật toán 2) dùng floor là lựa chọn của dự án."
        ),
    },
    {
        "id": "A08",
        "topic": "Chia ngân sách trigger",
        "choice": (
            "keep_total: 29 = 15 + 14, đoạn dài hơn cho kênh có ảnh hưởng cao hơn. Cấu hình thay thế: equal_segments "
            "14 + 14 = 28, ceil_segments 15 + 15 = 30."
        ),
        "status": "unknown",
        "config": "attack.budget_protocol",
        "note": "v2 viết độ dài đoạn t_TGR/|V_p| = 14.5, không nguyên.",
    },
    {
        "id": "A09",
        "topic": "Huấn luyện GAT",
        "choice": (
            "Nguồn: 1 tầng, 4 head, self-loop, z-score theo phần huấn luyện, dự báo bước kế tiếp trên V_t, attention "
            "trung bình theo head. Đề xuất: đặc trưng nút là cửa sổ 96 bước; hidden 64; Adam lr 1e-3; 20 epoch; batch"
            " 256; LeakyReLU 0.2."
        ),
        "status": "proposal",
        "config": "attack.gat",
    },
    {
        "id": "A10",
        "topic": "Kênh bị sửa có thể trùng kênh đích",
        "choice": "Cho phép (không loại kênh đích khỏi ứng viên).",
        "status": "source",
        "config": "attack.exclude_target_from_poison_channels",
        "note": "v2 Observation 2: V_p thường gồm cả kênh đích.",
    },
    {
        "id": "A11",
        "topic": "Tính độ trễ",
        "choice": (
            "Nguồn (Thuật toán 3): delta trong [ceil(t_TGR/|V_p|), t_IN] = [15, 96]; cộng tương quan Pearson qua mọi "
            "cửa sổ và mọi kênh đích. Đề xuất: cửa sổ là mọi đầu vào nằm trong phần huấn luyện; bỏ delta có t_IN - "
            "delta < 8; bỏ cửa sổ có phương sai ~0."
        ),
        "status": "proposal",
        "config": "attack.lag.min_corr_len",
    },
    {
        "id": "A12",
        "topic": "Khoảng cách mục tiêu khi chọn mẫu và khởi tạo",
        "choice": (
            "Euclid thường giữa nhãn [0, t_pin) của kênh đích và p, ở thang chuẩn hóa. t* lấy trong các nhãn của cửa "
            "sổ huấn luyện."
        ),
        "status": "proposal",
        "note": "Euclid theo Thuật toán 1 và Công thức 8; thang đo là lựa chọn của dự án.",
    },
    {
        "id": "A13",
        "topic": "Phép tổng hợp mất mát và trọng số",
        "choice": (
            "Tổng trên phần tử của mỗi mẫu rồi chia số mẫu (đúng cách viết Công thức 16-18). alpha = (1,1,1), beta = "
            "(1,1,1,1)."
        ),
        "status": "unknown",
        "config": "attack.loss_reduction / model_loss_weights / trigger_loss_weights",
        "note": "v2: các trọng số đặt theo kinh nghiệm và có tăng trọng số L_p^{V_t}, nhưng không cho giá trị.",
    },
    {
        "id": "A14",
        "topic": "Phạm vi mất mát kênh đích",
        "choice": (
            "L_p^{V_t} tính trên toàn khoảng dự báo: [0, t_pin) so với p, [t_pin, P) so với nhãn gốc (Công thức 14, "
            "17)."
        ),
        "status": "source",
    },
    {
        "id": "A15",
        "topic": "Lịch huấn luyện",
        "choice": (
            "Adam lr 1e-4, giảm 1/2 mỗi epoch (TSLib type1), batch 32, 10 epoch. Clean dừng sớm với patience 3 theo "
            "MSE val; BadTime chạy đủ 10 epoch."
        ),
        "status": "proposal",
        "config": "train",
    },
    {
        "id": "A16",
        "topic": "Cập nhật trigger",
        "choice": (
            "Nguồn: mỗi M = 2 epoch, cố định mô hình, duyệt các mẫu bị chọn, eps làm trơn 0.8. Đề xuất: sau các epoch"
            " có (epoch+1) % 2 == 0; 1 lượt; Adam lr 0.01 giữ trạng thái; batch 32; mô hình ở chế độ eval."
        ),
        "status": "proposal",
        "config": "attack.trigger_*",
    },
    {
        "id": "A17",
        "topic": "Chọn cặp trạng thái",
        "choice": (
            "Cặp (mô hình, trigger) cuối mỗi epoch. Ứng viên: MAE_c(val) <= 1.05 x min MAE_c(val); trong đó chọn "
            "MAE_pa(val) nhỏ nhất."
        ),
        "status": "proposal",
        "config": "attack.state_selection.clean_tolerance",
    },
    {
        "id": "A18",
        "topic": "Loss của nhánh sạch",
        "choice": "MSE như TSLib (mặc định). Dùng mse_mae để kiểm tra đường xử lý so với BadTime đã vô hiệu hóa.",
        "status": "proposal",
        "config": "train.clean_loss",
    },
    {
        "id": "A19",
        "topic": "Đánh giá có trigger",
        "choice": "Gắn trigger vào MỌI cửa sổ hợp lệ của tập đánh giá; báo cả val và test.",
        "status": "proposal",
        "note": (
            "v2 không nói rõ tập dùng cho Bảng IV/V; với phát hiện bất thường thì nói 'poisoned validation set'."
        ),
    },
    {
        "id": "A20",
        "topic": "Thang đo của số bài báo",
        "choice": "Giả định số trong bài báo ở thang chuẩn hóa (như TSLib); chưa xác nhận.",
        "status": "unknown",
    },
    {
        "id": "A21",
        "topic": "Dự báo TimesNet phụ thuộc lô",
        "choice": (
            "FFT_for_Period lấy trung bình biên độ trên cả lô, nên dự báo phụ thuộc thành phần lô. Đánh giá theo thứ "
            "tự thời gian, lô 32, không xáo; mọi nhánh dùng cùng quy tắc."
        ),
        "status": "proposal",
        "config": "train.eval_batch_size",
    },
]

ASSUMPTIONS: Tuple[Assumption, ...] = tuple(Assumption(**d) for d in _RAW)
