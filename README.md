# Thiết kế baseline tái lập BadTime

Trạng thái ngày 27/10/2026: **chưa tìm thấy mã nguồn BadTime công khai được xác nhận là chính thức**. Đây là bộ tài liệu thiết kế cho một bản tái lập độc lập. Đã có mã cho slot PoC trong `badtime_repro/` và dữ liệu `dataset/ETTh1/`, nhưng **chưa có kết quả thí nghiệm**; phần torch chưa được chạy kiểm tra. Xem [Bản triển khai PoC](docs/08-implementation.md).

Mục tiêu dài hạn là kiểm chứng các kết quả của [BadTime, bản v2](https://arxiv.org/abs/2508.04189v2). **Mục tiêu hiện tại đã chốt là thiết kế một PoC cho ETTh1 + TimesNet**, đầu vào 96 bước và dự báo 96 bước. Thiết kế hệ thống phải có chỗ tích hợp các lựa chọn còn lại, nhưng triển khai toàn bộ ma trận không phải điều kiện hoàn thành PoC. Không coi việc chạy được chương trình hoặc tạo được một hình dự báo tương tự là bằng chứng tái lập thành công.

## Cách đọc

| Tài liệu | Nội dung |
|---|---|
| [Kiểm tra mã nguồn](docs/01-code-search.md) | Nguồn đã kiểm tra, giới hạn tìm kiếm, phân biệt BadTime và BackTime |
| [Kiến trúc khái niệm](docs/02-conceptual-design.md) | Các thành phần, trách nhiệm, đầu vào và đầu ra |
| [Dữ liệu và giao thức](docs/03-data-protocol.md) | Phân chia thời gian, chuẩn hóa, cửa sổ và chống rò rỉ dữ liệu |
| [Kế hoạch thí nghiệm](docs/04-experiment-plan.md) | Baseline đầu tiên, đối chứng, mở rộng và thứ tự triển khai |
| [Điểm chưa xác định](docs/05-open-decisions.md) | Những quyết định phải làm rõ để kết quả có thể diễn giải |
| [Tiêu chí kiểm chứng](docs/06-validation-plan.md) | Kiểm tra tính đúng, báo cáo sai lệch và điều kiện kết luận |
| [Cấu hình và phạm vi hỗ trợ](docs/07-configuration-support.md) | Ánh xạ dataset sang thư mục, ma trận tổ hợp và thông báo chưa hỗ trợ |
| [Bản triển khai PoC](docs/08-implementation.md) | Cách chạy, ánh xạ thành phần sang mã, hiện vật, sổ giả định đang áp dụng |

## Phạm vi bản thiết kế

Slot PoC đã chốt là **ETTh1 với TimesNet, đầu vào 96 bước và dự báo 96 bước**. Cấu hình `config.dataset = ETTh1` ánh xạ tới thư mục `dataset/ETTh1/` bên trong dự án. Khi triển khai PoC, chỉ slot này được yêu cầu hoạt động; các slot khác có mô tả cấu hình và thư mục dự kiến, chưa cần bộ nạp hay mã mô hình. Chọn slot chưa triển khai phải báo rõ chưa hỗ trợ và dừng trước huấn luyện. Mốc tham chiếu của PoC là Bảng V và VII của bài báo.

Mọi thành phần được gọi bằng tên khái niệm dành cho người đọc. Ngoại lệ được chốt theo yêu cầu là cách chọn `config.dataset` và quy ước thư mục dữ liệu. Các trường cấu hình khác vẫn được mô tả theo khái niệm; chưa đặt tên biến nội bộ, hàm hoặc lớp. Slot PoC hiện mới là mục tiêu thiết kế, chưa phải mã đang hoạt động.

Ba trạng thái thông tin được dùng xuyên suốt:

- **Nguồn:** có vị trí đối chiếu trong tài liệu gốc.
- **Đề xuất:** lựa chọn thiết kế của dự án, chưa phải cài đặt đã xác nhận của tác giả.
- **Chưa xác định:** cần nguồn bổ sung hoặc thí nghiệm độ nhạy trước khi tuyên bố tái lập chính xác.

Hoàn thành bộ thiết kế này không đồng nghĩa bài báo đã được kiểm chứng. Việc triển khai và chạy thí nghiệm thuộc giai đoạn tiếp theo.

## Chạy nhanh

Không có GPU: dùng `notebooks/badtime_poc_colab_kaggle.ipynb` trên Colab hoặc Kaggle (xem [docs/08](docs/08-implementation.md)).

Máy có torch:

```bash
pip install -r requirements.txt
python -m pytest -q tests
python -m badtime_repro check
python -m badtime_repro train --method clean   --seeds 2021 2022 2023
python -m badtime_repro train --method badtime --seeds 2021 2022 2023
```

Các bước tiếp theo (đối chứng 4 điều kiện, báo cáo so với bài báo) có trong [docs/08](docs/08-implementation.md).
