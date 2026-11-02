# Nguồn ETTh1.csv

- Kho: https://github.com/zhouhaoyi/ETDataset, commit `1d16c8f4f943005d613b5bc962e9eeb06058cf07`, file `ETT-small/ETTh1.csv`
- sha256: `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`
- 17420 hàng, 2016-07-01 00:00 → 2018-06-26 19:00, tần suất 1 giờ
- Cột: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
- Kiểm tra (`python -m badtime_repro check`): 0 giá trị thiếu, 0 mốc trùng, 0 bước không đều

File được đặt nguyên trạng, không sửa nội dung. Mỗi lần chạy ghi lại sha256 vào profile.json.
