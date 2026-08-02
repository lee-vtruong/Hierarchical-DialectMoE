# Kho lưu trữ kết quả thực nghiệm

Thư mục này lưu các artifact nhỏ đã được tải từ server về máy local trước khi
server bị xóa. Mục tiêu là giữ lại dữ liệu nguồn cho các bảng và kết luận trong
`BAO_CAO_THUC_NGHIEM.md`, thay vì chỉ giữ số liệu đã chép vào báo cáo.

## Bản phục hồi ngày 2026-08-02

`recovered_2026-08-02/` chứa 57 artifact CSV/JSON, tổng dung lượng khoảng 200 KB:

| Nhóm | Số file | Nội dung |
|---|---:|---|
| `audit` | 4 | Audit overlap, conflict và speaker-disjoint split |
| `h3` | 8 | Router input, aggregate và paired comparison |
| `h4` | 10 | Load-balancing sweep và validation comparison |
| `h5` | 9 | Pitch/energy, spectral, handcrafted và multi-seed |
| `h6` | 5 | Repaired speaker-disjoint test |
| `h7` | 6 | Province analysis, confusion và calibration |
| `h8` | 4 | Duration, confidence và focus provinces |
| `h9` | 4 | Temperature scaling |
| `h10` | 7 | Multi-crop summary và paired comparisons |

`SHA256SUMS.txt` chứa checksum của toàn bộ 57 artifact. Trên Linux, kiểm tra bằng:

```bash
cd results_archive/recovered_2026-08-02
sha256sum --check SHA256SUMS.txt
```

## Phần không thể phục hồi từ máy local

- Checkpoint huấn luyện (`.pt`).
- Prediction-level JSONL đầy đủ.
- Log huấn luyện và log GPU.
- Dataset/cache model trên server.
- Output chi tiết ban đầu của H1/H2 không được tải xuống; các metric và kết luận
  quan trọng của chúng vẫn được giữ trong `BAO_CAO_THUC_NGHIEM.md`.
- Kết quả H11 chưa được xác nhận đã hoàn tất trước khi server bị xóa.

Các artifact trong kho này là dữ liệu báo cáo, không đủ để tái đánh giá checkpoint
mà không huấn luyện lại. Không sửa trực tiếp file đã phục hồi; nếu cần xử lý tiếp,
hãy ghi output mới sang một thư mục khác để giữ nguyên checksum gốc.
