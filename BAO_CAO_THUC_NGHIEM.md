# Báo cáo thực nghiệm Hierarchical DialectMoE

Tài liệu này ghi lại các thử nghiệm đã thực hiện, môi trường chạy, lỗi kỹ thuật
đã xử lý và kết quả thực tế. Không sử dụng kết quả smoke test làm kết quả
nghiên cứu chính thức.

## 1. Môi trường server

Thông tin đã xác nhận:

- Hệ điều hành: Linux server.
- Python: 3.10.12.
- OpenSSL: 3.0.2.
- GPU được chọn: NVIDIA A100-SXM4-80GB, GPU vật lý số 7.
- NVIDIA driver: 550.127.08.
- CUDA tối đa theo driver: 12.4.
- PyTorch: 2.5.1+cu124.
- Dataset: ViMD 102,56 giờ.
- Dung lượng ViMD sau reconstruction: khoảng 59,8 GB.
- Số file Parquet công bố: 103 train, 13 validation và 14 test.

GPU được cô lập cho tiến trình bằng:

```bash
export CUDA_VISIBLE_DEVICES=7
```

Trong PyTorch, GPU vật lý số 7 được ánh xạ thành `cuda:0`.

## 2. Các vấn đề kỹ thuật đã phát hiện và khắc phục

### 2.1 Python thiếu SSL

Python tự cài tại `/home/hvtham/python3.11` thiếu module `_ssl`, khiến `pip`
không thể truy cập HTTPS. Virtual environment được tạo lại bằng
`/usr/bin/python3` phiên bản 3.10.12 có OpenSSL.

### 2.2 PyTorch không tương thích driver

PyTorch 2.13.0+cu130 yêu cầu CUDA 13.0, trong khi driver server hỗ trợ CUDA
12.4. Đã thay bằng PyTorch 2.5.1+cu124.

### 2.3 Transformers từ chối `pytorch_model.bin`

Transformers mới chặn `torch.load` với PyTorch dưới 2.6 do vấn đề bảo mật.
Backbone được sửa để bắt buộc tải checkpoint `safetensors`:

```python
AutoModel.from_pretrained(..., use_safetensors=True)
```

### 2.4 TorchCodec không tương thích

TorchCodec được cài tự động không tương thích PyTorch 2.5.1 và tìm thư viện
CUDA 13 (`libnvrtc.so.13`). Pipeline được thay đổi để:

- Lấy bytes âm thanh trực tiếp từ Parquet.
- Decode bằng `soundfile`.
- Resample bằng `librosa`.
- Không phụ thuộc TorchCodec/FFmpeg/CUDA trong DataLoader.

### 2.5 Lỗi dtype trong sparse MoE

Khi dùng automatic mixed precision:

- Expert activation có dtype FP16.
- Router weight sau softmax có dtype FP32.

`index_add_` không cho phép cộng hai dtype khác nhau. Routing weight hiện được
ép về dtype của expert output trước khi sparse aggregation. Regression test cho
trường hợp FP16/FP32 đã được bổ sung.

### 2.6 Cảnh báo Wav2Vec2 `UNEXPECTED`

Các key sau thuộc phần quantizer/projector của pretraining checkpoint:

```text
quantizer.weight_proj.*
quantizer.codevectors
project_hid.*
project_q.*
```

Hệ thống sử dụng `Wav2Vec2Model` làm acoustic encoder, không sử dụng head
pretraining. Vì vậy các key này không được nạp và cảnh báo có thể bỏ qua.

## 3. Smoke test

Cấu hình:

```bash
python scripts/train.py \
  --config configs/vimd_moe.yaml \
  --max-samples 32
```

Kết quả sau 20 epoch:

- Validation loss: 0,1449.
- Region accuracy: 1,0000.
- Province accuracy: 1,0000.

Kết quả trên chỉ xác nhận pipeline chạy end-to-end. Do `--max-samples 32` lấy
các bản ghi đầu tiên thay vì lấy mẫu phân tầng, subset có thể chỉ chứa một vùng
hoặc một tỉnh. Không được dùng accuracy 100% này để so sánh mô hình.

## 4. Full training trên ViMD

Thiết lập chính:

- Train: 15.023 mẫu.
- Validation: 1.900 mẫu.
- Batch size: 4.
- Số batch mỗi epoch: 3.756.
- Tốc độ quan sát: khoảng 8,37-8,52 batch/giây.
- Thời gian mỗi epoch: khoảng 7 phút 21 giây đến 7 phút 28 giây.
- Early stopping patience: 5.
- Tiêu chí lưu `best.pt`: validation loss nhỏ nhất.

### 4.1 Kết quả từng epoch

| Epoch | Validation loss | Region accuracy | Province accuracy |
|---:|---:|---:|---:|
| 1 | 3,6671 | 0,7579 | 0,1058 |
| 2 | 2,7583 | 0,8426 | 0,3016 |
| 3 | 2,4186 | 0,8768 | 0,3595 |
| 4 | 2,1464 | 0,8879 | 0,4426 |
| 5 | **2,1006** | 0,8942 | 0,4611 |
| 6 | 2,1156 | 0,8858 | 0,4716 |
| 7 | 2,1315 | **0,8989** | 0,4826 |
| 8 | 2,3066 | 0,8932 | 0,4674 |
| 9 | 2,3416 | 0,8921 | 0,4932 |
| 10 | 2,4410 | 0,8942 | **0,4942** |

Training dừng sau epoch 10 do validation loss không cải thiện trong 5 epoch
liên tiếp sau epoch 5.

### 4.2 Kết quả tốt nhất theo từng tiêu chí

- Validation loss tốt nhất: **2,1006 tại epoch 5**.
- Region accuracy tốt nhất: **89,89% tại epoch 7**.
- Province accuracy tốt nhất: **49,42% tại epoch 10**.

### 4.3 Nhận xét ban đầu

Region accuracy tăng nhanh và ổn định quanh 89%. Province accuracy tiếp tục
tăng sau khi validation loss bắt đầu xấu đi. Điều này cho thấy:

1. Bài toán 63 tỉnh khó hơn đáng kể so với bài toán ba vùng.
2. Sau epoch 5, mô hình có dấu hiệu giảm độ hiệu chỉnh xác suất hoặc tăng
   cross-entropy dù top-1 accuracy vẫn tăng.
3. Chọn checkpoint chỉ theo tổng validation loss chưa chắc phù hợp với mục
   tiêu province classification.
4. `best.pt` hiện tương ứng epoch 5, không tương ứng province accuracy cao
   nhất tại epoch 10.
5. Loss hiển thị ở batch cuối mỗi epoch dao động mạnh và không đại diện cho
   loss trung bình toàn epoch.

## 5. Việc cần đánh giá tiếp theo

### 5.1 Đánh giá `best.pt`

```bash
python scripts/evaluate.py \
  --config configs/vimd_moe.yaml \
  --checkpoint outputs/vimd_moe/best.pt \
  --split test
```

### 5.2 Đánh giá `last.pt`

```bash
python scripts/evaluate.py \
  --config configs/vimd_moe.yaml \
  --checkpoint outputs/vimd_moe/last.pt \
  --split test
```

Cần so sánh:

- Accuracy và balanced accuracy.
- Macro-F1 và weighted-F1.
- Kết quả từng vùng.
- Kết quả từng tỉnh.
- Router entropy.
- Phân bố sử dụng expert.

### 5.3 Cải thiện checkpointing

Training runner nên lưu riêng:

- `best_loss.pt`.
- `best_region_accuracy.pt`.
- `best_province_accuracy.pt`.
- `last.pt`.

Điều này tránh mất checkpoint có province accuracy tốt nhất khi validation
loss không còn cải thiện.

### 5.4 Thí nghiệm cần có để xác nhận đóng góp

Kết quả hiện tại chưa đủ chứng minh proposal. Cần tối thiểu:

1. Acoustic-only baseline.
2. Acoustic + prosody nhưng không MoE.
3. Flat MoE.
4. Hierarchical region-conditioned MoE.
5. Ablation F0, energy và spectral features.
6. So sánh số expert và top-k.
7. Chạy ít nhất ba seed và báo cáo mean ± standard deviation.

## 6. Trạng thái kết luận

Thử nghiệm hiện tại xác nhận hệ thống MVP có thể train end-to-end trên toàn bộ
ViMD và đạt:

- Gần 90% validation accuracy ở mức vùng.
- Gần 50% validation accuracy ở mức 63 tỉnh.

Đây là kết quả development/validation ban đầu, chưa phải test-set result và
chưa phải bằng chứng rằng Hierarchical MoE tốt hơn baseline. Chỉ đưa ra kết
luận nghiên cứu sau khi chạy test set, các baseline, ablation và nhiều seed.

