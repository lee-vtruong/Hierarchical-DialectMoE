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

## 7. Kết quả chính thức trên test set

Test set gồm 2.026 mẫu. Hai checkpoint được đánh giá:

- `best.pt`: epoch 5, được chọn theo validation loss nhỏ nhất.
- `last.pt`: epoch 10, checkpoint cuối trước early stopping.

### 7.1 So sánh tổng thể

| Checkpoint | Region Acc. | Region Balanced Acc. | Region Macro-F1 | Province Acc. | Province Balanced Acc. | Province Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Epoch 5 - best loss | 0,8909 | 0,8855 | 0,8874 | 0,4067 | 0,4109 | 0,3948 |
| Epoch 10 - last | **0,8954** | **0,8901** | **0,8913** | **0,4418** | **0,4437** | **0,4380** |

Checkpoint epoch 10 tốt hơn epoch 5 trên tất cả metrics tổng thể:

- Region accuracy tăng khoảng 0,44 điểm phần trăm.
- Region macro-F1 tăng khoảng 0,39 điểm phần trăm.
- Province accuracy tăng khoảng 3,50 điểm phần trăm.
- Province balanced accuracy tăng khoảng 3,27 điểm phần trăm.
- Province macro-F1 tăng khoảng 4,32 điểm phần trăm.

Vì vậy, nếu mục tiêu chính là classification, checkpoint epoch 10 nên được dùng
thay cho checkpoint được chọn chỉ theo validation loss.

### 7.2 Kết quả theo vùng của checkpoint epoch 10

| Vùng | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Central | 0,9107 | 0,8026 | 0,8532 | 623 |
| North | 0,9013 | 0,9566 | **0,9281** | 783 |
| South | 0,8746 | 0,9113 | 0,8926 | 620 |

North có F1 cao nhất và recall đạt 95,66%. Central có recall thấp nhất,
80,26%, dù precision đạt 91,07%. Mô hình bỏ sót Central nhiều hơn North và
South. Support của ba vùng tương đối cân bằng nên balanced accuracy gần
accuracy.

### 7.3 Kết quả cấp tỉnh của checkpoint epoch 10

Các province code có F1 cao:

| Province code | F1 |
|---:|---:|
| 35 | 0,9259 |
| 37 | 0,8193 |
| 98 | 0,7761 |
| 77 | 0,7463 |
| 15 | 0,7385 |

Các province code có F1 thấp:

| Province code | F1 |
|---:|---:|
| 68 | 0,0000 |
| 62 | 0,1111 |
| 94 | 0,1224 |
| 48 | 0,1538 |
| 24 | 0,1587 |

Province code 68 có precision, recall và F1 bằng 0 trên 32 mẫu test. Cần phân
tích confusion matrix để xác định các mẫu của code 68 bị dự đoán nhầm thành
tỉnh nào. Chênh lệch lớn giữa các tỉnh cho thấy accuracy tổng thể chưa đủ để
mô tả chất lượng hệ thống.

### 7.4 Phân tích routing và expert collapse

Phân bố expert trung bình:

| Checkpoint | Expert 0 | Expert 1 | Expert 2 | Expert 3-7 | Mean entropy |
|---|---:|---:|---:|---:|---:|
| Epoch 5 | ~0,000004 | 0,8803 | 0,1197 | mỗi expert ~0,000004 | 0,00339 |
| Epoch 10 | ~0,000002 | 0,8822 | 0,1178 | mỗi expert ~0,000002 | 0,00279 |

Router gần như chỉ sử dụng hai trong tám expert:

- Expert 1 nhận khoảng 88% xác suất.
- Expert 2 nhận khoảng 12% xác suất.
- Sáu expert còn lại gần như không được dùng.
- Entropy gần 0 cho thấy routing rất tự tin nhưng đã collapse.

Do đó, lần chạy này **chưa chứng minh được expert specialization lành mạnh**.
MoE hiện hoạt động gần giống mô hình hai expert thay vì tám expert.

Nguyên nhân có khả năng:

1. `router_weight` đang khuyến khích entropy thấp, làm router tự tin quá sớm.
2. `load_balance_weight = 0.01` quá nhỏ để chống collapse.
3. Hard top-k assignment làm các expert ít được chọn khó nhận gradient hữu ích.
4. Chưa có warm-up hoặc noisy routing để phân phối dữ liệu lúc đầu.

Các thí nghiệm tiếp theo cần:

- Đặt `router_weight = 0` trước.
- Tăng `load_balance_weight`, ví dụ 0,05; 0,1 và 0,2.
- Thử 2 và 4 expert để so sánh với 8 expert.
- Theo dõi expert usage ở từng epoch.
- Thử temperature/noisy routing hoặc router warm-up.
- Báo cáo province-to-expert và region-to-expert matrix.

### 7.5 Kết luận test set

Kết quả tốt nhất hiện tại, sử dụng checkpoint epoch 10:

- Region accuracy: **89,54%**.
- Region macro-F1: **89,13%**.
- Province accuracy: **44,18%**.
- Province macro-F1: **43,80%**.

Đây là kết quả test set thực tế của hệ thống MVP. Kết quả chứng minh pipeline có
thể học tốt phân loại ba vùng và có khả năng phân biệt 63 tỉnh ở mức đáng kể.
Tuy nhiên, chưa thể tuyên bố MoE tốt hơn baseline hoặc các expert đã chuyên môn
hóa, do chưa chạy ablation và router đang bị expert collapse.

## 8. Ablation A1 - Acoustic-only baseline

Cấu hình `configs/experiments/acoustic_only.yaml` sử dụng:

- Wav2Vec2 acoustic encoder.
- Region classification head.
- Province classification head.
- Không sử dụng prosody.
- Không sử dụng hierarchical routing.
- Không sử dụng Mixture-of-Experts.

Checkpoint có province validation accuracy tốt nhất được đánh giá trên cùng
test set 2.026 mẫu.

### 8.1 Kết quả acoustic-only

| Metric | Acoustic-only |
|---|---:|
| Region accuracy | 0,8929 |
| Region balanced accuracy | 0,8868 |
| Region macro-F1 | 0,8882 |
| Province accuracy | 0,3786 |
| Province balanced accuracy | 0,3801 |
| Province macro-F1 | 0,3763 |

Kết quả theo vùng:

| Vùng | Precision | Recall | F1 |
|---|---:|---:|---:|
| Central | 0,9245 | 0,7865 | 0,8500 |
| North | 0,8956 | 0,9642 | 0,9287 |
| South | 0,8637 | 0,9097 | 0,8861 |

Province code 68 tiếp tục có F1 bằng 0, cho thấy đây không chỉ là lỗi riêng của
MoE mà là class đặc biệt khó đối với cả acoustic-only baseline.

### 8.2 So sánh acoustic-only với MVP Hierarchical MoE

| Metric | Acoustic-only | MVP MoE epoch 10 | Chênh lệch tuyệt đối |
|---|---:|---:|---:|
| Region accuracy | 0,8929 | **0,8954** | +0,0025 |
| Region balanced accuracy | 0,8868 | **0,8901** | +0,0033 |
| Region macro-F1 | 0,8882 | **0,8913** | +0,0031 |
| Province accuracy | 0,3786 | **0,4418** | **+0,0632** |
| Province balanced accuracy | 0,3801 | **0,4437** | **+0,0636** |
| Province macro-F1 | 0,3763 | **0,4380** | **+0,0617** |

MVP MoE cải thiện hơn sáu điểm phần trăm ở bài toán cấp tỉnh, trong khi cải
thiện ở bài toán ba vùng chỉ khoảng 0,25-0,33 điểm phần trăm. Đây là bằng chứng
ban đầu rằng phần mở rộng sau acoustic encoder đặc biệt hữu ích cho phân loại
fine-grained.

Tuy nhiên, MVP MoE đồng thời thêm cả:

- Prosody features.
- Gated fusion.
- Region-conditioned routing.
- Mixture-of-Experts.
- Số lượng tham số bổ sung.

Vì vậy, phép so sánh này **chưa xác định được thành phần nào tạo ra mức tăng
6,32 điểm phần trăm province accuracy**. Cần tiếp tục chạy
`acoustic_prosody.yaml`, flat MoE và balanced hierarchical MoE.

### 8.3 Lưu ý về routing metrics của acoustic-only

File đánh giá vẫn chứa router probabilities và entropy 2,0202. Các giá trị này
không có ý nghĩa đối với acoustic-only vì `use_moe: false`; router được khởi tạo
trong model để giữ cấu trúc code thống nhất nhưng output của nó không tham gia
tạo province prediction. Không so sánh routing entropy acoustic-only với MoE.

### 8.4 Kết luận tạm thời

Acoustic-only là baseline mạnh ở mức vùng nhưng yếu hơn rõ rệt ở mức tỉnh.
Kết quả hiện hỗ trợ việc tiếp tục nghiên cứu prosody/MoE cho fine-grained
dialect identification, nhưng chưa đủ để xác nhận riêng giả thuyết H1, H2 hay
H4.
