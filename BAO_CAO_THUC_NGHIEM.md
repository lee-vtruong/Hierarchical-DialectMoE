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

## 9. Ablation A2 - Acoustic + prosody, không MoE

Cấu hình `configs/experiments/acoustic_prosody.yaml` sử dụng:

- Wav2Vec2 acoustic encoder.
- Bộ đặc trưng prosody thống kê.
- Gated fusion.
- Region và province classification heads.
- Không sử dụng hierarchical routing để tạo prediction.
- Không sử dụng Mixture-of-Experts.

### 9.1 Kết quả test

| Metric | Acoustic + prosody |
|---|---:|
| Region accuracy | 0,9003 |
| Region balanced accuracy | 0,8953 |
| Region macro-F1 | 0,8966 |
| Province accuracy | 0,4329 |
| Province balanced accuracy | 0,4365 |
| Province macro-F1 | 0,4268 |

Kết quả theo vùng:

| Vùng | Precision | Recall | F1 |
|---|---:|---:|---:|
| Central | 0,9075 | 0,8186 | 0,8608 |
| North | 0,9059 | 0,9591 | 0,9318 |
| South | 0,8866 | 0,9081 | 0,8972 |

### 9.2 Đóng góp của prosody

| Metric | Acoustic-only | Acoustic + prosody | Cải thiện |
|---|---:|---:|---:|
| Region accuracy | 0,8929 | **0,9003** | **+0,0074** |
| Region balanced accuracy | 0,8868 | **0,8953** | **+0,0085** |
| Region macro-F1 | 0,8882 | **0,8966** | **+0,0083** |
| Province accuracy | 0,3786 | **0,4329** | **+0,0543** |
| Province balanced accuracy | 0,3801 | **0,4365** | **+0,0564** |
| Province macro-F1 | 0,3763 | **0,4268** | **+0,0505** |

Prosody cải thiện:

- Khoảng 0,74-0,85 điểm phần trăm cho bài toán ba vùng.
- Khoảng 5,05-5,64 điểm phần trăm cho bài toán cấp tỉnh.

Đây là bằng chứng seed 42 hỗ trợ giả thuyết H1: prosody bổ sung thông tin hữu
ích, đặc biệt cho fine-grained dialect identification.

Province code 68 tăng từ F1 bằng 0 ở acoustic-only lên 0,3509 khi thêm prosody.
Điều này cho thấy các đặc trưng ngữ điệu có thể đặc biệt hữu ích cho một số
tỉnh mà acoustic pooled embedding chưa phân biệt được.

### 9.3 So sánh với MVP Hierarchical MoE

| Metric | Acoustic + prosody | MVP MoE epoch 10 | MoE - không MoE |
|---|---:|---:|---:|
| Region accuracy | **0,9003** | 0,8954 | -0,0049 |
| Region balanced accuracy | **0,8953** | 0,8901 | -0,0051 |
| Region macro-F1 | **0,8966** | 0,8913 | -0,0053 |
| Province accuracy | 0,4329 | **0,4418** | +0,0089 |
| Province balanced accuracy | 0,4365 | **0,4437** | +0,0072 |
| Province macro-F1 | 0,4268 | **0,4380** | +0,0112 |

MVP MoE chỉ tăng thêm khoảng 0,72-1,12 điểm phần trăm ở cấp tỉnh so với
acoustic + prosody, đồng thời giảm khoảng 0,49-0,53 điểm phần trăm ở cấp vùng.

Phần lớn mức tăng từ acoustic-only lên MVP MoE đến từ prosody:

- Tổng mức tăng province accuracy: +6,32 điểm phần trăm.
- Riêng prosody tạo mức tăng: +5,43 điểm phần trăm.
- Phần chênh lệch còn lại khi thêm hierarchical MoE: +0,89 điểm phần trăm.

Do MVP router bị collapse, chưa thể kết luận mức tăng còn lại là đóng góp ổn
định của MoE. Cần flat-MoE và balanced-MoE ablations.

### 9.4 Kết luận tạm thời về H1

H1 được hỗ trợ trong lần chạy seed 42:

> Đặc trưng prosody giúp cải thiện nhận diện phương ngữ tiếng Việt.

Tuy nhiên, để dùng ngôn ngữ “cải thiện có ý nghĩa thống kê”, cần chạy ít nhất
ba seed và bootstrap/paired test trên prediction-level outputs. Metrics tổng
hợp của một seed chưa đủ cho kiểm định thống kê.

## 10. Ablation MoE và hierarchical routing - seed 42

Các thí nghiệm MoE được chạy với `router_weight = 0` và
`load_balance_weight = 0.1` để khắc phục expert collapse của lần chạy đầu.

### 10.1 Bảng kết quả tổng hợp

| Mô hình | Region Acc. | Region Macro-F1 | Province Acc. | Province Balanced Acc. | Province Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Acoustic-only | 0,8929 | 0,8882 | 0,3786 | 0,3801 | 0,3763 |
| Acoustic + prosody | 0,9003 | 0,8966 | 0,4329 | 0,4365 | 0,4268 |
| Flat MoE-8 balanced | 0,8929 | 0,8888 | 0,4373 | 0,4392 | 0,4332 |
| Hierarchical MoE-4 balanced | 0,8973 | 0,8926 | 0,4348 | 0,4376 | 0,4350 |
| Hierarchical MoE-8 balanced | 0,9003 | 0,8968 | 0,4472 | 0,4483 | 0,4400 |
| **Hierarchical MoE-2 balanced** | **0,9008** | **0,8974** | **0,4526** | **0,4552** | **0,4486** |
| MVP MoE-8 collapse, epoch 10 | 0,8954 | 0,8913 | 0,4418 | 0,4437 | 0,4380 |

Hierarchical MoE-2 là cấu hình tốt nhất theo region accuracy, region macro-F1
và toàn bộ metrics cấp tỉnh. Hierarchical MoE-8 có region balanced accuracy
cao nhất, 0,8962, nhưng thấp hơn MoE-2 ở các metrics chính còn lại.

### 10.2 Mức cải thiện của cấu hình tốt nhất

So với acoustic-only, Hierarchical MoE-2 cải thiện:

- Region accuracy: +0,79 điểm phần trăm.
- Region macro-F1: +0,92 điểm phần trăm.
- Province accuracy: **+7,40 điểm phần trăm**.
- Province balanced accuracy: **+7,51 điểm phần trăm**.
- Province macro-F1: **+7,23 điểm phần trăm**.

So với acoustic + prosody không MoE:

- Region accuracy: +0,05 điểm phần trăm.
- Region macro-F1: +0,09 điểm phần trăm.
- Province accuracy: **+1,97 điểm phần trăm**.
- Province balanced accuracy: **+1,87 điểm phần trăm**.
- Province macro-F1: **+1,83 điểm phần trăm**.

So với MVP MoE-8 bị collapse:

- Region accuracy: +0,54 điểm phần trăm.
- Province accuracy: +1,09 điểm phần trăm.
- Province macro-F1: +1,06 điểm phần trăm.

### 10.3 Đóng góp của hierarchical routing

So sánh cùng tám expert và top-k=2:

| Metric | Flat MoE-8 | Hierarchical MoE-8 | Cải thiện |
|---|---:|---:|---:|
| Region accuracy | 0,8929 | **0,9003** | +0,0074 |
| Region balanced accuracy | 0,8853 | **0,8962** | +0,0108 |
| Region macro-F1 | 0,8888 | **0,8968** | +0,0080 |
| Province accuracy | 0,4373 | **0,4472** | +0,0099 |
| Province balanced accuracy | 0,4392 | **0,4483** | +0,0091 |
| Province macro-F1 | 0,4332 | **0,4400** | +0,0068 |

Trong seed 42, region-conditioned hierarchical routing tốt hơn flat routing
trên toàn bộ metrics. Kết quả này hỗ trợ ban đầu cho giả thuyết H4.

### 10.4 Số lượng expert

Kết quả không tăng đơn điệu theo số expert:

| Số expert | Top-k | Province Acc. | Province Macro-F1 |
|---:|---:|---:|---:|
| 2 | 1 | **0,4526** | **0,4486** |
| 4 | 2 | 0,4348 | 0,4350 |
| 8 | 2 | 0,4472 | 0,4400 |

Hai expert cho kết quả tốt nhất. Có thể tám expert là quá nhiều so với khoảng
15.000 mẫu train, làm mỗi expert nhận ít tín hiệu hơn. Tuy nhiên, phép so sánh
đồng thời thay đổi cả `num_experts` và tỷ lệ `top_k/num_experts`; cần chạy thêm
MoE-4 top-1 và MoE-8 top-1 để tách hai yếu tố.

### 10.5 Routing sau khi chống collapse

| Mô hình | Phân bố xác suất expert | Entropy | Entropy cực đại |
|---|---|---:|---:|
| Flat MoE-8 | xấp xỉ 12,5% mỗi expert | 2,0788 | ln(8)=2,0794 |
| Hierarchical MoE-2 | xấp xỉ 50% mỗi expert | 0,6931 | ln(2)=0,6931 |
| Hierarchical MoE-4 | xấp xỉ 25% mỗi expert | 1,3829 | ln(4)=1,3863 |
| Hierarchical MoE-8 | xấp xỉ 12,5% mỗi expert | 2,0793 | ln(8)=2,0794 |

Load balancing 0,1 đã loại bỏ collapse hoàn toàn. Tuy nhiên, entropy gần cực
đại cho thấy router hiện gần như đồng đều trên từng mẫu, tức là chuyển từ một
cực đoan sang cực đoan khác:

- Lần đầu: routing quá tự tin và collapse.
- Balanced run: routing gần uniform và chưa thể hiện specialization rõ.

Do đó, kết quả balanced MoE tốt hơn nhưng vẫn chưa chứng minh expert
specialization. Cần đo top-1 assignment counts, region-to-expert matrix và
province-to-expert matrix. Cũng cần thử `load_balance_weight` trung gian như
0,02 và 0,05.

### 10.6 Kết luận ablation seed 42

Các kết quả hiện hỗ trợ:

- H1: prosody cải thiện đáng kể phân loại cấp tỉnh.
- H4: hierarchical routing tốt hơn flat routing với MoE-8 trong seed 42.
- MoE có thể tạo thêm khoảng 1,8-2,0 điểm phần trăm so với acoustic + prosody.

Chưa đủ bằng chứng cho:

- H2: expert tự động học được các nhóm phương ngữ chuyên biệt.
- H3: prosody-aware routing tốt hơn acoustic-only routing.
- Ý nghĩa thống kê và khả năng lặp lại qua seed.

Cấu hình ứng viên chính để chạy seed 43 và 44 là Hierarchical MoE-2 balanced.
Cấu hình đối chứng cần chạy nhiều seed là acoustic + prosody không MoE.

## 11. Kết quả đa seed - Acoustic + prosody và MoE-2

Hai cấu hình được chạy với seed 42, 43 và 44:

- Acoustic + prosody, không MoE.
- Hierarchical MoE-2 balanced, top-k=1.

Các con số dưới đây là mean ± sample standard deviation.

### 11.1 Bảng tổng hợp ba seed

| Metric | Acoustic + prosody | Hierarchical MoE-2 |
|---|---:|---:|
| Region accuracy | **0,9011 ± 0,0028** | 0,8927 ± 0,0073 |
| Region balanced accuracy | **0,8961 ± 0,0038** | 0,8879 ± 0,0068 |
| Region macro-F1 | **0,8974 ± 0,0034** | 0,8894 ± 0,0074 |
| Province accuracy | **0,4426 ± 0,0087** | 0,4408 ± 0,0172 |
| Province balanced accuracy | **0,4466 ± 0,0091** | 0,4442 ± 0,0175 |
| Province macro-F1 | **0,4365 ± 0,0084** | 0,4365 ± 0,0175 |

Trung bình ba seed, MoE-2 không cải thiện acoustic + prosody:

- Region accuracy giảm khoảng 0,84 điểm phần trăm.
- Region macro-F1 giảm khoảng 0,80 điểm phần trăm.
- Province accuracy giảm khoảng 0,18 điểm phần trăm.
- Province balanced accuracy giảm khoảng 0,24 điểm phần trăm.
- Province macro-F1 gần như bằng nhau, chênh dưới 0,01 điểm phần trăm.

### 11.2 Kết quả từng seed

| Seed | Acoustic+Prosody Province Acc. | MoE-2 Province Acc. | MoE - đối chứng |
|---:|---:|---:|---:|
| 42 | 0,4329 | **0,4526** | +0,0197 |
| 43 | 0,4452 | **0,4487** | +0,0035 |
| 44 | **0,4497** | 0,4210 | -0,0286 |

MoE-2 tốt hơn ở seed 42 và 43 nhưng giảm mạnh ở seed 44. Mức giảm seed 44 lớn
hơn tổng lợi ích ở hai seed còn lại, khiến mean thấp hơn đối chứng.

### 11.3 Độ ổn định

Standard deviation của MoE-2 lớn hơn rõ rệt:

- Province accuracy: 0,0172 so với 0,0087, gần gấp 2 lần.
- Province balanced accuracy: 0,0175 so với 0,0091.
- Province macro-F1: 0,0175 so với 0,0084, hơn gấp 2 lần.

MoE-2 hiện kém ổn định theo initialization seed. Kết quả tốt nhất seed 42
không đại diện cho hành vi trung bình.

### 11.4 Liên hệ với routing

Router entropy của MoE-2 ở cả ba seed gần như bằng `ln(2)`:

```text
seed 42: 0,6931446
seed 43: 0,6931431
seed 44: 0,6931350
```

Xác suất expert trung bình cũng gần 50/50. Router gần uniform trên từng mẫu,
chưa học được quyết định chuyên biệt rõ ràng. Với top-k=1, lựa chọn expert có
thể nhạy với các sai khác logit rất nhỏ, góp phần làm tăng variance giữa seed.

### 11.5 Điều chỉnh kết luận

Kết luận dựa riêng seed 42 rằng “MoE-2 tốt hơn acoustic + prosody” không còn
đứng vững khi xét ba seed.

Kết quả đa seed hiện tại:

- Prosody-only extension là lựa chọn ổn định hơn.
- MoE-2 chưa tạo cải thiện trung bình.
- H2 chưa được hỗ trợ.
- H4 mới chỉ có bằng chứng single-seed từ so sánh flat MoE-8 và hierarchical
  MoE-8; chưa có xác nhận đa seed.
- Chưa nên chọn MoE-2 làm mô hình cuối chỉ dựa trên checkpoint seed 42.

### 11.6 Thí nghiệm cần chạy tiếp

1. Chạy acoustic-only seed 43 và 44 để đánh giá H1 qua ba seed công bằng.
2. Bổ sung top-1 assignment counts thay vì chỉ mean probabilities.
3. Thử load balancing trung gian 0,02 và 0,05.
4. Thử MoE-2 với router warm-up hoặc temperature schedule.
5. Chạy flat/hierarchical MoE-8 thêm seed nếu tiếp tục đánh giá H4.
6. Lưu prediction-level outputs để bootstrap confidence interval và paired
   significance test.

## 12. Kết quả đa seed - Đánh giá giả thuyết H1

Acoustic-only và acoustic + prosody đều được chạy với seed 42, 43 và 44 trên
cùng train/validation/test split.

### 12.1 Mean ± standard deviation

| Metric | Acoustic-only | Acoustic + prosody | Chênh lệch mean |
|---|---:|---:|---:|
| Region accuracy | 0,8947 ± 0,0020 | **0,9011 ± 0,0028** | +0,0064 |
| Region balanced accuracy | 0,8894 ± 0,0024 | **0,8961 ± 0,0038** | +0,0067 |
| Region macro-F1 | 0,8907 ± 0,0023 | **0,8974 ± 0,0034** | +0,0067 |
| Province accuracy | 0,3944 ± 0,0147 | **0,4426 ± 0,0087** | **+0,0482** |
| Province balanced accuracy | 0,3973 ± 0,0163 | **0,4466 ± 0,0091** | **+0,0493** |
| Province macro-F1 | 0,3906 ± 0,0129 | **0,4365 ± 0,0084** | **+0,0459** |

Prosody cải thiện trung bình:

- Region accuracy: +0,64 điểm phần trăm.
- Region macro-F1: +0,67 điểm phần trăm.
- Province accuracy: **+4,82 điểm phần trăm**.
- Province balanced accuracy: **+4,93 điểm phần trăm**.
- Province macro-F1: **+4,59 điểm phần trăm**.

### 12.2 So sánh theo từng seed

| Seed | Acoustic-only Province Acc. | Acoustic+Prosody Province Acc. | Cải thiện |
|---:|---:|---:|---:|
| 42 | 0,3786 | **0,4329** | +0,0543 |
| 43 | 0,4077 | **0,4452** | +0,0375 |
| 44 | 0,3968 | **0,4497** | +0,0528 |

| Seed | Acoustic-only Province Macro-F1 | Acoustic+Prosody Province Macro-F1 | Cải thiện |
|---:|---:|---:|---:|
| 42 | 0,3763 | **0,4268** | +0,0505 |
| 43 | 0,4015 | **0,4410** | +0,0395 |
| 44 | 0,3940 | **0,4418** | +0,0478 |

Prosody tốt hơn acoustic-only ở cả ba seed, không phụ thuộc vào một
initialization thuận lợi.

### 12.3 Độ ổn định

Ở bài toán cấp tỉnh, acoustic + prosody có standard deviation thấp hơn:

- Province accuracy std giảm từ 0,0147 xuống 0,0087.
- Province balanced accuracy std giảm từ 0,0163 xuống 0,0091.
- Province macro-F1 std giảm từ 0,0129 xuống 0,0084.

Prosody không chỉ cải thiện mean mà còn làm kết quả cấp tỉnh ổn định hơn giữa
các seed trong thí nghiệm hiện tại.

### 12.4 Kết luận về H1

Kết quả ba seed cung cấp bằng chứng lặp lại hỗ trợ H1:

> Đặc trưng prosody cải thiện khả năng nhận diện phương ngữ tiếng Việt, đặc
> biệt ở mức 63 tỉnh.

Mức tăng province accuracy trung bình 4,82 điểm phần trăm và xuất hiện ở cả ba
seed. Đây là bằng chứng thực nghiệm mạnh hơn kết quả single-seed.

Prediction-level paired tests được trình bày trong mục tiếp theo. Test set đã
được dùng để so sánh nhiều cấu hình, vì vậy các hyperparameter mới phải được
chọn trên validation set và chỉ đánh giá test một lần sau khi chốt.

## 13. Kiểm định paired cho H1

Prediction-level outputs của acoustic-only và acoustic + prosody được ghép cặp
trên cùng 2.026 utterance thuộc 1.344 speaker.

Hai kiểm định được sử dụng:

- Speaker-level paired bootstrap, 10.000 lần lặp.
- Exact McNemar test trên correctness của từng utterance.

Bootstrap resample theo speaker và giữ toàn bộ utterance của speaker cùng nhau,
giảm giả định sai rằng các utterance cùng người nói là độc lập.

### 13.1 Province accuracy

| Seed | Accuracy tăng | Speaker-bootstrap CI 95% | P(candidate tốt hơn) | McNemar exact p |
|---:|---:|---:|---:|---:|
| 42 | +0,0543 | [0,0305; 0,0783] | 1,0000 | 6,52 × 10⁻⁷ |
| 43 | +0,0375 | [0,0149; 0,0599] | 0,9996 | 3,39 × 10⁻⁴ |
| 44 | +0,0528 | [0,0293; 0,0758] | 1,0000 | 8,41 × 10⁻⁷ |

CI 95% không chứa 0 ở cả ba seed. McNemar p-value đều nhỏ hơn 0,001. Kết quả
vẫn đạt ngưỡng nếu áp dụng Bonferroni cho ba seed (`alpha = 0,05/3 ≈ 0,0167`).

### 13.2 Province macro-F1

| Seed | Macro-F1 tăng | Speaker-bootstrap CI 95% |
|---:|---:|---:|
| 42 | +0,0505 | [0,0276; 0,0727] |
| 43 | +0,0395 | [0,0173; 0,0613] |
| 44 | +0,0478 | [0,0250; 0,0693] |

CI 95% của macro-F1 cũng không chứa 0 ở cả ba seed.

### 13.3 Region accuracy

| Seed | Accuracy tăng | Speaker-bootstrap CI 95% | P(candidate tốt hơn) | McNemar exact p |
|---:|---:|---:|---:|---:|
| 42 | +0,0074 | [-0,0053; 0,0201] | 0,8665 | 0,2246 |
| 43 | +0,0044 | [-0,0085; 0,0176] | 0,7370 | 0,5066 |
| 44 | +0,0074 | [-0,0035; 0,0182] | 0,8986 | 0,2066 |

Ở mức vùng:

- Cả ba CI đều chứa 0.
- Cả ba McNemar p-value đều lớn hơn 0,05.
- Chưa có bằng chứng cho cải thiện region accuracy có ý nghĩa thống kê.

Điều này hợp lý vì acoustic-only đã đạt khoảng 89-90% ở bài toán ba vùng, nên
prosody tạo mức tăng nhỏ.

### 13.4 Kết luận thống kê cho H1

Kết quả hỗ trợ kết luận:

> Prosody tạo cải thiện có ý nghĩa thống kê cho nhận diện phương ngữ cấp tỉnh,
> nhưng chưa chứng minh được cải thiện có ý nghĩa ở bài toán ba vùng.

Bằng chứng cấp tỉnh nhất quán trên:

- Ba initialization seed.
- Accuracy.
- Balanced accuracy/macro-F1 aggregate.
- Speaker-level bootstrap CI.
- Exact paired McNemar test.

Đây là đóng góp thực nghiệm mạnh nhất của hệ thống hiện tại. Khi viết bài, cần
nêu rõ đơn vị bootstrap là speaker và test set có 2.026 utterance/1.344 speaker.

## 14. Thí nghiệm H3 - Nguồn đặc trưng đầu vào cho router

H3 đánh giá liệu router có lợi hơn khi sử dụng thông tin prosody thay vì chỉ dùng
đặc trưng acoustic. Ba biến thể được giữ giống nhau về backbone, biểu diễn dùng cho
classification, hierarchical MoE, 4 expert, `top_k = 1`, hệ số load balancing và
quy trình huấn luyện; chỉ đầu vào của router thay đổi:

- `acoustic`: router chỉ nhận acoustic representation.
- `prosody`: router chỉ nhận prosody representation.
- `acoustic_prosody`: router nhận kết hợp acoustic và prosody.

Mỗi biến thể được chạy với seed 42, 43 và 44. Kết quả được chọn theo checkpoint
`best_province_accuracy` và đánh giá trên cùng 2.026 utterance thuộc 1.344 speaker.

### 14.1 Mean ± standard deviation trên ba seed

| Router input | Region accuracy | Region macro-F1 | Province accuracy | Province balanced accuracy | Province macro-F1 |
|---|---:|---:|---:|---:|---:|
| Acoustic | 0,9003 ± 0,0053 | 0,8968 ± 0,0058 | 0,4432 ± 0,0099 | 0,4460 ± 0,0099 | 0,4420 ± 0,0131 |
| Prosody | **0,9064 ± 0,0043** | **0,9032 ± 0,0047** | **0,4495 ± 0,0153** | **0,4513 ± 0,0156** | **0,4456 ± 0,0167** |
| Acoustic + prosody | 0,9056 ± 0,0033 | 0,9025 ± 0,0029 | 0,4467 ± 0,0044 | 0,4494 ± 0,0044 | 0,4439 ± 0,0043 |

So với acoustic routing, prosody routing tăng trung bình:

- Region accuracy: `+0,0061`, tương đương khoảng `+0,61` điểm phần trăm.
- Province accuracy: `+0,0063`, tương đương khoảng `+0,63` điểm phần trăm.
- Province macro-F1: `+0,0036`, tương đương khoảng `+0,36` điểm phần trăm.

Router kết hợp acoustic + prosody tăng trung bình:

- Region accuracy: `+0,0053`, tương đương khoảng `+0,53` điểm phần trăm.
- Province accuracy: `+0,0035`, tương đương khoảng `+0,35` điểm phần trăm.
- Province macro-F1: `+0,0019`, tương đương khoảng `+0,19` điểm phần trăm.

Mặc dù prosody routing có mean cao nhất, độ lệch chuẩn của các metric cấp tỉnh
cũng cao nhất. Router kết hợp acoustic + prosody ổn định nhất trên ba seed nhưng
mức tăng trung bình nhỏ.

### 14.2 Prosody routing so với acoustic routing theo từng seed

#### Region accuracy

| Seed | Chênh lệch | Speaker-bootstrap CI 95% | P(prosody tốt hơn) | McNemar exact p |
|---:|---:|---:|---:|---:|
| 42 | +0,0143 | [0,0044; 0,0246] | 0,9969 | 0,0051 |
| 43 | +0,0030 | [-0,0070; 0,0129] | 0,7005 | 0,6137 |
| 44 | +0,0010 | [-0,0098; 0,0118] | 0,5540 | 0,9273 |

Prosody routing cải thiện region accuracy có ý nghĩa ở seed 42, nhưng không lặp
lại ở seed 43 và 44.

#### Province accuracy

| Seed | Chênh lệch | Speaker-bootstrap CI 95% | P(prosody tốt hơn) | McNemar exact p |
|---:|---:|---:|---:|---:|
| 42 | -0,0099 | [-0,0307; 0,0115] | 0,1696 | 0,3433 |
| 43 | -0,0064 | [-0,0267; 0,0138] | 0,2631 | 0,5311 |
| 44 | +0,0350 | [0,0154; 0,0549] | 0,9998 | 0,00019 |

Ở cấp tỉnh, prosody routing kém hơn acoustic routing tại seed 42 và 43 nhưng tốt
hơn rõ rệt tại seed 44. Chỉ seed 44 có CI 95% không chứa 0 và McNemar p-value nhỏ
hơn 0,05. Hiệu ứng đổi dấu giữa các seed, do đó mean cao hơn không đủ để kết luận
prosody routing tạo cải thiện ổn định.

### 14.3 Router acoustic + prosody so với acoustic routing

#### Region accuracy

| Seed | Chênh lệch | Speaker-bootstrap CI 95% | P(kết hợp tốt hơn) | McNemar exact p |
|---:|---:|---:|---:|---:|
| 42 | +0,0133 | [0,0030; 0,0238] | 0,9929 | 0,0150 |
| 43 | +0,0039 | [-0,0070; 0,0148] | 0,7420 | 0,5159 |
| 44 | -0,0015 | [-0,0131; 0,0098] | 0,3715 | 0,8570 |

Chỉ seed 42 cho cải thiện region accuracy có ý nghĩa; seed 43 và 44 không xác
nhận hiệu ứng này.

#### Province accuracy

| Seed | Chênh lệch | Speaker-bootstrap CI 95% | P(kết hợp tốt hơn) | McNemar exact p |
|---:|---:|---:|---:|---:|
| 42 | +0,0030 | [-0,0172; 0,0235] | 0,5986 | 0,8007 |
| 43 | -0,0074 | [-0,0284; 0,0135] | 0,2357 | 0,4756 |
| 44 | +0,0148 | [-0,0059; 0,0352] | 0,9179 | 0,1419 |

Cả ba CI 95% đều chứa 0 và cả ba McNemar p-value đều lớn hơn 0,05. Vì vậy chưa
có bằng chứng thống kê rằng router kết hợp cải thiện province accuracy so với
acoustic routing.

### 14.4 Phân tích routing

Entropy trung bình của cả ba biến thể đều xấp xỉ:

```text
ln(4) = 1,386294
```

Giá trị quan sát:

| Router input | Mean entropy |
|---|---:|
| Acoustic | 1,386281 |
| Prosody | 1,386281 |
| Acoustic + prosody | 1,386288 |

Mean expert probability của tất cả cấu hình đều gần:

```text
[0,25; 0,25; 0,25; 0,25]
```

Load-balancing đã ngăn expert collapse, nhưng router hiện gần phân phối đều tối
đa. Điều này chưa chứng minh router học được chuyên môn hóa theo vùng, tỉnh hoặc
đặc trưng prosody. Sự khác biệt accuracy giữa các cấu hình có thể đến từ biến
thiên tối ưu hóa hoặc projection của router, thay vì một chính sách expert
selection có ý nghĩa.

### 14.5 Kết luận về H3

Kết quả hiện tại **không hỗ trợ H3 một cách ổn định**:

> Thay đầu vào router từ acoustic sang prosody hoặc acoustic + prosody chưa tạo
> cải thiện cấp tỉnh nhất quán qua ba seed.

Prosody routing đạt mean cao nhất, nhưng:

- Province accuracy giảm ở seed 42 và 43, chỉ tăng có ý nghĩa ở seed 44.
- Router kết hợp không đạt ý nghĩa thống kê ở province accuracy trong bất kỳ seed
  nào.
- Các router đều gần uniform và chưa thể hiện chuyên môn hóa expert.

Do đó không nên tuyên bố prosody-aware routing là đóng góp đã được xác nhận. Kết
quả mạnh hơn vẫn là H1: prosody có ích khi được thêm vào biểu diễn phục vụ
classification. Hướng tiếp theo nên ưu tiên đo chuyên môn hóa expert theo
region/province, điều chỉnh load-balancing để tránh uniform routing, và chọn
hyperparameter trên validation set trước một lần đánh giá test cuối.

### 14.6 Sự cố vận hành khi chạy H3

Chín thí nghiệm H3 đã hoàn thành đầy đủ. Trong lần chạy song song ban đầu, ba job
bị CUDA out-of-memory vì GPU được chọn bị tiến trình khác chiếm gần hết VRAM:

- `h3_router_prosody_seed43`
- `h3_router_prosody_seed44`
- `h3_router_acoustic_prosody_seed44`

Ba job được chạy lại thành công trên các GPU vật lý 7, 2 và 5, mỗi GPU một job,
với `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Kết quả cuối có đủ 9 file
`metrics_test_best_province_accuracy.json`; các lần lỗi cũ chỉ được giữ trong log
để truy vết và không được dùng trong bảng tổng hợp.

## 15. Thiết kế H4 - Sweep load-balancing trên validation

H4 được thiết kế sau khi H3 cho thấy soft routing gần uniform tối đa. Mục tiêu là
tìm hệ số load-balancing cân bằng giữa hai cực:

```text
expert collapse <- chuyên môn hóa hữu ích -> uniform routing
```

Sáu giá trị được lên kế hoạch: `0`, `0,0001`, `0,001`, `0,005`, `0,01` và `0,02`.
Giai đoạn sweep chỉ dùng seed 42 và validation set. Cấu hình, checkpoint và
hyperparameter không được chọn bằng test set.

Ngoài accuracy, balanced accuracy và macro-F1, pipeline đánh giá được bổ sung:

- Entropy mềm chuẩn hóa theo `ln(num_experts)`.
- Số expert hiệu dụng `exp(entropy)`.
- Entropy của top-1 expert assignment.
- Số expert thực sự hoạt động và tỉ lệ expert lớn nhất.
- NMI giữa expert assignment với region.
- NMI giữa expert assignment với province.

Quy tắc chọn là province macro-F1 validation cao nhất trong các run không
collapse; province balanced accuracy và province accuracy được dùng để phá hòa.
Sau sweep, cấu hình được chọn phải được lặp lại trên seed 42/43/44 ở validation
trước khi khóa và đánh giá test cuối.

### 15.1 Kết quả sweep seed 42

| Load balance | Region acc. | Province acc. | Province balanced acc. | Province macro-F1 | Soft entropy chuẩn hóa | Top-1 entropy chuẩn hóa | Expert lớn nhất | Region NMI | Province NMI |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0,8921 | 0,4895 | 0,4822 | 0,4749 | 0,9587 | 0,7424 | 0,4647 | **0,3759** | **0,2141** |
| 0,0001 | 0,8937 | 0,4884 | 0,4830 | 0,4752 | 1,0000 | 0,7912 | 0,4937 | 0,2482 | 0,1609 |
| **0,001** | 0,8963 | 0,4979 | 0,4928 | **0,4867** | 1,0000 | 0,7124 | 0,6432 | 0,2285 | 0,1566 |
| 0,005 | 0,8937 | **0,4989** | 0,4932 | 0,4857 | 1,0000 | 0,8363 | 0,5368 | 0,2372 | 0,1702 |
| 0,01 | **0,9021** | 0,4984 | **0,4937** | 0,4835 | 1,0000 | 0,5205 | 0,7468 | 0,2771 | 0,1426 |
| 0,02 | 0,8884 | 0,4879 | 0,4821 | 0,4668 | 1,0000 | **0,9976** | **0,2732** | 0,2021 | 0,1619 |

Không run nào vượt ngưỡng collapse 90%. Cấu hình `0,001` được chọn vì đạt
province macro-F1 validation cao nhất (`0,4867`) theo quy tắc đã định trước.
`0,005` có province accuracy cao hơn khoảng 0,11 điểm phần trăm nhưng macro-F1
thấp hơn khoảng 0,10 điểm phần trăm; vì vậy không thay đổi tiêu chí sau khi xem
kết quả.

### 15.2 Diễn giải routing

Khi bỏ load-balancing (`weight = 0`), soft entropy giảm xuống 0,9587 và NMI tăng
cao nhất, cho thấy router có tín hiệu phân hóa rõ hơn. Tuy nhiên province
macro-F1 chỉ đạt 0,4749, thấp hơn cấu hình được chọn khoảng 1,18 điểm phần trăm.

Với mọi hệ số khác 0, soft probability vẫn gần uniform tuyệt đối và số expert
hiệu dụng gần 4. Dù vậy, top-1 assignment không uniform vì các chênh lệch xác suất
rất nhỏ vẫn thay đổi expert đứng đầu. Vì thế cần phân biệt:

- Soft probabilities gần uniform không chứng minh router tự tin.
- Top-1 assignment mất cân bằng không đồng nghĩa với soft router đã chuyên môn
  hóa mạnh.
- NMI cho thấy association có tồn tại, nhưng chưa chứng minh quan hệ nhân quả hay
  lợi ích lặp lại qua seed.

### 15.3 Cấu hình được chọn cho xác nhận đa seed

Hệ số `load_balance_weight = 0,001` được khóa cho giai đoạn xác nhận validation
trên seed 43 và 44. Giai đoạn này vẫn không sử dụng test set. Sau khi có ba seed,
cần kiểm tra đồng thời province macro-F1, variance, collapse, entropy và NMI trước
khi quyết định có đưa cấu hình vào đánh giá test cuối hay không.

### 15.4 Kết quả xác nhận đa seed trên validation

| Seed | Region accuracy | Region macro-F1 | Province accuracy | Province balanced accuracy | Province macro-F1 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0,8963 | 0,8922 | 0,4979 | 0,4928 | 0,4867 |
| 43 | 0,8995 | 0,8955 | 0,4963 | 0,4919 | 0,4861 |
| 44 | 0,8905 | 0,8853 | 0,4947 | 0,4916 | 0,4824 |
| **Mean ± SD** | **0,8954 ± 0,0045** | **0,8910 ± 0,0052** | **0,4963 ± 0,0016** | **0,4921 ± 0,0006** | **0,4851 ± 0,0023** |

Hiệu năng cấp tỉnh lặp lại tốt:

- Province accuracy chỉ dao động từ 0,4947 đến 0,4979.
- Province balanced accuracy có SD khoảng 0,0006.
- Province macro-F1 có SD khoảng 0,0023.

Độ ổn định này mạnh hơn kết quả MoE-2 trước đó, nhưng mới chỉ chứng minh cấu hình
H4 tự lặp lại; chưa chứng minh nó tốt hơn baseline không MoE.

### 15.5 Routing qua ba seed

| Seed | Soft entropy chuẩn hóa | Top-1 entropy chuẩn hóa | Expert lớn nhất | Region NMI | Province NMI | Collapse |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0,999994 | 0,7124 | 0,6432 | 0,2285 | 0,1566 | Không |
| 43 | 0,999982 | 0,7127 | 0,6421 | 0,1234 | 0,1185 | Không |
| 44 | 0,999979 | 0,6813 | 0,6711 | 0,2440 | 0,1549 | Không |

Cả bốn expert đều hoạt động ở mọi seed và không có expert vượt ngưỡng collapse
90%. Tỉ lệ expert lớn nhất khá ổn định, khoảng 64-67%. Tuy nhiên:

- Soft entropy vẫn gần 1, tức xác suất router gần uniform.
- Expert chiếm ưu thế không có cùng chỉ số qua các seed.
- Region/province NMI giảm đáng kể ở seed 43.

Do expert label có tính hoán vị giữa các initialization, việc expert số mấy chiếm
ưu thế không cần giống nhau. Dẫu vậy, soft routing gần uniform cho thấy bằng chứng
về specialization vẫn yếu. Kết quả hiện tại chỉ hỗ trợ nhận định thận trọng rằng
top-1 assignment có cấu trúc và không collapse.

### 15.6 Cổng quyết định trước test cuối

Chưa chuyển trực tiếp sang test. Cần đánh giá ba checkpoint baseline
`acoustic + prosody, no MoE` trên validation bằng cùng pipeline, sau đó thực hiện
paired bootstrap và McNemar trên validation predictions.

Chỉ đưa H4 sang test cuối nếu:

- Mean province macro-F1/accuracy validation không kém baseline đáng kể.
- Hiệu ứng không phụ thuộc duy nhất một seed.
- Paired results không cho thấy H4 suy giảm rõ ràng.

Quy tắc này tránh dùng test set để quyết định có giữ MoE hay không.

### 15.7 So sánh H4 với baseline không MoE trên validation

Baseline là acoustic + prosody, không MoE, sử dụng cùng checkpoint selection và
cùng validation set.

| Cấu hình | Region accuracy | Province accuracy | Province balanced accuracy | Province macro-F1 |
|---|---:|---:|---:|---:|
| Baseline không MoE | 0,8921 ± 0,0009 | 0,4940 ± 0,0095 | 0,4900 ± 0,0074 | 0,4825 ± 0,0085 |
| H4 MoE-4, load balance 0,001 | 0,8954 ± 0,0045 | 0,4963 ± 0,0016 | 0,4921 ± 0,0006 | 0,4851 ± 0,0023 |
| Chênh lệch mean H4 - baseline | +0,0033 | +0,0023 | +0,0020 | +0,0025 |

H4 có mean cao hơn rất nhẹ, khoảng 0,23 điểm phần trăm province accuracy và 0,25
điểm phần trăm province macro-F1. H4 cũng có variance cấp tỉnh thấp hơn trong ba
seed này. Tuy nhiên hiệu ứng theo seed không nhất quán.

#### Province accuracy paired

| Seed | H4 - baseline | Speaker-bootstrap CI 95% | P(H4 tốt hơn) | McNemar exact p |
|---:|---:|---:|---:|---:|
| 42 | +0,0032 | [-0,0209; 0,0277] | 0,6042 | 0,8129 |
| 43 | +0,0121 | [-0,0123; 0,0375] | 0,8300 | 0,3002 |
| 44 | -0,0084 | [-0,0324; 0,0154] | 0,2309 | 0,4776 |

Cả ba CI 95% chứa 0 và cả ba McNemar p-value lớn hơn 0,05. H4 tăng ở seed 42,
43 nhưng giảm ở seed 44.

#### Province macro-F1 paired

| Seed | H4 - baseline | Speaker-bootstrap CI 95% |
|---:|---:|---:|
| 42 | +0,0068 | [-0,0168; 0,0315] |
| 43 | +0,0105 | [-0,0128; 0,0356] |
| 44 | -0,0096 | [-0,0330; 0,0146] |

Không CI nào loại trừ 0. Kết quả không cung cấp bằng chứng thống kê rằng H4 cải
thiện province macro-F1.

#### Region accuracy paired

| Seed | H4 - baseline | Speaker-bootstrap CI 95% | McNemar exact p |
|---:|---:|---:|---:|
| 42 | +0,0053 | [-0,0073; 0,0175] | 0,4300 |
| 43 | +0,0068 | [-0,0056; 0,0194] | 0,2869 |
| 44 | -0,0021 | [-0,0146; 0,0104] | 0,7807 |

Kết quả region cũng đổi dấu và không có ý nghĩa thống kê.

### 15.8 Quyết định chính thức cho H4

Theo cổng quyết định được đặt trước khi xem baseline validation:

> Không có đủ bằng chứng để đưa H4 MoE vào một vòng test cuối.

Lý do:

- Mức tăng mean rất nhỏ.
- Hiệu ứng đổi dấu ở seed 44.
- Tất cả paired bootstrap CI 95% đều chứa 0.
- Tất cả McNemar p-value đều lớn hơn 0,05.
- Soft routing vẫn gần uniform, nên bằng chứng chuyên môn hóa expert yếu.
- H3 trên test trước đó cũng không cho thấy lợi ích MoE lặp lại ổn định.

Nhánh tuning MoE/load-balancing được dừng tại đây để tránh tiếp tục tối ưu gián
tiếp theo test set và tiêu tốn compute. Kết quả H4 được báo cáo như một negative
result có kiểm soát. Baseline acoustic + prosody không MoE tiếp tục là hệ thống
tham chiếu chính.

Hướng thực nghiệm tiếp theo chuyển sang đặc trưng spectral/FFT, được đánh giá
validation-only trước khi khóa bất kỳ cấu hình mới nào.

## 16. Thiết kế H5 - Đặc trưng spectral/FFT

Trong quá trình chuẩn bị H5, phát hiện vector `prosody` legacy 12 chiều đã chứa
spectral centroid, bandwidth và roll-off. Vì vậy kết luận H1 cần được diễn đạt
chính xác là nhóm đặc trưng prosody kết hợp một số thống kê phổ cơ bản có ích cho
classification; H1 chưa tách riêng hoàn toàn prosody khỏi spectral.

Để tạo ablation sạch, H5 bổ sung:

- Bộ `pitch_energy` 9 chiều không chứa centroid, bandwidth hay roll-off.
- Bộ spectral 24 chiều trích từ FFT/STFT.
- Chế độ fusion acoustic, pitch/energy và spectral độc lập.
- Baseline handcrafted MLP không sử dụng pretrained backbone.

Bốn cấu hình seed 42 được lên kế hoạch:

1. Acoustic + pitch/energy.
2. Acoustic + spectral.
3. Acoustic + pitch/energy + spectral.
4. Handcrafted pitch/energy + spectral, không backbone.

Mọi cấu hình H5 đều không sử dụng MoE và chỉ được chọn trên validation. Tiêu chí
chính là province macro-F1; province balanced accuracy và province accuracy là
tiêu chí phụ. Paired speaker bootstrap/McNemar được thực hiện trên validation
predictions trước khi quyết định cấu hình nào được lặp lại ở seed 43 và 44.

### 16.1 Sửa lỗi smoke test H5

Lần smoke test đầu tiên train với `--max-samples 32` nhưng evaluate không nhận
tham số này. Vocabulary train cũng được dựng sau khi cắt dữ liệu nên checkpoint
chỉ có head 1 vùng/1 tỉnh, trong khi evaluate toàn bộ dataset dựng head 3 vùng/63
tỉnh và gây lỗi size mismatch.

Đã sửa theo hai lớp:

- Dựng region/province vocabulary từ toàn bộ dataset trước khi áp dụng
  `max_samples`.
- Truyền cùng `--max-samples` từ experiment runner sang evaluator.
- Khởi tạo best accuracy bằng âm vô cùng để epoch đầu tiên luôn ghi đè checkpoint
  cũ, kể cả khi smoke accuracy bằng 0.

Checkpoint smoke lỗi phải được ghi đè bằng một lần chạy lại sau khi pull bản sửa.
Lỗi chỉ thuộc pipeline smoke test, chưa ảnh hưởng bất kỳ full experiment H5 nào.

### 16.2 Kết quả H5 seed 42 trên validation

| Cấu hình | Region accuracy | Province accuracy | Province balanced accuracy | Province macro-F1 |
|---|---:|---:|---:|---:|
| Acoustic + pitch/energy | 0,8937 | 0,4811 | 0,4772 | 0,4696 |
| Acoustic + spectral | **0,8984** | 0,4389 | 0,4333 | 0,4293 |
| Acoustic + pitch/energy + spectral | 0,8974 | **0,5021** | **0,4986** | **0,4899** |
| Handcrafted pitch/energy + spectral | 0,5958 | 0,2000 | 0,1974 | 0,1710 |

Spectral một mình tăng nhẹ metric vùng nhưng làm giảm rõ rệt metric cấp tỉnh.
So với acoustic + pitch/energy:

- Province accuracy giảm 0,0421.
- Province balanced accuracy giảm 0,0440.
- Province macro-F1 giảm 0,0403.
- Bootstrap CI 95% của cả ba metric cấp tỉnh đều hoàn toàn dưới 0.
- McNemar exact p-value của province accuracy bằng 0,00045.

Do đó spectral không thay thế được pitch/energy cho phân loại 63 tỉnh.

### 16.3 Spectral bổ sung cho pitch/energy

Fusion acoustic + pitch/energy + spectral so với acoustic + pitch/energy:

| Metric | Chênh lệch | Speaker-bootstrap CI 95% | P(fusion tốt hơn) |
|---|---:|---:|---:|
| Province accuracy | +0,0211 | [-0,0026; 0,0448] | 0,9576 |
| Province balanced accuracy | +0,0214 | [-0,0015; 0,0442] | 0,9658 |
| Province macro-F1 | +0,0203 | [-0,0035; 0,0439] | 0,9535 |

McNemar exact p-value của province accuracy là 0,0605. Tín hiệu tăng khoảng 2,0-
2,1 điểm phần trăm khá lớn về thực tiễn nhưng chưa đạt ngưỡng 0,05 ở seed 42 và
CI còn hơi chứa 0. Vì vậy kết quả đủ điều kiện xác nhận đa seed nhưng chưa đủ để
tuyên bố H5 được hỗ trợ.

### 16.4 Baseline handcrafted

Không có pretrained backbone, handcrafted MLP đạt:

- Region accuracy 0,5958.
- Province accuracy 0,2000.
- Province macro-F1 0,1710.

Mặc dù thấp hơn Wav2Vec2 rõ rệt, kết quả cao hơn nhiều so với random chance
(`1/3` cho vùng và `1/63` cho tỉnh), xác nhận pitch/energy và spectral chứa tín
hiệu phương ngữ độc lập. Tuy nhiên handcrafted không đủ để thay thế learned
acoustic representation.

### 16.5 Quyết định đa seed

Không chạy lại spectral-only hoặc handcrafted ở seed 43/44 vì hai cấu hình này
không cạnh tranh với baseline theo tiêu chí cấp tỉnh. Chỉ hai cấu hình sau được
lặp lại:

1. Acoustic + pitch/energy làm baseline sạch.
2. Acoustic + pitch/energy + spectral làm candidate.

Cả hai phải chạy seed 43 và 44 trên validation để giữ paired comparison công
bằng. Chưa sử dụng test set.

### 16.6 Kết quả đa seed H5 trên validation

| Cấu hình | Region accuracy | Province accuracy | Province balanced accuracy | Province macro-F1 |
|---|---:|---:|---:|---:|
| Acoustic + pitch/energy | 0,8954 ± 0,0022 | 0,4921 ± 0,0148 | 0,4893 ± 0,0159 | 0,4830 ± 0,0145 |
| Acoustic + pitch/energy + spectral | 0,8968 ± 0,0014 | 0,4963 ± 0,0079 | 0,4930 ± 0,0082 | 0,4839 ± 0,0111 |
| Chênh lệch mean fusion - baseline | +0,0014 | +0,0042 | +0,0037 | +0,0010 |

Fusion có mean province accuracy cao hơn khoảng 0,42 điểm phần trăm, nhưng mean
province macro-F1 chỉ cao hơn khoảng 0,10 điểm phần trăm. Mức tăng nhỏ hơn nhiều
so với tín hiệu single-seed.

### 16.7 Paired comparison qua ba seed

#### Province accuracy

| Seed | Fusion - baseline | Speaker-bootstrap CI 95% | P(fusion tốt hơn) | McNemar exact p |
|---:|---:|---:|---:|---:|
| 42 | +0,0211 | [-0,0026; 0,0448] | 0,9576 | 0,0605 |
| 43 | +0,0132 | [-0,0096; 0,0366] | 0,8615 | 0,2443 |
| 44 | -0,0216 | [-0,0438; 0,0011] | 0,0294 | 0,0473 |

Hiệu ứng đổi dấu ở seed 44. Seed 42 và 43 không đạt ý nghĩa thống kê; seed 44
cho bằng chứng McNemar rằng fusion làm giảm accuracy.

#### Province macro-F1

| Seed | Fusion - baseline | Speaker-bootstrap CI 95% |
|---:|---:|---:|
| 42 | +0,0203 | [-0,0035; 0,0439] |
| 43 | +0,0098 | [-0,0130; 0,0339] |
| 44 | -0,0272 | [-0,0487; -0,0043] |

Tại seed 44, CI 95% hoàn toàn dưới 0. Đây là bằng chứng fusion spectral có thể
làm giảm macro-F1 đáng kể tùy initialization.

#### Region accuracy

Fusion thay đổi region accuracy lần lượt `+0,0037`, `-0,0026` và `+0,0032`.
Tất cả CI 95% đều chứa 0 và McNemar p-value đều lớn hơn 0,05. Không có bằng
chứng cải thiện ổn định ở cấp vùng.

### 16.8 Quyết định chính thức cho H5

Theo cổng validation đã định trước:

> Không khóa fusion spectral để chạy test.

Lý do:

- Hiệu ứng cấp tỉnh đổi dấu giữa các seed.
- Mean macro-F1 chỉ tăng khoảng 0,10 điểm phần trăm.
- Seed 44 giảm province accuracy khoảng 2,16 điểm phần trăm.
- Seed 44 giảm balanced accuracy và macro-F1 với bootstrap CI hoàn toàn dưới 0.
- Spectral-only đã giảm mạnh hiệu năng cấp tỉnh ở seed 42.

Kết luận phù hợp là:

> Các đặc trưng spectral/FFT thủ công chứa tín hiệu phương ngữ và có thể bổ sung
> pitch/energy ở một số initialization, nhưng cách fusion hiện tại không tạo cải
> thiện ổn định cho nhận diện 63 tỉnh.

H5 được dừng mà không đánh giá test, tránh chọn cấu hình dựa gián tiếp trên test
set. Baseline acoustic + prosody legacy không MoE vẫn là hệ thống tham chiếu
chính đã có bằng chứng mạnh nhất.

## 17. Thiết kế H6 - Audit leakage và speaker-disjoint split

H6 ưu tiên kiểm tra tính hợp lệ của protocol đánh giá trước khi thử thêm kiến
trúc. Các nguy cơ được audit:

- Speaker xuất hiện ở nhiều split.
- Filename xuất hiện ở nhiều split.
- Audio SHA-256 trùng giữa các split.
- Một speaker mang nhiều nhãn region hoặc province.

Pipeline audit chỉ đọc metadata ở bước đầu. Duration header và SHA-256 là hai chế
độ tùy chọn vì có chi phí I/O cao hơn.

Nếu speaker overlap tồn tại, pipeline tạo manifest speaker-disjoint thay vì sao
chép audio. Manifest ánh xạ từng `(original_split, row_index)` sang split mới và
được kiểm tra bao phủ toàn bộ dataset. Data loader từ chối:

- Manifest thiếu row hoặc stale.
- Row bị lặp.
- Index vượt phạm vi.
- Speaker xuất hiện trong nhiều split mới.
- Manifest không tạo đủ train/valid/test.

Speaker được gán một lần và stratify xấp xỉ theo majority province, với seed 42
và tỷ lệ utterance mục tiêu gần split gốc: train 0,793, valid 0,100, test 0,107.
Nếu speaker có xung đột nhãn, builder mặc định dừng để yêu cầu điều tra.

Sau khi audit được duyệt, H6 sẽ chạy acoustic-only và acoustic + prosody legacy
không MoE trên speaker-disjoint split. Mục tiêu là kiểm tra H1 còn lặp lại khi
người nói hoàn toàn không trùng giữa train, validation và test hay không.

### 17.1 Kết quả metadata audit

| Split | Utterance | Speaker |
|---|---:|---:|
| Train | 15.023 | 10.291 |
| Validation | 1.900 | 1.320 |
| Test | 2.026 | 1.344 |

Audit phát hiện 2 speaker xuất hiện ở cả validation và test, ảnh hưởng tổng cộng
5 utterance. Không có speaker overlap với train, không filename trùng giữa các
split và không speaker mang nhiều nhãn region/province.

| Speaker | Validation | Test |
|---|---:|---:|
| `spk_73_0186` | 1 | 2 |
| `spk_76_0219` | 1 | 1 |

Mức leakage rất nhỏ nhưng vẫn vi phạm nguyên tắc test speaker chưa từng thấy khi
validation được dùng để chọn mô hình.

### 17.2 Quyết định sửa split

Không rebuild toàn bộ dataset vì điều đó thay đổi gần 19 nghìn assignment và tạo
thêm variance không cần thiết. H6 dùng minimal repair với priority:

```text
train > valid > test
```

Hai speaker overlap được giữ trong validation; ba utterance tương ứng được chuyển
từ test sang validation. Dự kiến train giữ nguyên 15.023 utterance, validation
tăng lên 1.903 và test giảm xuống 2.023. Quyết định chỉ dựa trên metadata, không
sử dụng prediction hay metric mô hình.

### 17.3 Xác nhận manifest

Manifest tạo thành công với:

- `strategy = preserve`.
- `moved_speakers = 2`.
- `moved_utterances = 3`.
- Train: 15.023 utterance, 10.291 speaker.
- Validation: 1.903 utterance, 1.320 speaker.
- Test: 2.023 utterance, 1.342 speaker.
- Không speaker overlap sau repair.
- Cả 63 province vẫn có mặt trong từng split.

Vì train split không thay đổi, không huấn luyện lại mô hình. H6 tái sử dụng sáu
checkpoint acoustic-only/acoustic+prosody đã khóa và chỉ đánh giá chúng trên
repaired test. Cách làm này cô lập tác động của ba utterance leakage, tránh thêm
variance do retraining và tiết kiệm compute.

### 17.4 Lỗi PyArrow khi đánh giá manifest và cách sửa

Lần đánh giá đầu tiên dừng với:

```text
pyarrow.lib.ArrowInvalid: offset overflow while concatenating arrays
```

Nguyên nhân không nằm ở manifest. Sau `select`/`concatenate_datasets`, dataset có
indirection indices. Lệnh `dataset.unique()` để dựng label vocabulary kích hoạt
`flatten_indices()`, khiến PyArrow cố nối cột audio nhúng lớn hơn giới hạn offset
32-bit.

Đã sửa bằng cách dựng region/province vocabulary trên Arrow table gốc trước khi
áp manifest. Sau đó mới select/concatenate và cast audio. Cách này không materialize
lại cột audio, giữ nguyên label mapping của checkpoint cũ và tránh overflow.

### 17.5 Tương thích fusion của checkpoint acoustic-only cũ

Sau khi sửa Arrow, checkpoint acoustic-only cũ báo fusion weight có input 384
trong khi model mới tạo input 256. Nguyên nhân là model trước H5 luôn nối acoustic
256 chiều với prosody zero 128 chiều, kể cả khi `use_prosody = false`. H5 đã tối
ưu bằng cách loại nhánh tắt khỏi fusion, vô tình thay đổi architecture của config
legacy.

Đã thêm hai chế độ:

- Config pre-H5 không khai báo `use_spectral`/`prosody_feature_set`: giữ fusion
  legacy `acoustic_dim + prosody_dim`, tương thích checkpoint cũ.
- Config H5 khai báo rõ feature keys: dùng dynamic fusion theo các nhánh bật.

Không chuyển đổi hoặc sửa checkpoint; architecture tương thích được phục hồi từ
config.

### 17.6 Kết quả H6 trên repaired test

Tập test sau minimal repair có 2.023 utterance và 1.342 speaker. Sáu checkpoint đã
khóa của H1 được đánh giá lại, không huấn luyện lại và không chọn lại
hyperparameter.

| Mô hình | Region accuracy | Province accuracy | Province balanced accuracy | Province macro-F1 |
|---|---:|---:|---:|---:|
| Acoustic-only | 0,8950 ± 0,0020 | 0,3946 ± 0,0148 | 0,3975 ± 0,0163 | 0,3911 ± 0,0131 |
| Acoustic + prosody | **0,9015 ± 0,0030** | **0,4427 ± 0,0087** | **0,4469 ± 0,0090** | **0,4368 ± 0,0082** |
| Chênh lệch | +0,0064 | **+0,0481** | **+0,0494** | **+0,0457** |

Prosody cải thiện province accuracy trung bình 4,81 điểm phần trăm, province
balanced accuracy 4,94 điểm phần trăm và province macro-F1 4,57 điểm phần trăm.

### 17.7 Kiểm định paired theo seed

| Seed | Province accuracy acoustic | Province accuracy + prosody | Chênh lệch | Bootstrap 95% CI | McNemar p |
|---:|---:|---:|---:|---:|---:|
| 42 | 0,3786 | 0,4330 | +0,0544 | [0,0304; 0,0783] | 6,52e-7 |
| 43 | 0,4078 | 0,4454 | +0,0376 | [0,0152; 0,0600] | 3,18e-4 |
| 44 | 0,3974 | 0,4498 | +0,0524 | [0,0290; 0,0762] | 1,16e-6 |

Cả ba seed đều có confidence interval không chứa 0 và McNemar p nhỏ hơn ngưỡng
Bonferroni 0,05/3 = 0,0167. Vì vậy cải thiện nhận dạng cấp tỉnh có ý nghĩa thống kê
và lặp lại ổn định giữa các seed. Trái lại, confidence interval của region accuracy
đều chứa 0; chưa có đủ bằng chứng để kết luận prosody cải thiện cấp vùng.

### 17.8 Ảnh hưởng của speaker overlap và kết luận H6

So với test gốc, thay đổi tuyệt đối của province accuracy chỉ là +0,026 điểm phần
trăm cho acoustic-only và +0,016 điểm phần trăm cho acoustic + prosody. Chênh lệch
prosody so với acoustic giảm khoảng 0,009 điểm phần trăm; province macro-F1 giảm
khoảng 0,025 điểm phần trăm. Đây là mức không đáng kể.

**Kết luận H6:** speaker overlap giữa validation và test chỉ gồm hai speaker và ba
test utterance, không làm thay đổi kết luận chính. H1 được xác nhận trên repaired
speaker-disjoint test: prosody cải thiện rõ rệt và có ý nghĩa thống kê cho nhận
dạng cấp tỉnh, nhưng chưa tạo cải thiện có ý nghĩa ở cấp vùng.

## 18. H7: phân tích lỗi và calibration

H7 dùng prediction H6 đã khóa, không huấn luyện lại và không dùng test để chọn
hyperparameter. Phân tích gồm hiệu quả theo từng tỉnh qua ba seed, các cặp tỉnh
nhầm lẫn, chuyển trạng thái sai-thành-đúng/đúng-thành-sai và calibration bằng ECE,
NLL, Brier score.

Script `scripts/analyze_h7.py` sinh bảng từng seed, bảng aggregate theo tỉnh, bảng
confusion pair, calibration bin và `h7_summary.json`. Prediction hiện chưa lưu
thời lượng audio nên phân tích duration được tách khỏi H7 chính; không suy diễn
thời lượng từ filename hay kích thước probability.

### 18.1 Chuyển trạng thái dự đoán

Mỗi seed có 2.023 mẫu repaired test. Khi thay acoustic-only bằng acoustic +
prosody:

| Seed | Cả hai đúng | Prosody sửa đúng | Cả hai sai | Prosody làm sai |
|---:|---:|---:|---:|---:|
| 42 | 579 | 297 | 960 | 187 |
| 43 | 645 | 256 | 942 | 180 |
| 44 | 622 | 288 | 931 | 182 |
| Tổng | 1.846 | **841** | 2.833 | **549** |

Prosody tạo lợi ròng 292 lượt dự đoán đúng qua ba seed. Kết quả này phù hợp với
chênh lệch province accuracy dương đã được kiểm định paired ở H6, nhưng vẫn còn
549 trường hợp acoustic-only đúng bị prosody làm sai.

### 18.2 Hiệu quả theo tỉnh

Trong 63 tỉnh, accuracy trung bình tăng ở 34 tỉnh, giảm ở 23 tỉnh và không đổi ở
6 tỉnh. Có 17 tỉnh tăng ở cả ba seed và 6 tỉnh giảm ở cả ba seed.

| Tỉnh | Acoustic | Acoustic + prosody | Chênh lệch | Seed cải thiện |
|---:|---:|---:|---:|---:|
| 17 | 0,1616 | 0,6667 | **+0,5051** | 3/3 |
| 30 | 0,4598 | 0,8046 | **+0,3448** | 3/3 |
| 22 | 0,1569 | 0,4706 | **+0,3137** | 3/3 |
| 81 | 0,4242 | 0,7172 | **+0,2929** | 3/3 |
| 77 | 0,7024 | 0,9762 | **+0,2738** | 3/3 |

Các suy giảm nhất quán lớn nhất là tỉnh 38 (-0,1818), 70 (-0,1667), 14
(-0,1373) và 11 (-0,1238), đều giảm ở 3/3 seed. Vì mỗi tỉnh chỉ có khoảng
27--40 mẫu trong test, đây là phân tích mô tả; chưa xem từng chênh lệch theo tỉnh
là một kiểm định độc lập có ý nghĩa thống kê.

### 18.3 Các cặp tỉnh nhầm lẫn

Prosody loại bỏ hoặc giảm mạnh một số lỗi: 78→77 giảm 18 lượt, 94→83 giảm 17,
76→86 giảm 15, 17→27 và 47→48 cùng giảm 13. Ngược lại, một số hướng nhầm tăng:
38→73 tăng 18 lượt, 76→77 tăng 16, 69→84 và 43→78 cùng tăng 14.

Hai chiều của cùng một cặp có thể thay đổi trái dấu. Ví dụ 73→38 giảm 10 nhưng
38→73 tăng 18; 47→48 giảm 13 nhưng 48→47 tăng 12. Điều này cho thấy prosody có
thể dịch chuyển ranh giới quyết định về một phía thay vì giải quyết hoàn toàn sự
tương đồng giữa hai tỉnh.

### 18.4 Calibration

| Mô hình | ECE | NLL | Brier | Confidence trung bình |
|---|---:|---:|---:|---:|
| Acoustic-only | 0,2918 | 2,8241 | 0,8569 | 0,6864 |
| Acoustic + prosody | **0,2453** | **2,4748** | **0,7820** | 0,6880 |

Trung bình ba seed, prosody giảm ECE 0,0465, NLL 0,3493 và Brier 0,0749 trong
khi confidence trung bình gần như không đổi. Tuy vậy, cải thiện calibration không
đồng đều: seed 42 và 44 tốt hơn rõ, còn seed 43 có ECE tăng từ 0,2467 lên 0,2849
và NLL tăng nhẹ từ 2,6097 lên 2,6396; riêng Brier seed 43 vẫn giảm.

### 18.5 Kết luận H7

Prosody tạo cải thiện tổng thể thực và có lợi ròng, đồng thời cải thiện calibration
trung bình. Lợi ích tập trung mạnh ở một nhóm tỉnh thay vì phân bố đồng đều. Một
số tỉnh và cặp nhầm bị suy giảm nhất quán, nên hướng tiếp theo cần phân tích đặc
trưng âm học/prosody của các nhóm 17, 30, 22 so với 38, 70, 14 và kiểm tra cơ chế
fusion hoặc calibration theo tỉnh. H7 không thay đổi kết luận H6, nhưng chỉ ra
rằng một con số accuracy tổng hợp chưa mô tả hết hành vi của mô hình.

## 19. H8: thời lượng, confidence và nhóm tỉnh trọng điểm

H8 được thiết kế như phân tích hậu nghiệm trên prediction H6 đã khóa, không huấn
luyện lại và không chạy backbone. Script đọc header audio của repaired test để
lấy thời lượng thật, đồng thời dùng thời lượng hiệu dụng sau giới hạn 20 giây đúng
như input mô hình.

Các phân tích được khai báo trước gồm bucket thời lượng `[0,2)`, `[2,4)`,
`[4,6)`, `[6,10)`, `[10,20]` giây; bucket confidence `[0,0.4)`, `[0.4,0.6)`,
`[0.6,0.8)`, `[0.8,1]`; và nhóm tỉnh trọng điểm 17, 30, 22, 38, 70, 14, 11.
Mục tiêu là định vị xu hướng, không chọn lại hyperparameter từ test.
