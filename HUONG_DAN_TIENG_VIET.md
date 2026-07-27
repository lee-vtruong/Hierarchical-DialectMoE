# Hướng dẫn đầy đủ - Hierarchical DialectMoE

Tài liệu này hướng dẫn cài đặt, tải dữ liệu, huấn luyện, đánh giá và quản lý
thí nghiệm cho hệ thống nhận diện phương ngữ tiếng Việt Hierarchical
DialectMoE.

## 1. Hệ thống hiện tại làm được gì?

Pipeline hiện tại:

```text
Audio thô
  |
  +--> Pretrained Acoustic Encoder (mặc định Wav2Vec2 Base)
  |      |
  |      +--> Masked Mean Pooling
  |      +--> Acoustic Projection
  |
  +--> Prosody Extractor
         |
         +--> Duration
         +--> RMS Energy (mean/std)
         +--> Zero-Crossing Rate
         +--> Spectral Centroid
         +--> Spectral Bandwidth
         +--> Spectral Rolloff
         +--> F0 (mean/std/min/max)
         +--> Voiced Fraction

Acoustic embedding + Prosody embedding
  |
  +--> Gated Fusion
  +--> Region Head: Bắc / Trung / Nam
  +--> Region-conditioned Router
  +--> Sparse Top-k Mixture-of-Experts
  +--> Province Head: phân loại tỉnh/thành
```

Hệ thống huấn luyện đồng thời:

- Nhận diện vùng phương ngữ: Bắc, Trung và Nam.
- Nhận diện phương ngữ cấp tỉnh.
- Học cách điều hướng mẫu tới các expert.
- Cân bằng lượng dữ liệu được chuyển tới các expert.

## 2. Cấu trúc repository

```text
HierarchicalDialectMoE/
|-- configs/
|   `-- vimd_moe.yaml          # Cấu hình dữ liệu, mô hình và huấn luyện
|-- dialect_moe/
|   |-- config.py              # Đọc YAML
|   |-- data.py                # ViMD loader và batch collator
|   |-- labels.py              # Mã hóa nhãn vùng/tỉnh
|   |-- losses.py              # Hàm mất mát đa nhiệm
|   |-- model.py               # Hierarchical Prosody-aware MoE
|   |-- prosody.py             # Trích đặc trưng ngữ điệu/phổ
|   `-- utils.py
|-- scripts/
|   |-- download_data.py       # Tải ViMD/ViSpeech
|   |-- train.py               # Huấn luyện và resume
|   `-- evaluate.py            # Đánh giá test
|-- tests/
|   `-- test_components.py
|-- requirements.txt
|-- pyproject.toml
`-- README.md
```

Các thư mục sau không được đưa lên GitHub:

- `data/`: dữ liệu âm thanh.
- `outputs/`: checkpoint và kết quả.
- `external/`: repository tham khảo.
- `*.pdf`: bài báo và proposal.

## 3. Yêu cầu server

Khuyến nghị:

- Linux Ubuntu 20.04/22.04 hoặc tương đương.
- Python 3.10 hoặc 3.11.
- CUDA tương thích với PyTorch.
- Tối thiểu 100 GB trống cho ViMD và cache.
- GPU từ 16 GB VRAM trở lên để fine-tune thuận tiện.
- RAM từ 32 GB.

GPU ít VRAM vẫn có thể chạy bằng cách giảm:

- `batch_size`.
- `max_seconds`.
- Kích thước backbone.
- Số lượng expert.

## 4. Đưa code lên GitHub

Không chạy `git add data` và không đưa dữ liệu ViMD lên GitHub.

```bash
git status
git add .gitignore README.md HUONG_DAN_TIENG_VIET.md requirements.txt \
  pyproject.toml pytest.ini configs dialect_moe scripts tests
git commit -m "Implement hierarchical prosody-aware dialect MoE"
git branch -M main
git remote add origin <URL_REPOSITORY_CUA_BAN>
git push -u origin main
```

Nếu remote `origin` đã tồn tại:

```bash
git remote set-url origin <URL_REPOSITORY_CUA_BAN>
git push -u origin main
```

## 5. Pull code trên server

```bash
git clone <URL_REPOSITORY_CUA_BAN>
cd HierarchicalDialectMoE
```

Nếu repository đã có trên server:

```bash
git pull origin main
```

## 6. Tạo môi trường Python

### Cách 1: dùng venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Cách 2: dùng Conda

```bash
conda create -n dialect-moe python=3.11 -y
conda activate dialect-moe
python -m pip install --upgrade pip
```

## 7. Cài PyTorch và dependency

Nên cài bản PyTorch phù hợp CUDA của server trước. Ví dụ dưới đây chỉ mang
tính minh họa; cần lấy lệnh đúng từ trang cài đặt chính thức của PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Kiểm tra:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Kết quả mong đợi:

- `torch.cuda.is_available()` trả về `True`.
- Tên GPU được in chính xác.

## 8. Chạy unit test

```bash
pip install pytest
python -m pytest -q
```

Kết quả hiện tại:

```text
3 passed
```

Các test kiểm tra:

- Vector prosody có đúng kích thước và không chứa NaN/Inf.
- Sparse MoE trả về đúng shape và có gradient.
- Hàm mất mát phân cấp trả về giá trị hợp lệ.

## 9. Tải dataset

### 9.1 ViMD

ViMD gồm khoảng 102,56 giờ, gần 19.000 utterance và 63 phương ngữ cấp tỉnh.
Dung lượng repository Hugging Face khoảng 74,2 GB.

```bash
python scripts/download_data.py --dataset vimd --output-dir data
```

Dữ liệu được lưu tại:

```text
data/ViMD_Dataset/
|-- README.md
`-- data/
    |-- train-*.parquet
    |-- valid-*.parquet
    `-- test-*.parquet
```

Nếu download bị gián đoạn, chạy lại cùng lệnh. Hugging Face sẽ tái sử dụng
file/cache đã có.

ViMD dùng giấy phép CC BY-NC-ND 4.0. Không đưa dữ liệu lên GitHub và cần kiểm
tra điều kiện giấy phép trước khi sử dụng ngoài mục đích nghiên cứu.

### 9.2 ViSpeech (tùy chọn)

```bash
python scripts/download_data.py --dataset vispeech --output-dir data
```

ViSpeech hiện được dùng như nguồn cross-domain/noise test tiềm năng. Pipeline
train chính hiện tại dùng ViMD; chưa có loader thống nhất ViSpeech trong
`train.py`.

## 10. Cấu hình

File chính:

```text
configs/vimd_moe.yaml
```

### 10.1 Data

```yaml
data:
  dataset_name: nguyendv02/ViMD_Dataset
  local_dir: data/ViMD_Dataset
  cache_dir: data/huggingface
  sample_rate: 16000
  max_seconds: 20.0
  num_workers: 4
```

- `local_dir`: nơi script downloader lưu ViMD.
- Nếu `local_dir` không tồn tại, code tự tải dataset qua Hugging Face.
- `sample_rate`: audio được resample về 16 kHz.
- `max_seconds`: giới hạn độ dài mỗi utterance.
- `num_workers`: số worker chuẩn bị batch.

### 10.2 Model

```yaml
model:
  backbone: facebook/wav2vec2-base
  freeze_feature_encoder: true
  freeze_backbone: false
  gradient_checkpointing: true
  acoustic_dim: 256
  prosody_dim: 128
  fusion_dim: 384
  num_experts: 8
  top_k: 2
  expert_hidden_dim: 512
  dropout: 0.2
```

- `backbone`: acoustic pretrained model từ Hugging Face.
- `freeze_feature_encoder`: khóa convolutional frontend.
- `freeze_backbone`: khóa toàn bộ acoustic backbone.
- `gradient_checkpointing`: giảm VRAM, đổi lại train chậm hơn.
- `num_experts`: số expert.
- `top_k`: số expert được chọn cho mỗi mẫu.

Có thể đổi backbone, ví dụ:

```yaml
backbone: facebook/hubert-base-ls960
```

Backbone phải tương thích với `transformers.AutoModel` và nhận raw waveform.

### 10.3 Training

```yaml
training:
  epochs: 20
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 0.00002
  head_learning_rate: 0.0002
  mixed_precision: fp16
```

Backbone dùng `learning_rate`; các projection, router, expert và head dùng
`head_learning_rate`.

Batch hiệu dụng:

```text
effective_batch_size = batch_size * gradient_accumulation_steps
```

Với cấu hình mặc định:

```text
4 * 8 = 32
```

## 11. Kiểm thử nhanh trước khi train đầy đủ

Sau khi đã tải ViMD:

```bash
python scripts/train.py \
  --config configs/vimd_moe.yaml \
  --max-samples 32
```

Lệnh này dùng tối đa 32 mẫu cho mỗi split. Mục tiêu là kiểm tra:

- Dataset đọc được.
- Backbone tải được.
- Forward/backward chạy được.
- Checkpoint được ghi.

Không sử dụng kết quả 32 mẫu để báo cáo nghiên cứu.

## 12. Huấn luyện đầy đủ

```bash
python scripts/train.py --config configs/vimd_moe.yaml
```

Output:

```text
outputs/vimd_moe/
|-- config.json
|-- labels.json
|-- best.pt
`-- last.pt
```

- `best.pt`: checkpoint có validation loss tốt nhất.
- `last.pt`: checkpoint ở epoch gần nhất.
- `labels.json`: ánh xạ nhãn vùng và tỉnh.
- `config.json`: cấu hình thực tế của lần chạy.

## 13. Tiếp tục lần train bị gián đoạn

```bash
python scripts/train.py \
  --config configs/vimd_moe.yaml \
  --resume outputs/vimd_moe/last.pt
```

Checkpoint chứa:

- Trọng số model.
- Trạng thái optimizer.
- Trạng thái learning-rate scheduler.
- Epoch.
- Validation metrics.

## 14. Đánh giá

```bash
python scripts/evaluate.py \
  --config configs/vimd_moe.yaml \
  --checkpoint outputs/vimd_moe/best.pt \
  --split test
```

Kết quả được lưu tại:

```text
outputs/vimd_moe/metrics_test.json
```

Metrics hiện có:

- Region accuracy.
- Region balanced accuracy.
- Province accuracy.
- Province balanced accuracy.
- Precision/recall/F1 cho từng class.
- Macro average.
- Weighted average.
- Phân bố xác suất trung bình theo expert.
- Router entropy.

## 15. Cách chạy các ablation quan trọng

Mỗi thí nghiệm nên dùng output directory riêng để không ghi đè checkpoint.

### 15.1 Số expert

Tạo các config riêng:

```text
configs/experts_2.yaml
configs/experts_4.yaml
configs/experts_8.yaml
configs/experts_16.yaml
```

Thay đổi:

```yaml
model:
  num_experts: 4
  top_k: 2

training:
  output_dir: outputs/experts_4
```

### 15.2 Top-k routing

```yaml
model:
  num_experts: 8
  top_k: 1
```

So sánh `top_k` bằng 1, 2 và 4.

### 15.3 Frozen backbone

```yaml
model:
  freeze_backbone: true
```

Sau đó so sánh với fine-tune:

```yaml
model:
  freeze_backbone: false
```

### 15.4 Bỏ prosody

Code hiện chưa có cờ `use_prosody`. Để có ablation chuẩn, cần bổ sung cờ thay
vì sửa trực tiếp tensor hoặc source code. Đây là một phần còn thiếu được nêu
trong mục đối chiếu proposal bên dưới.

## 16. Khuyến nghị thiết kế thí nghiệm

Mỗi cấu hình nên chạy ít nhất 3 seed:

```yaml
seed: 42
```

Sau đó thử:

```yaml
seed: 43
```

và:

```yaml
seed: 44
```

Nên báo cáo:

- Mean và standard deviation.
- Macro-F1.
- Balanced accuracy.
- Accuracy theo từng vùng.
- Accuracy/F1 theo từng tỉnh.
- Expert usage.
- Router entropy.

Không chỉ báo cáo accuracy vì phân bố 63 tỉnh có thể không cân bằng.

## 17. Xử lý lỗi thường gặp

### CUDA out of memory

Giảm:

```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 16

data:
  max_seconds: 12.0
```

Hoặc:

```yaml
model:
  freeze_backbone: true
  num_experts: 4
```

### Không tìm thấy dataset

Kiểm tra:

```bash
ls data/ViMD_Dataset/data
```

Phải có các file `train-*.parquet`, `valid-*.parquet` và `test-*.parquet`.

### Không nhận GPU

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu `nvidia-smi` chạy nhưng PyTorch trả về `False`, thường là đã cài nhầm
PyTorch CPU hoặc CUDA wheel không phù hợp.

### Download backbone thất bại

Kiểm tra kết nối tới Hugging Face. Có thể đặt cache:

```bash
export HF_HOME=/duong-dan-o-dia-lon/huggingface
```

Không đặt cache trên phân vùng sắp hết dung lượng.

### DataLoader bị treo

Đặt:

```yaml
data:
  num_workers: 0
```

Nếu chạy ổn, tăng dần lên 2, 4 hoặc 8.

## 18. Đối chiếu code với proposal

### Những phần đã được cài đặt

| Thành phần proposal | Trạng thái code hiện tại |
|---|---|
| Stage 1 - Acoustic Encoder | Có, dùng `AutoModel`, mặc định Wav2Vec2 Base |
| Stage 2 - Prosody Encoder | Có, vector prosody và MLP |
| Stage 3 - Feature Fusion | Có, gated fusion |
| Level 1 Router Bắc/Trung/Nam | Có region head |
| Region-aware routing | Có, region probability điều kiện hóa expert router |
| Sparse Mixture-of-Experts | Có, top-k routing và load balancing |
| Province Classification | Có |
| Multi-task Region + Province | Có |
| Router entropy | Có |
| Load-balancing loss | Có |
| Checkpoint/resume | Có |
| Metrics classification cơ bản | Có |
| Expert usage và routing entropy | Có |

### Những phần chưa cài đặt hoặc chưa khớp hoàn toàn

| Thành phần proposal | Trạng thái |
|---|---|
| Level 2 - Router tiểu vùng | Chưa có nhãn và router tiểu vùng riêng |
| Level 3 - Province-specific expert | Hiện expert được học tự động, không phải mỗi tỉnh một expert |
| Pitch contour encoder theo thời gian | Chưa có; hiện dùng thống kê F0 toàn utterance |
| Jitter và shimmer | Chưa có |
| Speaking rate | Chưa có vì cần forced alignment/ASR hoặc ước lượng âm tiết |
| Pause duration | Chưa có |
| Formant F1/F2/F3 | Chưa có |
| CNN1D/BiLSTM/Transformer prosody encoder | Chưa có, hiện dùng MLP |
| Concat/Cross-attention/Bilinear fusion | Chưa có, hiện chỉ gated fusion |
| Hard/soft/random routing ablation | Chưa có cờ cấu hình đầy đủ |
| Acoustic-only/prosody-only ablation | Chưa có cờ cấu hình |
| ASR CTC head | Chưa có |
| Dialect + Province + ASR multi-task | Chưa có |
| Gender auxiliary head | Chưa có |
| ViSpeech/VDSPEC unified loader | Chưa có |
| Cross-domain experiment | Chưa có pipeline tự động |
| Noise robustness theo SNR | Chưa có |
| Low-resource sampling theo 10/20/50/100 | Chưa có pipeline riêng |
| Top-3/Top-5/MRR | Chưa có |
| WER/CER/TER | Chưa có vì chưa có ASR |
| FLOPs/latency/throughput/GPU memory | Chưa có báo cáo tự động |
| UMAP/t-SNE/SHAP/Integrated Gradients | Chưa có |
| Bootstrap/t-test/Wilcoxon/McNemar | Chưa có |

## 19. Kết luận về mức độ hoàn thiện

Code hiện tại là một **MVP nghiên cứu hợp lệ cho lõi ý tưởng**:

- Acoustic encoder.
- Prosody-aware gated fusion.
- Phân loại phân cấp vùng và tỉnh.
- Region-conditioned sparse MoE.
- Multi-task region/province.

Tuy nhiên, code **chưa phải toàn bộ hệ thống mô tả trong proposal**. Đặc biệt,
proposal còn yêu cầu router ba tầng, chuỗi prosody theo thời gian, ASR,
cross-domain/noise/low-resource experiments, đầy đủ ablation và kiểm định
thống kê.

Trình tự phát triển hợp lý:

1. Chạy và xác nhận baseline acoustic-only.
2. Hoàn thiện cờ ablation.
3. Bổ sung prosody contour và formant.
4. Bổ sung Level-2 subregion router.
5. Chạy đầy đủ dialect/province experiments.
6. Sau khi phần classification ổn định mới bổ sung ASR CTC.

