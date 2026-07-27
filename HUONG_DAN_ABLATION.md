# Hướng dẫn chạy baseline và ablation

Tài liệu này mô tả chuỗi thí nghiệm tiếp theo sau kết quả MVP:

- Region test accuracy: 89,54%.
- Province test accuracy: 44,18%.
- Province macro-F1: 43,80%.
- Router bị collapse vào hai trong tám expert.

## 1. Các cấu hình có sẵn

| Config | Acoustic | Prosody | Hierarchical router | MoE | Mục đích |
|---|---:|---:|---:|---:|---|
| `acoustic_only.yaml` | Có | Không | Không | Không | Baseline chính |
| `acoustic_prosody.yaml` | Có | Có | Không | Không | Đo đóng góp prosody |
| `flat_moe_8_balanced.yaml` | Có | Có | Không | 8 | Flat MoE baseline |
| `moe_2_balanced.yaml` | Có | Có | Có | 2, top-1 | So sánh số expert |
| `moe_4_balanced.yaml` | Có | Có | Có | 4, top-2 | So sánh số expert |
| `moe_8_balanced.yaml` | Có | Có | Có | 8, top-2 | Chống expert collapse |

Các config kế thừa `configs/vimd_moe.yaml`. Chỉ những trường cần thay đổi mới
được ghi lại trong file thí nghiệm.

## 2. Thứ tự chạy

Chạy lần lượt, không chạy đồng thời vào cùng GPU:

1. `acoustic_only`.
2. `acoustic_prosody`.
3. `flat_moe_8_balanced`.
4. `moe_2_balanced`.
5. `moe_4_balanced`.
6. `moe_8_balanced`.

Hai thí nghiệm đầu tiên quan trọng nhất. Nếu thêm prosody không cải thiện
acoustic-only, cần xem lại bộ đặc trưng trước khi đánh giá MoE.

## 3. Cập nhật code trên server

```bash
cd /raid/hvtham/whale/Hierarchical-DialectMoE
git pull --rebase origin main
source .venv/bin/activate
python -m pytest -q
```

## 4. Smoke test một cấu hình

Ví dụ acoustic-only:

```bash
export CUDA_VISIBLE_DEVICES=7

python scripts/run_experiment.py \
  --config configs/experiments/acoustic_only.yaml \
  --max-samples 32
```

Smoke test chỉ xác nhận code chạy được. Không dùng metrics của subset 32 mẫu
trong báo cáo.

Sau smoke test, đổi tên output hoặc xóa nó trước full run. Cách an toàn là đổi
tên:

```bash
mv outputs/acoustic_only_seed42 outputs/smoke_acoustic_only_seed42
```

## 5. Chạy bằng nohup

### 5.1 Acoustic-only

```bash
mkdir -p logs

nohup env CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -u scripts/run_experiment.py \
  --config configs/experiments/acoustic_only.yaml \
  > logs/acoustic_only_seed42.log 2>&1 &

echo $! > logs/acoustic_only_seed42.pid
```

Theo dõi:

```bash
tail -f logs/acoustic_only_seed42.log
```

### 5.2 Acoustic + prosody, không MoE

Chỉ chạy sau khi acoustic-only kết thúc:

```bash
nohup env CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -u scripts/run_experiment.py \
  --config configs/experiments/acoustic_prosody.yaml \
  > logs/acoustic_prosody_seed42.log 2>&1 &

echo $! > logs/acoustic_prosody_seed42.pid
```

### 5.3 Flat MoE

```bash
nohup env CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -u scripts/run_experiment.py \
  --config configs/experiments/flat_moe_8_balanced.yaml \
  > logs/flat_moe_8_balanced_seed42.log 2>&1 &

echo $! > logs/flat_moe_8_balanced_seed42.pid
```

### 5.4 Hierarchical MoE với 2 expert

```bash
nohup env CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -u scripts/run_experiment.py \
  --config configs/experiments/moe_2_balanced.yaml \
  > logs/moe_2_balanced_seed42.log 2>&1 &

echo $! > logs/moe_2_balanced_seed42.pid
```

### 5.5 Hierarchical MoE với 4 expert

```bash
nohup env CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -u scripts/run_experiment.py \
  --config configs/experiments/moe_4_balanced.yaml \
  > logs/moe_4_balanced_seed42.log 2>&1 &

echo $! > logs/moe_4_balanced_seed42.pid
```

### 5.6 Hierarchical MoE với 8 expert, chống collapse

```bash
nohup env CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -u scripts/run_experiment.py \
  --config configs/experiments/moe_8_balanced.yaml \
  > logs/moe_8_balanced_seed42.log 2>&1 &

echo $! > logs/moe_8_balanced_seed42.pid
```

## 6. Kiểm tra tiến trình

Ví dụ:

```bash
ps -fp "$(cat logs/acoustic_only_seed42.pid)"
```

Kiểm tra GPU:

```bash
nvidia-smi
```

Dừng đúng tiến trình:

```bash
kill "$(cat logs/acoustic_only_seed42.pid)"
```

## 7. Output của mỗi thí nghiệm

Ví dụ acoustic-only:

```text
outputs/acoustic_only_seed42/
|-- config.json
|-- labels.json
|-- best.pt
|-- best_loss.pt
|-- best_region_accuracy.pt
|-- best_province_accuracy.pt
|-- last.pt
|-- metrics_test.json
`-- metrics_test_best_province_accuracy.json
```

`run_experiment.py` ưu tiên đánh giá `best_province_accuracy.pt`, sau đó mới
dùng `last.pt` hoặc `best.pt`.

Đối với cấu hình không dùng MoE, routing metrics không có ý nghĩa nghiên cứu
vì router không tham gia tạo province features.

## 8. Tổng hợp kết quả

Sau khi có ít nhất hai thí nghiệm:

```bash
python scripts/summarize_experiments.py \
  --outputs outputs \
  --destination outputs/experiment_summary.csv
```

Xem bảng:

```bash
column -s, -t < outputs/experiment_summary.csv | less -S
```

Các metrics chính cần so sánh:

- Region macro-F1.
- Province macro-F1.
- Province balanced accuracy.
- Province accuracy.
- Expert probabilities.
- Router entropy.

## 9. Cách kết luận từng bước

### Acoustic-only so với acoustic + prosody

Nếu prosody cải thiện ổn định qua nhiều seed, giả thuyết H1 có bằng chứng hỗ
trợ. Một lần chạy seed 42 chỉ là kết quả ban đầu.

### Acoustic + prosody so với flat MoE

Đo đóng góp của MoE khi chưa dùng cấu trúc phân cấp.

### Flat MoE so với hierarchical MoE

Đo đóng góp của region-conditioned routing, liên quan giả thuyết H4.

### MoE 2/4/8 expert

Đánh giá trade-off giữa capacity và expert collapse. Không chỉ chọn model theo
accuracy; cần kiểm tra expert usage.

## 10. Nhiều seed

Sau khi tìm được 1-2 cấu hình tốt nhất, sao chép config và đổi:

```yaml
seed: 43

training:
  output_dir: outputs/<ten_thi_nghiem>_seed43
```

Tiếp tục với seed 44. Báo cáo mean và standard deviation của ba seed.

