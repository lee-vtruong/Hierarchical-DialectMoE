# Hướng dẫn H11: benchmark backbone tiếng Việt

## 1. Mục tiêu và ma trận thí nghiệm

H11 thay backbone `facebook/wav2vec2-base` bằng hai model pretrained trên khoảng
13.000 giờ tiếng Việt:

- `nguyenvulebinh/wav2vec2-base-vi` (~95M tham số).
- `nguyenvulebinh/wav2vec2-large-vi` (~317M tham số).

Mỗi backbone chạy:

- Acoustic-only.
- Acoustic + prosody.
- Seed 42, 43, 44.

Tổng cộng 12 lần train. Mỗi lần đánh giá cả checkpoint tốt nhất theo province và
checkpoint tốt nhất theo region, tạo 24 kết quả test. Tất cả dùng repaired
speaker-disjoint manifest của H6.

Base dùng batch 4, accumulation 8. Large dùng batch 2, accumulation 16 để giữ
effective batch size 32 và giảm nguy cơ OOM.

## 2. Cập nhật server

```bash
cd /raid/hvtham/whale/Hierarchical-DialectMoE
git pull --rebase origin main
source .venv/bin/activate
python -m pytest -q
mkdir -p logs outputs
```

Kiểm tra PyTorch/CUDA:

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
"
```

Hai backbone chỉ phát hành `pytorch_model.bin`, vì vậy cần PyTorch >= 2.6. Nếu
thấp hơn 2.6, dừng và gửi kết quả kiểm tra; không nâng PyTorch tùy tiện vì có thể
làm hỏng CUDA environment đang hoạt động.

## 3. Preflight tải và kiểm tra backbone

Chạy trên login/CPU; lệnh tải khoảng 380 MB cho base và 1,27 GB cho large:

```bash
python scripts/check_backbone.py \
  --config configs/experiments/h11_base_vi_prosody.yaml

python scripts/check_backbone.py \
  --config configs/experiments/h11_large_vi_prosody.yaml
```

Hai lệnh phải in ra backbone, hidden size và số tham số, không có traceback.
Preflight chỉ tải/khởi tạo model, không train và không ghi checkpoint.

## 4. Chọn ba GPU

### Quản lý dung lượng checkpoint

Các cấu hình H11 bật `compact_best_checkpoints`: `last.pt` vẫn chứa optimizer và
scheduler để resume, còn hai checkpoint tốt nhất theo region/province chỉ chứa
model và metadata cần cho evaluation. H11 không ghi `best_loss.pt` và `best.pt`
vì hai file này không nằm trong ma trận đánh giá. Sau khi một run đã train và
evaluate thành công, có thể tải `last.pt` về nơi lưu trữ khác rồi xóa bản trên
server; không xóa hai checkpoint region/province trước khi hoàn tất tổng hợp.

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv
```

Nên dùng ba GPU còn ít nhất 35--40 GB trống. Các ví dụ dưới dùng GPU 1, 7, 2;
thay số GPU theo trạng thái thực tế.

## 5. Giai đoạn A: base-vi

Mỗi GPU chạy acoustic rồi prosody của một seed. Hàm sau train, đánh giá checkpoint
province, sau đó đánh giá thêm checkpoint region:

```bash
run_h11_pair () {
  gpu="$1"
  seed="$2"
  acoustic_config="$3"
  prosody_config="$4"

  nohup env CUDA_VISIBLE_DEVICES="$gpu" bash -c "
set -e

.venv/bin/python -u scripts/run_experiment.py \
  --config '$acoustic_config'

.venv/bin/python -u scripts/run_experiment.py \
  --config '$acoustic_config' \
  --checkpoint 'outputs/h11_base_vi_acoustic_seed${seed}/best_region_accuracy.pt' \
  --split test --skip-train

.venv/bin/python -u scripts/run_experiment.py \
  --config '$prosody_config'

.venv/bin/python -u scripts/run_experiment.py \
  --config '$prosody_config' \
  --checkpoint 'outputs/h11_base_vi_prosody_seed${seed}/best_region_accuracy.pt' \
  --split test --skip-train
" > "logs/h11_base_vi_seed${seed}.log" 2>&1 &

  echo $! > "logs/h11_base_vi_seed${seed}.pid"
}

run_h11_pair 1 42 \
  configs/experiments/h11_base_vi_acoustic.yaml \
  configs/experiments/h11_base_vi_prosody.yaml

run_h11_pair 7 43 \
  configs/experiments/h11_base_vi_acoustic_seed43.yaml \
  configs/experiments/h11_base_vi_prosody_seed43.yaml

run_h11_pair 2 44 \
  configs/experiments/h11_base_vi_acoustic_seed44.yaml \
  configs/experiments/h11_base_vi_prosody_seed44.yaml
```

Theo dõi:

```bash
for seed in 42 43 44; do
  pid=$(cat "logs/h11_base_vi_seed${seed}.pid")
  if ps -p "$pid" > /dev/null; then
    echo "base seed ${seed}: ĐANG CHẠY PID=${pid}"
  else
    echo "base seed ${seed}: ĐÃ DỪNG"
  fi
  tail -n 4 "logs/h11_base_vi_seed${seed}.log"
  echo
done
```

Kiểm tra lỗi:

```bash
grep -HniE \
  "out of memory|CUDA out of memory|Traceback|RuntimeError|Error|Killed" \
  logs/h11_base_vi_seed*.log
```

## 6. Xác nhận base-vi hoàn tất

```bash
for seed in 42 43 44; do
  for variant in acoustic prosody; do
    dir="outputs/h11_base_vi_${variant}_seed${seed}"
    for checkpoint in province region; do
      file="${dir}/metrics_test_best_${checkpoint}_accuracy.json"
      test -f "$file" && echo "OK $file" || echo "THIẾU $file"
    done
  done
done
```

Phải có 12 file `OK` trước khi chuyển sang large.

## 7. Giai đoạn B: large-vi

Không giữ hàm cũ vì đường dẫn output khác. Khai báo hàm large:

```bash
run_h11_large_pair () {
  gpu="$1"
  seed="$2"
  acoustic_config="$3"
  prosody_config="$4"

  nohup env CUDA_VISIBLE_DEVICES="$gpu" bash -c "
set -e

.venv/bin/python -u scripts/run_experiment.py \
  --config '$acoustic_config'

.venv/bin/python -u scripts/run_experiment.py \
  --config '$acoustic_config' \
  --checkpoint 'outputs/h11_large_vi_acoustic_seed${seed}/best_region_accuracy.pt' \
  --split test --skip-train

.venv/bin/python -u scripts/run_experiment.py \
  --config '$prosody_config'

.venv/bin/python -u scripts/run_experiment.py \
  --config '$prosody_config' \
  --checkpoint 'outputs/h11_large_vi_prosody_seed${seed}/best_region_accuracy.pt' \
  --split test --skip-train
" > "logs/h11_large_vi_seed${seed}.log" 2>&1 &

  echo $! > "logs/h11_large_vi_seed${seed}.pid"
}

run_h11_large_pair 1 42 \
  configs/experiments/h11_large_vi_acoustic.yaml \
  configs/experiments/h11_large_vi_prosody.yaml

run_h11_large_pair 7 43 \
  configs/experiments/h11_large_vi_acoustic_seed43.yaml \
  configs/experiments/h11_large_vi_prosody_seed43.yaml

run_h11_large_pair 2 44 \
  configs/experiments/h11_large_vi_acoustic_seed44.yaml \
  configs/experiments/h11_large_vi_prosody_seed44.yaml
```

Nếu large OOM, giảm `training.batch_size` xuống 1 và tăng
`gradient_accumulation_steps` lên 32 trong bốn config large gốc rồi chạy lại.
Không đổi riêng một seed.

Theo dõi:

```bash
for seed in 42 43 44; do
  pid=$(cat "logs/h11_large_vi_seed${seed}.pid")
  if ps -p "$pid" > /dev/null; then
    echo "large seed ${seed}: ĐANG CHẠY PID=${pid}"
  else
    echo "large seed ${seed}: ĐÃ DỪNG"
  fi
  tail -n 4 "logs/h11_large_vi_seed${seed}.log"
  echo
done
```

## 8. Xác nhận đủ 24 metrics

```bash
find outputs -maxdepth 2 \
  -path 'outputs/h11_*_vi_*_seed*/metrics_test_best_*_accuracy.json' \
  | wc -l
```

Kết quả phải là `24`.

## 9. Tổng hợp H11

```bash
python scripts/summarize_h11.py \
  --outputs outputs \
  --destination outputs/h11_summary.csv \
  --aggregate-destination outputs/h11_aggregate.csv

cat outputs/h11_aggregate.csv
```

Mỗi group phải có `runs = 3`. Có tám group:

```text
2 backbone × 2 variant × 2 checkpoint
```

## 10. Kiểm định prosody trong từng backbone

Ưu tiên checkpoint `best_province_accuracy`:

```bash
for backbone in base large; do
  for seed in 42 43 44; do
    python scripts/compare_predictions.py \
      --baseline "outputs/h11_${backbone}_vi_acoustic_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
      --candidate "outputs/h11_${backbone}_vi_prosody_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
      --output "outputs/h11_${backbone}_vi_prosody_vs_acoustic_seed${seed}.json" \
      --bootstrap-iterations 10000 \
      --seed "$seed"
  done
done
```

## 11. So sánh backbone tiếng Việt với model H6

```bash
for backbone in base large; do
  for seed in 42 43 44; do
    python scripts/compare_predictions.py \
      --baseline "outputs/h6_speaker_disjoint_prosody_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
      --candidate "outputs/h11_${backbone}_vi_prosody_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
      --output "outputs/h11_${backbone}_vi_vs_original_prosody_seed${seed}.json" \
      --bootstrap-iterations 10000 \
      --seed "$seed"
  done
done
```

## 12. Tiêu chí kết luận

So sánh bài gốc dùng macro-F1:

- Mốc region macro-F1: 0,9147.
- Mốc province macro-F1: 0,4107.
- Model H6 hiện tại: region macro-F1 khoảng 0,8977 và province macro-F1 0,4368.

Backbone chỉ được coi là cải thiện đồng thời nếu:

1. Region macro-F1 tăng ổn định qua ba seed.
2. Province macro-F1 không giảm so với H6.
3. Paired confidence interval hỗ trợ kết luận, không chỉ một seed.
4. Không chọn checkpoint bằng test: báo cáo riêng checkpoint province và region.

## 13. File cần gửi

```text
outputs/h11_summary.csv
outputs/h11_aggregate.csv
outputs/h11_base_vi_prosody_vs_acoustic_seed42.json
outputs/h11_base_vi_prosody_vs_acoustic_seed43.json
outputs/h11_base_vi_prosody_vs_acoustic_seed44.json
outputs/h11_large_vi_prosody_vs_acoustic_seed42.json
outputs/h11_large_vi_prosody_vs_acoustic_seed43.json
outputs/h11_large_vi_prosody_vs_acoustic_seed44.json
outputs/h11_base_vi_vs_original_prosody_seed42.json
outputs/h11_base_vi_vs_original_prosody_seed43.json
outputs/h11_base_vi_vs_original_prosody_seed44.json
outputs/h11_large_vi_vs_original_prosody_seed42.json
outputs/h11_large_vi_vs_original_prosody_seed43.json
outputs/h11_large_vi_vs_original_prosody_seed44.json
```
