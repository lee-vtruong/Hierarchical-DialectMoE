# Hướng dẫn H13: phân tích mô hình cuối và efficiency

H13 không huấn luyện lại. Phân tích chính so sánh Large-VI acoustic-only với
Large-VI acoustic+prosody qua ba seed, sử dụng checkpoint
`best_province_accuracy` đã khóa.

## 1. Sinh prediction validation để fit calibration

H13 cần sáu prediction validation. Chạy tuần tự ba seed trên mỗi GPU.

### GPU 0: Large-VI acoustic

```bash
nohup env CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  bash -lc '
set -euo pipefail
cd /home/stackops/whale/Hierarchical-DialectMoE
PYTHON=/home/stackops/whale/envs/hierarchical-dialect-moe/bin/python
for seed in 42 43 44; do
  config="configs/experiments/h11_large_vi_acoustic.yaml"
  [ "$seed" != 42 ] && config="configs/experiments/h11_large_vi_acoustic_seed${seed}.yaml"
  "$PYTHON" -u scripts/run_experiment.py \
    --config "$config" \
    --checkpoint "outputs/h11_large_vi_acoustic_seed${seed}/best_province_accuracy.pt" \
    --split valid --skip-train
done
' > logs/h13_valid_acoustic.log 2>&1 &
echo $! > logs/h13_valid_acoustic.pid
```

### GPU 1: Large-VI prosody

```bash
nohup env CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  bash -lc '
set -euo pipefail
cd /home/stackops/whale/Hierarchical-DialectMoE
PYTHON=/home/stackops/whale/envs/hierarchical-dialect-moe/bin/python
for seed in 42 43 44; do
  config="configs/experiments/h11_large_vi_prosody.yaml"
  [ "$seed" != 42 ] && config="configs/experiments/h11_large_vi_prosody_seed${seed}.yaml"
  "$PYTHON" -u scripts/run_experiment.py \
    --config "$config" \
    --checkpoint "outputs/h11_large_vi_prosody_seed${seed}/best_province_accuracy.pt" \
    --split valid --skip-train
done
' > logs/h13_valid_prosody.log 2>&1 &
echo $! > logs/h13_valid_prosody.pid
```

Kiểm tra đủ sáu file:

```bash
for variant in acoustic prosody; do
  for seed in 42 43 44; do
    file="outputs/h11_large_vi_${variant}_seed${seed}/predictions_valid_best_province_accuracy.jsonl"
    test -s "$file" && echo "OK: $file" || echo "THIẾU: $file"
  done
done
```

## 2. Chạy phân tích H13

```bash
conda activate /home/stackops/whale/envs/hierarchical-dialect-moe
mkdir -p logs results_archive/h13

nohup python -u scripts/analyze_h13.py \
  --outputs outputs \
  --destination results_archive/h13 \
  --seeds 42 43 44 \
  --calibration-bins 15 \
  > logs/h13.log 2>&1 &

echo $! > logs/h13.pid
```

H13 sinh:

- Accuracy và chuyển trạng thái theo từng tỉnh.
- Các confusion pair chính.
- Calibration thô của acoustic/prosody.
- Temperature scaling fit riêng trên repaired validation rồi áp dụng test.
- ECE, NLL, Brier score trước/sau calibration.
- Metadata kích thước checkpoint và số sample prediction.

## 3. Benchmark efficiency có kiểm soát

Chỉ chạy khi GPU trống. Benchmark dùng cùng 64 mẫu, batch size 1, bỏ qua thời
gian đọc disk và trích đặc trưng CPU; số đo phản ánh forward pass GPU. Chạy các
cấu hình tuần tự trên cùng một GPU để tránh nhiễu do tranh chấp tài nguyên.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_h13.py \
  --config configs/experiments/h11_base_vi_prosody.yaml \
  --checkpoint outputs/h11_base_vi_prosody_seed42/best_province_accuracy.pt \
  --max-samples 64 --batch-size 1 --warmup-repeats 1 --timed-repeats 3 \
  --output results_archive/h13/benchmark_base_prosody_seed42.json

CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_h13.py \
  --config configs/experiments/h11_large_vi_acoustic.yaml \
  --checkpoint outputs/h11_large_vi_acoustic_seed42/best_province_accuracy.pt \
  --max-samples 64 --batch-size 1 --warmup-repeats 1 --timed-repeats 3 \
  --output results_archive/h13/benchmark_large_acoustic_seed42.json

CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_h13.py \
  --config configs/experiments/h11_large_vi_prosody.yaml \
  --checkpoint outputs/h11_large_vi_prosody_seed42/best_province_accuracy.pt \
  --max-samples 64 --batch-size 1 --warmup-repeats 1 --timed-repeats 3 \
  --output results_archive/h13/benchmark_large_prosody_seed42.json
```

Benchmark chỉ mô tả chi phí trên RTX 5090 của server này; không suy rộng trực
tiếp sang GPU khác. Không chạy đồng thời job khác trên cùng GPU.

## 4. Kiểm tra artifact

```bash
find results_archive/h13 -maxdepth 2 -type f | sort
test -s results_archive/h13/h13_summary.json && echo OK_summary
test -s results_archive/h13/error_analysis/h7_summary.json && echo OK_errors
test -s results_archive/h13/calibration/h9_aggregate.csv && echo OK_calibration
test -s results_archive/h13/h13_artifact_metadata.csv && echo OK_metadata
```

Tải toàn bộ `results_archive/h13/` về để lưu cùng paper artifact.
