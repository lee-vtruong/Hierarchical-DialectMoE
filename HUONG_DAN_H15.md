# Hướng dẫn H15: Temporal Prosody Adapter

## 1. Mục tiêu và phạm vi

H15-B kiểm tra giả thuyết rằng **diễn biến prosody theo thời gian** cung cấp
thông tin tỉnh tốt hơn một vector thống kê toàn câu. Pipeline mới trích sáu
chuỗi đặc trưng theo frame (log-RMS, ZCR, spectral centroid, bandwidth,
rolloff và F0 autocorrelation), chuẩn hóa trong từng utterance, lấy tối đa 256
frame phân bố đều, rồi dùng cross-attention để điều kiện hóa chuỗi acoustic của
Wav2Vec2-Large-VI. Một gated residual bảo toàn đường acoustic gốc.

H15 **không phải MoE**. H14 cho thấy router MoE-2 gần như sụp về một expert;
do đó baseline chính của H15 là `h11_large_vi_prosody`, không phải H14.

H15-A (ViP-VL) chưa được giả lập bằng `AutoModel`: ViP-VL dùng official
ChunkFormer/log-Mel pipeline. Chỉ gọi đó là controlled baseline sau khi tích
hợp đúng implementation chính thức.

## 2. Cập nhật và kiểm tra môi trường

```bash
cd /home/stackops/whale/Hierarchical-DialectMoE
git pull --ff-only origin main
conda activate /home/stackops/whale/envs/hierarchical-dialect-moe

python -m pip check
python -m pytest -q tests/test_components.py tests/test_config.py tests/test_train_checkpoint.py
```

Kết quả mong đợi tối thiểu: các test đều `passed`.

## 3. Smoke test trước khi chạy thật

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python -u scripts/run_experiment.py \
  --config configs/experiments/h15_temporal_prosody_smoke.yaml \
  --max-samples 16 2>&1 | tee logs/h15_smoke.log
```

Kiểm tra:

```bash
test -f outputs/h15_temporal_prosody_smoke/metrics_test_best_province_accuracy.json \
  && echo "SMOKE OK" || echo "SMOKE THIẾU KẾT QUẢ"
```

Không chạy full nếu smoke test có traceback, CUDA OOM hoặc NaN.

## 4. Chạy ba seed trên hai RTX 5090

GPU 0 chạy seed 42 rồi tự động chạy seed 44. GPU 1 chạy seed 43. Cách này
không đặt hai model Large-VI lên cùng một GPU 32 GB.

```bash
cd /home/stackops/whale/Hierarchical-DialectMoE
conda activate /home/stackops/whale/envs/hierarchical-dialect-moe
mkdir -p logs

nohup env CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  bash -lc '
    set -euo pipefail
    conda activate /home/stackops/whale/envs/hierarchical-dialect-moe
    cd /home/stackops/whale/Hierarchical-DialectMoE
    python -u scripts/run_experiment.py --config configs/experiments/h15_temporal_prosody_seed42.yaml
    python -u scripts/run_experiment.py --config configs/experiments/h15_temporal_prosody_seed44.yaml
  ' > logs/h15_seed42_44.log 2>&1 &
echo $! > logs/h15_seed42_44.pid

nohup env CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  bash -lc '
    set -euo pipefail
    conda activate /home/stackops/whale/envs/hierarchical-dialect-moe
    cd /home/stackops/whale/Hierarchical-DialectMoE
    python -u scripts/run_experiment.py --config configs/experiments/h15_temporal_prosody_seed43.yaml
  ' > logs/h15_seed43.log 2>&1 &
echo $! > logs/h15_seed43.pid
```

Nếu `conda activate` không hoạt động trong `bash -lc`, thay hai dòng `python`
bằng đường dẫn tuyệt đối:

```text
/home/stackops/whale/envs/hierarchical-dialect-moe/bin/python
```

## 5. Theo dõi

```bash
watch -n 10 nvidia-smi
```

Trong terminal khác:

```bash
tail -f logs/h15_seed42_44.log
tail -f logs/h15_seed43.log
```

Kiểm tra PID và lỗi:

```bash
for name in h15_seed42_44 h15_seed43; do
  pid=$(cat "logs/${name}.pid")
  if ps -p "$pid" >/dev/null; then
    echo "$name: ĐANG CHẠY PID=$pid"
  else
    echo "$name: ĐÃ DỪNG"
  fi
done

grep -HniE "Traceback|OutOfMemory|out of memory|CUDA error|Killed|nan" logs/h15_*.log
```

`ĐÃ DỪNG` không đồng nghĩa lỗi. Nếu cuối log có `Experiment complete` và đủ
artifact thì job đã hoàn tất.

## 6. Xác nhận 3/3 kết quả

```bash
for seed in 42 43 44; do
  dir="outputs/h15_temporal_prosody_seed${seed}"
  for file in \
    metrics_test_best_province_accuracy.json \
    predictions_test_best_province_accuracy.jsonl; do
    test -f "$dir/$file" && echo "OK: $dir/$file" || echo "THIẾU: $dir/$file"
  done
done
```

Không dùng `last.pt` để báo cáo nếu đã có `best_province_accuracy.pt`.

## 7. Tổng hợp cùng H11

Script tổng hợp hiện có thể quét cả hai họ experiment:

```bash
python scripts/summarize_experiments.py \
  --outputs outputs \
  --pattern 'h15_temporal_prosody_seed*' \
  --destination results_archive/h15_summary.csv \
  --aggregate-destination results_archive/h15_aggregate.csv
```

So sánh paired phải dùng prediction của cùng seed:

```bash
for seed in 42 43 44; do
  python scripts/compare_predictions.py \
    --baseline "outputs/h11_large_vi_prosody_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
    --candidate "outputs/h15_temporal_prosody_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
    --output "results_archive/h15_vs_h11_seed${seed}.json"
done
```

Xem `python scripts/compare_predictions.py -h` trước nếu checkout trên server
có tên option khác; không tự đổi file prediction hoặc seed.

## 8. Quy tắc quyết định

- **Tiếp tục H15-C** nếu province macro-F1 tăng ở ít nhất 2/3 seed và mean tăng
  khoảng 1 điểm trở lên.
- Nếu tăng nhỏ hơn 0.5 điểm hoặc không nhất quán, kiểm tra attention/gate và
  temporal normalization trước khi thêm objective khác.
- Nếu H15-B có lợi, bước tiếp theo là speaker-adversarial head và validation
  speaker probe; chưa mở test thêm trong quá trình chọn cấu hình.
- Region đã gần bão hòa, province macro-F1 là primary metric.

## 9. Artifact cần tải về

```bash
tar -czf h15_complete.tar.gz \
  results_archive/h15*.csv \
  results_archive/h15_vs_h11_seed*.json \
  outputs/h15_temporal_prosody_seed*/metrics_test_best_province_accuracy.json \
  outputs/h15_temporal_prosody_seed*/predictions_test_best_province_accuracy.jsonl \
  logs/h15_*.log
```

Không đưa checkpoint `.pt`, dataset hay Hugging Face cache lên GitHub.

