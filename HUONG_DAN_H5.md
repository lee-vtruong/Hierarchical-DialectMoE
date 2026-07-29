# Hướng dẫn H5: đặc trưng spectral/FFT

## 1. Mục tiêu

H5 kiểm tra liệu thông tin phổ FFT có bổ sung tín hiệu cho nhận diện phương ngữ
cấp vùng và tỉnh hay không.

Một điểm cần lưu ý: vector `prosody` legacy của các thí nghiệm H1 chứa ba đại
lượng phổ:

- Spectral centroid.
- Spectral bandwidth.
- Spectral roll-off.

Do đó H1 chính xác hơn là prosody + một phần spectral. H5 giữ nguyên chế độ
`legacy` để checkpoint cũ vẫn tái lập được, nhưng dùng bộ `pitch_energy` mới đã
loại ba đại lượng phổ để tạo ablation không chồng lấn.

## 2. Đặc trưng

### Pitch/energy

Bộ `pitch_energy` gồm 9 chiều:

- Log duration.
- RMS mean và standard deviation.
- Zero-crossing rate.
- F0 mean, standard deviation, minimum và maximum.
- Voiced fraction.

### Spectral/FFT

Bộ spectral gồm 24 chiều được trích từ STFT:

- Centroid mean/std.
- Bandwidth mean/std.
- Roll-off 85% mean/std.
- Spectral flatness mean/std.
- Spectral flux mean/std.
- 12 dải công suất FFT tương đối.
- Log low/high-frequency energy ratio.
- Spectral entropy.

Các đại lượng vị trí tần số được chuẩn hóa theo Nyquist. Band power là công suất
tương đối để giảm phụ thuộc âm lượng.

## 3. Bốn cấu hình seed 42

| Config | Backbone | Pitch/energy | Spectral | Mục tiêu |
|---|---:|---:|---:|---|
| `h5_acoustic_pitch_energy` | Có | Có | Không | Baseline sạch |
| `h5_acoustic_spectral` | Có | Không | Có | Đóng góp spectral |
| `h5_acoustic_pitch_energy_spectral` | Có | Có | Có | Fusion đầy đủ |
| `h5_handcrafted_pitch_energy_spectral` | Không | Có | Có | Baseline MLP đơn giản |

Tất cả cấu hình:

- Không MoE.
- Không hierarchical routing.
- Chọn checkpoint bằng province accuracy validation.
- Chỉ đánh giá validation trong giai đoạn chọn cấu hình.

## 4. Cập nhật và kiểm tra server

```bash
cd /raid/hvtham/whale/Hierarchical-DialectMoE
git pull --rebase origin main
source .venv/bin/activate
python -m pytest -q
mkdir -p logs
```

## 5. Smoke test H5

```bash
python scripts/run_experiment.py \
  --config configs/experiments/h5_smoke.yaml \
  --split valid \
  --max-samples 32
```

Smoke test phải tạo:

```text
outputs/h5_smoke/metrics_valid_best_province_accuracy.json
outputs/h5_smoke/predictions_valid_best_province_accuracy.jsonl
```

Vocabulary được dựng từ toàn bộ dataset trước khi cắt 32 mẫu, nên checkpoint
smoke vẫn có đúng head 3 vùng và 63 tỉnh. `--max-samples` được áp dụng đồng nhất
cho cả train và evaluate. Kết quả accuracy của smoke không có ý nghĩa nghiên cứu.

## 6. Kiểm tra GPU

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv
```

Chọn GPU có utilization thấp và còn nhiều VRAM. Baseline handcrafted không chạy
backbone nên nhẹ hơn nhiều và có thể đặt cùng GPU với một job khác.

## 7. Chạy bốn cấu hình

Ví dụ giả sử GPU 2, 6 và 7 trống:

```bash
CONFIGS=(
  h5_acoustic_pitch_energy
  h5_acoustic_spectral
  h5_acoustic_pitch_energy_spectral
  h5_handcrafted_pitch_energy_spectral
)

GPUS=(7 6 2 7)

for i in "${!CONFIGS[@]}"; do
  name="${CONFIGS[$i]}"
  gpu="${GPUS[$i]}"

  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -u scripts/run_experiment.py \
    --config "configs/experiments/${name}.yaml" \
    --split valid \
    > "logs/${name}_seed42.log" 2>&1 &

  pid=$!
  echo "$pid" > "logs/${name}_seed42.pid"
  echo "Started ${name} on GPU ${gpu}, PID=${pid}"
done
```

Phải thay `GPUS` theo trạng thái server thực tế.

## 8. Theo dõi

```bash
for name in \
  h5_acoustic_pitch_energy \
  h5_acoustic_spectral \
  h5_acoustic_pitch_energy_spectral \
  h5_handcrafted_pitch_energy_spectral
do
  pid=$(cat "logs/${name}_seed42.pid")
  log="logs/${name}_seed42.log"

  if ps -p "$pid" > /dev/null; then
    echo "RUNNING: $name"
  elif grep -q "Experiment complete" "$log"; then
    echo "DONE: $name"
  else
    echo "FAILED: $name"
  fi
  tail -n 2 "$log"
  echo
done
```

Tìm lỗi:

```bash
grep -HniE \
  "Traceback|CUDA out of memory|torch\.OutOfMemoryError|returned non-zero|Killed" \
  logs/h5_*.log
```

## 9. Kiểm tra đủ kết quả

```bash
find outputs -maxdepth 2 \
  -path 'outputs/h5_*_seed42/metrics_valid_best_province_accuracy.json' \
  | sort
```

Phải có 4 file, không tính `h5_smoke`.

## 10. Tổng hợp

```bash
python scripts/summarize_experiments.py \
  --outputs outputs \
  --pattern 'h5_*_seed42/metrics_valid_best_province_accuracy.json' \
  --destination outputs/h5_validation_seed42.csv \
  --aggregate-destination outputs/h5_validation_seed42_aggregate.csv
```

## 11. Paired comparison

So sánh spectral với baseline pitch/energy:

```bash
python scripts/compare_predictions.py \
  --baseline outputs/h5_acoustic_pitch_energy_seed42/predictions_valid_best_province_accuracy.jsonl \
  --candidate outputs/h5_acoustic_spectral_seed42/predictions_valid_best_province_accuracy.jsonl \
  --output outputs/h5_spectral_vs_pitch_energy_valid_seed42.json \
  --bootstrap-iterations 10000 \
  --seed 42
```

So sánh fusion đầy đủ với baseline:

```bash
python scripts/compare_predictions.py \
  --baseline outputs/h5_acoustic_pitch_energy_seed42/predictions_valid_best_province_accuracy.jsonl \
  --candidate outputs/h5_acoustic_pitch_energy_spectral_seed42/predictions_valid_best_province_accuracy.jsonl \
  --output outputs/h5_joint_vs_pitch_energy_valid_seed42.json \
  --bootstrap-iterations 10000 \
  --seed 42
```

## 12. File cần gửi

```text
outputs/h5_validation_seed42.csv
outputs/h5_spectral_vs_pitch_energy_valid_seed42.json
outputs/h5_joint_vs_pitch_energy_valid_seed42.json
```

Không chạy test. Sau khi phân tích seed 42, chỉ cấu hình được chọn trên validation
mới được tạo thêm seed 43 và 44.

## 13. Kết quả seed 42 và cấu hình được chọn

Fusion acoustic + pitch/energy + spectral tăng so với baseline:

```text
province accuracy          +0.0211
province balanced accuracy +0.0214
province macro-F1          +0.0203
```

Tuy nhiên CI 95% vẫn hơi chứa 0 và McNemar p bằng 0,0605. Vì vậy fusion được chọn
để xác nhận đa seed, chưa được xem là kết luận cuối.

Spectral-only giảm province accuracy 0,0421 với bằng chứng thống kê rõ ràng nên
không chạy thêm seed. Handcrafted cũng không được chạy thêm seed.

## 14. Chạy baseline và fusion ở seed 43–44

Các config:

```text
h5_acoustic_pitch_energy_seed43.yaml
h5_acoustic_pitch_energy_seed44.yaml
h5_acoustic_pitch_energy_spectral_seed43.yaml
h5_acoustic_pitch_energy_spectral_seed44.yaml
```

Kiểm tra GPU:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv
```

Ví dụ với GPU 1, 2, 6 và 7 trống:

```bash
CONFIGS=(
  h5_acoustic_pitch_energy_seed43
  h5_acoustic_pitch_energy_seed44
  h5_acoustic_pitch_energy_spectral_seed43
  h5_acoustic_pitch_energy_spectral_seed44
)

GPUS=(1 2 6 7)

for i in "${!CONFIGS[@]}"; do
  name="${CONFIGS[$i]}"
  gpu="${GPUS[$i]}"

  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -u scripts/run_experiment.py \
    --config "configs/experiments/${name}.yaml" \
    --split valid \
    > "logs/${name}.log" 2>&1 &

  echo $! > "logs/${name}.pid"
done
```

Thay GPU theo trạng thái server thực tế.

## 15. Tổng hợp đa seed

Baseline:

```bash
python scripts/summarize_experiments.py \
  --outputs outputs \
  --pattern 'h5_acoustic_pitch_energy_seed*/metrics_valid_best_province_accuracy.json' \
  --destination outputs/h5_pitch_energy_multiseed_valid.csv \
  --aggregate-destination outputs/h5_pitch_energy_multiseed_valid_aggregate.csv
```

Fusion:

```bash
python scripts/summarize_experiments.py \
  --outputs outputs \
  --pattern 'h5_acoustic_pitch_energy_spectral_seed*/metrics_valid_best_province_accuracy.json' \
  --destination outputs/h5_joint_multiseed_valid.csv \
  --aggregate-destination outputs/h5_joint_multiseed_valid_aggregate.csv
```

Paired comparison:

```bash
for seed in 42 43 44; do
  python scripts/compare_predictions.py \
    --baseline "outputs/h5_acoustic_pitch_energy_seed${seed}/predictions_valid_best_province_accuracy.jsonl" \
    --candidate "outputs/h5_acoustic_pitch_energy_spectral_seed${seed}/predictions_valid_best_province_accuracy.jsonl" \
    --output "outputs/h5_joint_vs_pitch_energy_valid_seed${seed}.json" \
    --bootstrap-iterations 10000 \
    --seed "$seed"
done
```

File cần gửi:

```text
h5_pitch_energy_multiseed_valid_aggregate.csv
h5_joint_multiseed_valid_aggregate.csv
h5_joint_vs_pitch_energy_valid_seed42.json
h5_joint_vs_pitch_energy_valid_seed43.json
h5_joint_vs_pitch_energy_valid_seed44.json
```

Chưa chạy test trước khi phân tích đủ ba seed validation.

## 16. Quyết định cuối H5

Kết quả ba seed:

```text
fusion - baseline:
region accuracy             +0.0014
province accuracy           +0.0042
province balanced accuracy  +0.0037
province macro-F1           +0.0010
```

Theo từng seed, province accuracy thay đổi:

```text
seed 42: +0.0211
seed 43: +0.0132
seed 44: -0.0216
```

Seed 44 có McNemar p bằng 0,0473; balanced accuracy và macro-F1 bootstrap CI 95%
hoàn toàn dưới 0. Fusion vì vậy không ổn định và không vượt qua cổng validation.

**Không chạy test cho H5.** Không chọn seed tốt nhất hoặc quay lại chỉnh fusion
dựa trên các kết quả này. H5 được lưu như negative result: spectral có tín hiệu
nhưng fusion thủ công hiện tại chưa tạo cải thiện lặp lại.
