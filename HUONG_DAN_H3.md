# Hướng dẫn thí nghiệm H3 - Router input

## 1. Giả thuyết

H3:

> Routing sử dụng prosody hiệu quả hơn routing chỉ dùng acoustic embedding.

Ba biến thể:

1. `acoustic`: router chỉ nhận acoustic embedding.
2. `prosody`: router chỉ nhận prosody embedding.
3. `acoustic_prosody`: router nhận kết hợp hai nguồn.

Trong cả ba biến thể:

- Acoustic encoder giống nhau.
- Prosody encoder giống nhau.
- Gated fusion và classification heads giống nhau.
- Region-conditioned hierarchy giống nhau.
- 4 experts, top-k=1.
- `load_balance_weight=0.02`.
- `router_weight=0`.
- Chỉ tín hiệu đi vào expert router thay đổi.

Mỗi model vẫn dùng acoustic + prosody cho classification representation. Đây
là controlled ablation của router input, không phải ablation feature fusion.

## 2. Config

```text
h3_router_acoustic.yaml
h3_router_acoustic_seed43.yaml
h3_router_acoustic_seed44.yaml

h3_router_prosody.yaml
h3_router_prosody_seed43.yaml
h3_router_prosody_seed44.yaml

h3_router_acoustic_prosody.yaml
h3_router_acoustic_prosody_seed43.yaml
h3_router_acoustic_prosody_seed44.yaml
```

## 3. Cập nhật server

```bash
cd /raid/hvtham/whale/Hierarchical-DialectMoE
git pull --rebase origin main
source .venv/bin/activate
python -m pytest -q
```

## 4. Kiểm tra GPU

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv
```

Mỗi job trước đây dùng khoảng 4-6 GB. Chỉ chọn GPU có đủ headroom. Có thể chạy
hai job trên A100 80 GB nếu GPU còn nhiều VRAM và compute chưa đạt 100%.

## 5. Launch chín run

Tạo mảng GPU sau khi kiểm tra `nvidia-smi`. Ví dụ:

```bash
GPUS=(2 0 1 6 3 7 2 0 1)
```

Nếu trạng thái GPU khác, sửa chín số trên.

Khai báo config và tên:

```bash
CONFIGS=(
  h3_router_acoustic
  h3_router_acoustic_seed43
  h3_router_acoustic_seed44
  h3_router_prosody
  h3_router_prosody_seed43
  h3_router_prosody_seed44
  h3_router_acoustic_prosody
  h3_router_acoustic_prosody_seed43
  h3_router_acoustic_prosody_seed44
)
```

Launch:

```bash
mkdir -p logs

for i in "${!CONFIGS[@]}"; do
  name="${CONFIGS[$i]}"
  gpu="${GPUS[$i]}"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" \
    .venv/bin/python -u scripts/run_experiment.py \
    --config "configs/experiments/${name}.yaml" \
    > "logs/${name}.log" 2>&1 &
  echo $! > "logs/${name}.pid"
  echo "Started $name on GPU $gpu with PID $(cat "logs/${name}.pid")"
done
```

Không chạy lại vòng lặp nếu các job đã tồn tại.

## 6. Theo dõi

```bash
for name in "${CONFIGS[@]}"; do
  pid="$(cat "logs/${name}.pid")"
  ps -p "$pid" -o pid,etime,cmd
done
```

Kiểm tra lỗi:

```bash
grep -iE "traceback|error|cuda out of memory|nan" \
  logs/h3_router_*.log
```

Theo dõi một job:

```bash
tail -f logs/h3_router_acoustic.log
```

## 7. Tổng hợp metrics

```bash
python scripts/summarize_experiments.py \
  --outputs outputs \
  --destination outputs/experiment_summary.csv \
  --aggregate-destination outputs/experiment_aggregate.csv
```

Xem riêng H3:

```bash
grep "^h3_router_" outputs/experiment_aggregate.csv
```

## 8. Expert assignment artifacts

Mỗi H3 output có:

```text
predictions_test_best_province_accuracy.jsonl
region_to_expert_test_best_province_accuracy.csv
province_to_expert_test_best_province_accuracy.csv
metrics_test_best_province_accuracy.json
```

Trong metrics:

```text
routing.top1_assignment_counts
routing.top1_assignment_fractions
routing.region_to_expert_counts
routing.province_to_expert_counts
```

Mean probability/entropy đo độ mềm của router. Top-1 counts đo expert thực sự
được chọn.

## 9. Paired comparison H3

Tạo thư mục:

```bash
mkdir -p outputs/statistics_h3
```

So sánh prosody router với acoustic router:

```bash
for seed in 42 43 44; do
  acoustic_name="h3_router_acoustic_seed${seed}"
  prosody_name="h3_router_prosody_seed${seed}"
  if [ "$seed" = "42" ]; then
    acoustic_name="h3_router_acoustic_seed42"
    prosody_name="h3_router_prosody_seed42"
  fi

  python scripts/compare_predictions.py \
    --baseline "outputs/${acoustic_name}/predictions_test_best_province_accuracy.jsonl" \
    --candidate "outputs/${prosody_name}/predictions_test_best_province_accuracy.jsonl" \
    --bootstrap-iterations 10000 \
    --seed "3030${seed}" \
    --output "outputs/statistics_h3/prosody_vs_acoustic_router_seed${seed}.json"
done
```

Lưu ý: output directory của seed 42 đã được đặt rõ là:

```text
outputs/h3_router_acoustic_seed42
outputs/h3_router_prosody_seed42
outputs/h3_router_acoustic_prosody_seed42
```

So sánh joint router với acoustic router:

```bash
for seed in 42 43 44; do
  python scripts/compare_predictions.py \
    --baseline "outputs/h3_router_acoustic_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
    --candidate "outputs/h3_router_acoustic_prosody_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
    --bootstrap-iterations 10000 \
    --seed "4040${seed}" \
    --output "outputs/statistics_h3/joint_vs_acoustic_router_seed${seed}.json"
done
```

## 10. Cách kết luận H3

H3 được hỗ trợ nếu prosody router:

- Có mean province macro-F1/accuracy cao hơn acoustic router qua ba seed.
- Paired bootstrap CI chủ yếu không chứa 0.
- McNemar p-value cho thấy cải thiện paired.
- Expert assignment có cấu trúc theo vùng/tỉnh, không collapse hoặc uniform
  ngẫu nhiên.

Nếu prosody router chỉ thắng một seed hoặc variance lớn, H3 chưa được hỗ trợ.

Joint router là biến thể bổ sung. Joint tốt nhất không đồng nghĩa H3 đúng;
prosody-only vẫn phải được so sánh trực tiếp với acoustic-only.

## 11. File cần gửi

```text
outputs/experiment_summary.csv
outputs/experiment_aggregate.csv
outputs/statistics_h3/prosody_vs_acoustic_router_seed42.json
outputs/statistics_h3/prosody_vs_acoustic_router_seed43.json
outputs/statistics_h3/prosody_vs_acoustic_router_seed44.json
outputs/statistics_h3/joint_vs_acoustic_router_seed42.json
outputs/statistics_h3/joint_vs_acoustic_router_seed43.json
outputs/statistics_h3/joint_vs_acoustic_router_seed44.json
```

Gửi thêm ba `metrics_test_best_province_accuracy.json` seed 42 nếu cần phân
tích chi tiết expert matrix.

