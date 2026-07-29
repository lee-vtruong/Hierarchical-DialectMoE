# Hướng dẫn H4: điều chỉnh load-balancing để router chuyên môn hóa

## 1. Mục tiêu

H3 cho thấy cả ba kiểu đầu vào router đều có entropy gần `ln(4)` và mean expert
probability gần `[0,25; 0,25; 0,25; 0,25]`. H4 kiểm tra giả thuyết:

> Hệ số load-balancing hiện tại quá mạnh, khiến router gần uniform và hạn chế khả
> năng chuyên môn hóa expert.

H4 chỉ dùng **validation set** để chọn hệ số. Không dùng test set trong giai đoạn
sweep.

## 2. Sáu cấu hình

| Config | `load_balance_weight` | Output |
|---|---:|---|
| `h4_lb_0.yaml` | 0 | `outputs/h4_lb_0_seed42` |
| `h4_lb_0001.yaml` | 0,0001 | `outputs/h4_lb_0001_seed42` |
| `h4_lb_001.yaml` | 0,001 | `outputs/h4_lb_001_seed42` |
| `h4_lb_005.yaml` | 0,005 | `outputs/h4_lb_005_seed42` |
| `h4_lb_01.yaml` | 0,01 | `outputs/h4_lb_01_seed42` |
| `h4_lb_02.yaml` | 0,02 | `outputs/h4_lb_02_seed42` |

Mọi cấu hình đều giữ cố định:

- Seed 42.
- Acoustic + prosody cho classification.
- Acoustic + prosody làm đầu vào router.
- Hierarchical MoE với 4 expert.
- `top_k = 1`.
- `router_weight = 0`.

## 3. Cập nhật code trên server

```bash
cd /raid/hvtham/whale/Hierarchical-DialectMoE
git pull --rebase origin main
source .venv/bin/activate
python -m pytest -q
mkdir -p logs
```

Kiểm tra GPU trước khi chạy:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv
```

Chỉ chọn GPU có đủ VRAM và utilization thấp. Không dựa riêng vào utilization vì
một GPU utilization 0% vẫn có thể đang giữ gần hết bộ nhớ.

## 4. Chạy sweep

Ví dụ bên dưới giả sử GPU 1, 2 và 6 đang trống. Phải thay mảng `GPUS` theo trạng
thái thực tế tại thời điểm chạy.

```bash
CONFIGS=(
  h4_lb_0
  h4_lb_0001
  h4_lb_001
  h4_lb_005
  h4_lb_01
  h4_lb_02
)

GPUS=(1 2 6 1 2 6)

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

  echo $! > "logs/${name}_seed42.pid"
  echo "Started ${name} on physical GPU ${gpu}, PID=$!"
done
```

Nếu server đang dùng chung, nên chạy tối đa một job trên mỗi GPU. Nếu chỉ có hai
GPU trống, chạy hai hoặc bốn job trước, chờ xong rồi chạy phần còn lại.

## 5. Theo dõi

```bash
for name in h4_lb_0 h4_lb_0001 h4_lb_001 h4_lb_005 h4_lb_01 h4_lb_02; do
  pid_file="logs/${name}_seed42.pid"
  log_file="logs/${name}_seed42.log"
  pid=$(cat "$pid_file")

  if ps -p "$pid" > /dev/null; then
    echo "RUNNING: $name (PID=$pid)"
  elif grep -q "Experiment complete" "$log_file"; then
    echo "DONE:    $name"
  else
    echo "FAILED:  $name"
  fi

  tail -n 2 "$log_file"
  echo
done
```

Tìm lỗi:

```bash
grep -HniE \
  "Traceback|CUDA out of memory|torch\.OutOfMemoryError|returned non-zero|Killed" \
  logs/h4_lb_*.log
```

Lưu ý: `Done` trong thông báo Bash có nghĩa tiến trình kết thúc thành công.

## 6. Kiểm tra đủ kết quả validation

```bash
find outputs -maxdepth 2 \
  -path 'outputs/h4_lb_*_seed42/metrics_valid_best_province_accuracy.json' \
  | sort
```

Đếm:

```bash
find outputs -maxdepth 2 \
  -path 'outputs/h4_lb_*_seed42/metrics_valid_best_province_accuracy.json' \
  | wc -l
```

Kết quả phải là `6`.

## 7. Tổng hợp và chọn cấu hình

```bash
python scripts/summarize_h4.py \
  --outputs outputs \
  --destination outputs/h4_validation_summary.csv \
  --recommendation outputs/h4_validation_recommendation.json
```

Xem đề xuất:

```bash
cat outputs/h4_validation_recommendation.json
```

Xem bảng:

```bash
column -s, -t < outputs/h4_validation_summary.csv
```

Quy tắc chọn tự động:

1. Loại cấu hình collapse nếu một expert nhận hơn 90% top-1 assignment hoặc chỉ
   còn dưới hai expert hoạt động.
2. Chọn province macro-F1 validation cao nhất.
3. Nếu bằng nhau, dùng province balanced accuracy rồi province accuracy.

Các cột quan trọng:

- `normalized_soft_router_entropy`: entropy của xác suất router chia cho `ln(4)`.
- `effective_experts`: `exp(entropy)`.
- `normalized_top1_assignment_entropy`: độ cân bằng của expert được chọn top-1.
- `max_top1_assignment_fraction`: tỉ lệ của expert được chọn nhiều nhất.
- `region_expert_nmi`: mức liên hệ giữa expert và region.
- `province_expert_nmi`: mức liên hệ giữa expert và province.
- `collapsed`: router collapse.
- `near_uniform_soft_router`: xác suất mềm gần uniform.

Không chọn cấu hình chỉ vì entropy thấp hoặc NMI cao. Cấu hình phải đồng thời có
province macro-F1 validation tốt và không collapse.

## 8. Sau khi có kết quả sweep

Tải về:

```text
outputs/h4_validation_summary.csv
outputs/h4_validation_recommendation.json
```

Chưa chạy test. Gửi hai file này để tạo ba config seed 42/43/44 cho hệ số được
chọn. Sau khi kiểm tra tính lặp lại trên validation, mới khóa cấu hình và thực
hiện một lần đánh giá test cuối.

## 9. Kết quả sweep seed 42

Theo quy tắc đã định trước, cấu hình được chọn là:

```text
load_balance_weight = 0.001
```

Kết quả validation:

```text
province accuracy          = 0.4979
province balanced accuracy = 0.4928
province macro-F1          = 0.4867
```

Cấu hình không collapse, cả bốn expert đều được sử dụng. Tuy nhiên soft router
vẫn gần uniform (`normalized entropy = 0.999994`). Top-1 assignment có phân hóa:
expert lớn nhất nhận khoảng 64,32%, region–expert NMI bằng 0,2285 và
province–expert NMI bằng 0,1566.

`0.005` đạt province accuracy nhỉnh hơn `0.001` khoảng 0,0011 nhưng province
macro-F1 thấp hơn khoảng 0,0010. Vì tiêu chí chính đã khóa là province macro-F1,
không đổi sang `0.005` sau khi xem kết quả.

## 10. Xác nhận cấu hình được chọn trên seed 43 và 44

Hai config:

```text
configs/experiments/h4_lb_001_seed43.yaml
configs/experiments/h4_lb_001_seed44.yaml
```

Kiểm tra GPU:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv
```

Giả sử GPU 6 và 7 đang trống:

```bash
nohup env \
  CUDA_VISIBLE_DEVICES=6 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u scripts/run_experiment.py \
  --config configs/experiments/h4_lb_001_seed43.yaml \
  --split valid \
  > logs/h4_lb_001_seed43.log 2>&1 &

echo $! > logs/h4_lb_001_seed43.pid
```

```bash
nohup env \
  CUDA_VISIBLE_DEVICES=7 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u scripts/run_experiment.py \
  --config configs/experiments/h4_lb_001_seed44.yaml \
  --split valid \
  > logs/h4_lb_001_seed44.log 2>&1 &

echo $! > logs/h4_lb_001_seed44.pid
```

Theo dõi:

```bash
for name in h4_lb_001_seed43 h4_lb_001_seed44; do
  pid=$(cat "logs/${name}.pid")
  if ps -p "$pid" > /dev/null; then
    echo "RUNNING: $name"
  elif grep -q "Experiment complete" "logs/${name}.log"; then
    echo "DONE: $name"
  else
    echo "FAILED: $name"
  fi
  tail -n 2 "logs/${name}.log"
  echo
done
```

Sau khi hoàn thành, tổng hợp riêng ba seed của cấu hình `0.001`:

```bash
python scripts/summarize_experiments.py \
  --outputs outputs \
  --pattern 'h4_lb_001_seed*/metrics_valid_best_province_accuracy.json' \
  --destination outputs/h4_lb_001_multiseed_validation.csv \
  --aggregate-destination outputs/h4_lb_001_multiseed_validation_aggregate.csv
```

Đồng thời chạy lại bộ tổng hợp chuyên môn hóa để lấy các chỉ số routing của cả ba
seed:

```bash
python scripts/summarize_h4.py \
  --outputs outputs \
  --pattern 'h4_lb_001_seed*/metrics_valid_best_province_accuracy.json' \
  --destination outputs/h4_lb_001_routing_multiseed.csv \
  --recommendation outputs/h4_lb_001_routing_multiseed_note.json
```

Tải về:

```text
outputs/h4_lb_001_multiseed_validation.csv
outputs/h4_lb_001_multiseed_validation_aggregate.csv
outputs/h4_lb_001_routing_multiseed.csv
```

Không chạy test trước khi phân tích xong ba seed validation.
