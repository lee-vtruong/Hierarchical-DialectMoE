# Hướng dẫn H14: Large-VI + prosody + hierarchical MoE-2

## 1. Mục tiêu và protocol khóa trước

H14 kiểm tra đúng một cấu hình MoE trên backbone tốt nhất H11:

```text
Backbone: nguyenvulebinh/wav2vec2-large-vi
Input: acoustic + prosody
Fusion: gated
Hierarchical router: bật, dùng acoustic + prosody và region context
Experts: 2
Top-k: 1
Router entropy weight: 0
Load-balance weight: 0.001
Checkpoint selection: province validation accuracy
Seeds: 42, 43, 44
```

Không sweep expert/top-k/load balance bằng test. Baseline được khóa là H11
Large-VI acoustic+prosody không MoE cùng seed. Mỗi H14 test prediction chỉ được
sinh một lần sau khi checkpoint đã được chọn bằng validation.

## 2. Kiểm tra code và GPU

```bash
cd /home/stackops/whale/Hierarchical-DialectMoE
conda activate /home/stackops/whale/envs/hierarchical-dialect-moe

python -m pytest -q tests/test_analyze_h14.py

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv
```

## 3. Chạy ba seed trên hai GPU

GPU 0 chạy seed 42 rồi seed 44. GPU 1 chạy seed 43. Mỗi GPU chỉ giữ một model
tại một thời điểm.

### GPU 0: seed 42 và 44

```bash
mkdir -p logs

nohup env CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  bash -lc '
set -euo pipefail
cd /home/stackops/whale/Hierarchical-DialectMoE
PYTHON=/home/stackops/whale/envs/hierarchical-dialect-moe/bin/python

for seed in 42 44; do
  config="configs/experiments/h14_large_vi_prosody_moe2.yaml"
  [ "$seed" != 42 ] && config="configs/experiments/h14_large_vi_prosody_moe2_seed${seed}.yaml"
  echo "===== H14 SEED ${seed} START ====="
  date
  "$PYTHON" -u scripts/run_experiment.py --config "$config"
  echo "===== H14 SEED ${seed} FINISHED ====="
  date
done
' > logs/h14_seed42_44.log 2>&1 &

echo $! > logs/h14_seed42_44.pid
```

### GPU 1: seed 43

```bash
nohup env CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  bash -lc '
set -euo pipefail
cd /home/stackops/whale/Hierarchical-DialectMoE
PYTHON=/home/stackops/whale/envs/hierarchical-dialect-moe/bin/python

echo "===== H14 SEED 43 START ====="
date
"$PYTHON" -u scripts/run_experiment.py \
  --config configs/experiments/h14_large_vi_prosody_moe2_seed43.yaml
echo "===== H14 SEED 43 FINISHED ====="
date
' > logs/h14_seed43.log 2>&1 &

echo $! > logs/h14_seed43.pid
```

Theo dõi:

```bash
for name in h14_seed42_44 h14_seed43; do
  pid=$(cat "logs/${name}.pid")
  ps -p "$pid" -o pid,etime,%cpu,%mem,stat,cmd || true
  tail -n 6 "logs/${name}.log"
  echo
done
```

Nếu gặp OOM, không tự đổi batch size cho riêng một seed. Dừng và báo lỗi để cập
nhật đồng nhất cả ba config/protocol.

## 4. Kiểm tra artifact

```bash
for seed in 42 43 44; do
  dir="outputs/h14_large_vi_prosody_moe2_seed${seed}"
  for file in \
    best_province_accuracy.pt \
    metrics_test_best_province_accuracy.json \
    predictions_test_best_province_accuracy.jsonl \
    region_to_expert_test_best_province_accuracy.csv \
    province_to_expert_test_best_province_accuracy.csv
  do
    test -s "$dir/$file" && echo "OK: $dir/$file" || echo "THIẾU: $dir/$file"
  done
done
```

## 5. Chạy kiểm định paired H14

Phân tích chỉ dùng CPU:

```bash
mkdir -p results_archive/h14

nohup python -u scripts/analyze_h14.py \
  --outputs outputs \
  --destination results_archive/h14 \
  --seeds 42 43 44 \
  --bootstrap-iterations 10000 \
  --bootstrap-seed 14026 \
  > logs/h14_analysis.log 2>&1 &

echo $! > logs/h14_analysis.pid
```

Kết quả:

```text
results_archive/h14/h14_per_seed.csv
results_archive/h14/h14_aggregate.csv
results_archive/h14/h14_routing.csv
results_archive/h14/h14_details.json
results_archive/h14/h14_summary.json
```

## 6. In kết quả chính

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("results_archive/h14/h14_aggregate.csv")
cols = [
    "task",
    "baseline_accuracy_mean",
    "candidate_accuracy_mean",
    "difference_accuracy_mean",
    "difference_macro_f1_mean",
    "bootstrap_ci_excludes_zero_accuracy_runs",
    "mcnemar_holm_significant_runs",
]
print(df[cols].to_string(index=False))

print("\n===== ROUTING =====")
print(pd.read_csv("results_archive/h14/h14_routing.csv").to_string(index=False))
PY
```

MoE chỉ được xem là cải thiện xác nhận nếu hiệu ứng cấp tỉnh dương ổn định qua
ba seed, bootstrap CI loại 0 và McNemar vẫn có ý nghĩa sau Holm correction.
