# Hướng dẫn H18: phân rã mềm vùng–tỉnh

## 1. H18 kiểm tra giả thuyết gì?

H11 dự đoán vùng và tỉnh bằng hai head phẳng gần như độc lập. Trong khi đó,
mỗi tỉnh của ViMD thuộc đúng một trong ba vùng. H18 đưa cấu trúc đã biết này
vào classifier:

```text
P(tỉnh p | x) = P(vùng r(p) | x) × P(tỉnh p | vùng r(p), x)
```

Đây là phân rã **mềm**:

- không lấy `argmax` vùng để hard-mask tỉnh;
- cả ba xác suất vùng vẫn tham gia posterior 63 tỉnh;
- tổng xác suất các tỉnh trong một vùng đúng bằng xác suất của vùng đó;
- head tỉnh vẫn có 63 output và H18 không thêm trainable parameter;
- loss giữ nguyên trọng số H11: `0.4 × CE(vùng) + 1.0 × CE(tỉnh|vùng)`.

Tất cả yếu tố còn lại được giữ nguyên từ H11 Large-VI + static prosody: split
speaker-disjoint, crop 20 giây đầu, backbone, mean pooling, fusion, optimizer,
effective batch size, lịch học và checkpoint selection.

H18 là một ablation mới. Tuyệt đối không dùng test để chỉnh loss, threshold hay
mapping.

## 2. Files mới và files đã sửa

Các phần chính:

- `dialect_moe/labels.py`: dựng và kiểm tra ánh xạ tỉnh → vùng;
- `dialect_moe/model.py`: posterior phân rã mềm;
- `dialect_moe/losses.py`: conditional province loss, không đếm region loss hai lần;
- `dialect_moe/data.py`: tạo mapping từ metadata gốc;
- `scripts/evaluate.py`: đo consistency và cross-region error;
- `scripts/summarize_h18.py`: tổng hợp riêng H18;
- `scripts/compare_h18.py`: so khớp từng seed với H11 và áp dụng gate định trước;
- `configs/experiments/h18_soft_hierarchy_*.yaml`: smoke và ba seed;
- `tests/test_h18_hierarchy.py`, `tests/test_compare_h18.py`,
  `tests/test_summarize_h18.py`: unit tests.

## 3. Đẩy code từ máy Windows

Trong PowerShell tại repository local:

```powershell
cd C:\Users\ASUS\Downloads\HierarchicalDialectMoE

git add `
  dialect_moe/data.py `
  dialect_moe/labels.py `
  dialect_moe/losses.py `
  dialect_moe/model.py `
  scripts/train.py `
  scripts/evaluate.py `
  scripts/evaluate_multicrop.py `
  scripts/benchmark_h13.py `
  scripts/check_backbone.py `
  scripts/summarize_h18.py `
  scripts/compare_h18.py `
  configs/experiments/h18_soft_hierarchy_seed42.yaml `
  configs/experiments/h18_soft_hierarchy_seed43.yaml `
  configs/experiments/h18_soft_hierarchy_seed44.yaml `
  configs/experiments/h18_soft_hierarchy_smoke.yaml `
  tests/test_h18_hierarchy.py `
  tests/test_compare_h18.py `
  tests/test_summarize_h18.py `
  tests/test_config.py `
  HUONG_DAN_H18.md `
  BAO_CAO_THUC_NGHIEM.md

git commit -m "Add H18 soft hierarchical province factorization"
git pull --rebase origin main
git push origin main
```

Lệnh trên dùng danh sách file tường minh nên không đưa nhầm `outputs/`, model
checkpoint hoặc dữ liệu lớn lên GitHub.

## 4. Cập nhật server và kiểm tra code

```bash
cd /home/stackops/whale/Hierarchical-DialectMoE
git pull --ff-only origin main

source /home/stackops/miniconda3/etc/profile.d/conda.sh
conda activate /home/stackops/whale/envs/hierarchical-dialect-moe

PYTHON_BIN=/home/stackops/whale/envs/hierarchical-dialect-moe/bin/python
"$PYTHON_BIN" --version
```

Chạy test:

```bash
"$PYTHON_BIN" -m pytest -q \
  tests/test_h18_hierarchy.py \
  tests/test_compare_h18.py \
  tests/test_summarize_h18.py \
  tests/test_config.py \
  tests/test_components.py \
  tests/test_train_checkpoint.py
```

Kết quả mong đợi: tất cả test `passed`. Warning của sklearn hoặc Transformer
không phải lỗi nếu không có `FAILED`/`Traceback`.

## 5. Smoke test

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 \
"$PYTHON_BIN" -u scripts/run_experiment.py \
  --config configs/experiments/h18_soft_hierarchy_smoke.yaml \
  --max-samples 16 \
  --split valid \
  2>&1 | tee logs/h18_smoke.log
```

Kiểm tra:

```bash
grep -nEi "Traceback|CUDA error|out of memory|nan" logs/h18_smoke.log \
  || echo "H18 SMOKE KHÔNG CÓ LỖI"

test -f \
  outputs/h18_soft_hierarchy_smoke/metrics_valid_best_province_accuracy.json \
  && echo "H18 SMOKE OK" \
  || echo "H18 SMOKE CHƯA XONG"
```

Smoke chỉ xác nhận pipeline, không phải kết quả để báo cáo.

## 6. Screening seed 42 — chỉ train và validation

H18 trước hết chỉ được chạy seed 42. Không gọi `run_experiment.py` mặc định vì
mặc định của script là test.

```bash
cd /home/stackops/whale/Hierarchical-DialectMoE
mkdir -p logs

nohup setsid env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" -u scripts/train.py \
    --config configs/experiments/h18_soft_hierarchy_seed42.yaml \
  </dev/null > logs/h18_soft_hierarchy_seed42.log 2>&1 &

pid=$!
echo "$pid" | tee logs/h18_soft_hierarchy_seed42.pid
disown "$pid" 2>/dev/null || true
```

Theo dõi:

```bash
pid=$(cat logs/h18_soft_hierarchy_seed42.pid)
ps -p "$pid" -o pid,etime,%cpu,%mem,stat,cmd
tail -n 20 logs/h18_soft_hierarchy_seed42.log
nvidia-smi
```

Nếu process đã dừng, phân biệt hoàn thành và lỗi:

```bash
grep -nEi "Traceback|CUDA error|out of memory|nan" \
  logs/h18_soft_hierarchy_seed42.log \
  || echo "KHÔNG THẤY LỖI TRONG LOG"

test -f outputs/h18_soft_hierarchy_seed42/best_province_accuracy.pt \
  && echo "TRAIN SEED 42 OK" \
  || echo "TRAIN SEED 42 THIẾU CHECKPOINT"
```

Đánh giá **validation**:

```bash
CUDA_VISIBLE_DEVICES=0 \
"$PYTHON_BIN" -u scripts/run_experiment.py \
  --config configs/experiments/h18_soft_hierarchy_seed42.yaml \
  --skip-train \
  --split valid \
  2>&1 | tee logs/h18_soft_hierarchy_seed42_valid.log
```

## 7. So seed 42 với H11 và áp dụng gate

```bash
"$PYTHON_BIN" scripts/summarize_h18.py \
  --outputs outputs \
  --split valid \
  --destination results_archive/h18

"$PYTHON_BIN" scripts/compare_h18.py \
  --outputs outputs \
  --split valid \
  --seeds 42 \
  --destination results_archive/h18

cat results_archive/h18/h18_valid_per_seed.csv
cat results_archive/h18/h18_vs_h11_valid_per_seed.csv
cat results_archive/h18/h18_vs_h11_valid_decision.json
```

Gate screening đã được code trước khi thấy kết quả:

- province accuracy H18 phải cao hơn H11 seed 42;
- province macro-F1 được giảm tối đa `0.003`;
- region accuracy được giảm tối đa `0.005`.

Nếu JSON ghi `"passed": false`, đóng H18 như negative validation result và
không chạy seed 43/44, không mở test. Hãy gửi ba file trên cho Codex để cập nhật
báo cáo.

## 8. Chỉ khi seed 42 pass: chạy seed 43 và 44 song song

GPU 0, seed 43:

```bash
nohup setsid env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" -u scripts/train.py \
    --config configs/experiments/h18_soft_hierarchy_seed43.yaml \
  </dev/null > logs/h18_soft_hierarchy_seed43.log 2>&1 &
echo $! | tee logs/h18_soft_hierarchy_seed43.pid
```

GPU 1, seed 44:

```bash
nohup setsid env \
  CUDA_VISIBLE_DEVICES=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" -u scripts/train.py \
    --config configs/experiments/h18_soft_hierarchy_seed44.yaml \
  </dev/null > logs/h18_soft_hierarchy_seed44.log 2>&1 &
echo $! | tee logs/h18_soft_hierarchy_seed44.pid
```

Theo dõi cả hai:

```bash
watch -n 15 '
for seed in 43 44; do
  name="h18_soft_hierarchy_seed${seed}"
  pid=$(cat "logs/${name}.pid")
  if ps -p "$pid" >/dev/null; then
    echo "$name: ĐANG CHẠY PID=$pid"
  else
    echo "$name: ĐÃ DỪNG"
  fi
  tail -n 3 "logs/${name}.log"
  echo
done
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
'
```

Sau khi cả hai có checkpoint, đánh giá validation song song:

```bash
nohup setsid env CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" -u scripts/run_experiment.py \
    --config configs/experiments/h18_soft_hierarchy_seed43.yaml \
    --skip-train --split valid \
  </dev/null > logs/h18_soft_hierarchy_seed43_valid.log 2>&1 &
echo $! | tee logs/h18_soft_hierarchy_seed43_valid.pid

nohup setsid env CUDA_VISIBLE_DEVICES=1 \
  "$PYTHON_BIN" -u scripts/run_experiment.py \
    --config configs/experiments/h18_soft_hierarchy_seed44.yaml \
    --skip-train --split valid \
  </dev/null > logs/h18_soft_hierarchy_seed44_valid.log 2>&1 &
echo $! | tee logs/h18_soft_hierarchy_seed44_valid.pid
```

Kiểm tra đủ 3/3:

```bash
for seed in 42 43 44; do
  file="outputs/h18_soft_hierarchy_seed${seed}/metrics_valid_best_province_accuracy.json"
  test -f "$file" && echo "OK: $file" || echo "THIẾU: $file"
done
```

## 9. Gate multi-seed

```bash
"$PYTHON_BIN" scripts/summarize_h18.py \
  --outputs outputs \
  --split valid \
  --destination results_archive/h18

"$PYTHON_BIN" scripts/compare_h18.py \
  --outputs outputs \
  --split valid \
  --seeds 42 43 44 \
  --destination results_archive/h18

cat results_archive/h18/h18_valid_aggregate.csv
cat results_archive/h18/h18_vs_h11_valid_per_seed.csv
cat results_archive/h18/h18_vs_h11_valid_decision.json
```

Gate cuối để được mở test:

- mean province accuracy cao hơn H11;
- mean province macro-F1 không thấp hơn H11;
- H18 thắng province accuracy ít nhất 2/3 seed;
- mean region accuracy giảm không quá `0.005`.

Nếu không pass, H18 là negative result và test vẫn đóng. Không đổi threshold,
loss hoặc mapping sau khi nhìn validation.

## 10. Chỉ khi multi-seed pass: mở test một lần

```bash
for item in "42 0" "43 1"; do
  set -- $item
  seed=$1
  gpu=$2
  name="h18_soft_hierarchy_seed${seed}"
  nohup setsid env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" -u scripts/run_experiment.py \
      --config "configs/experiments/${name}.yaml" \
      --skip-train --split test \
    </dev/null > "logs/${name}_test.log" 2>&1 &
  echo $! | tee "logs/${name}_test.pid"
done
```

Sau khi 42/43 xong, chạy seed 44 trên GPU trống:

```bash
nohup setsid env CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON_BIN" -u scripts/run_experiment.py \
    --config configs/experiments/h18_soft_hierarchy_seed44.yaml \
    --skip-train --split test \
  </dev/null > logs/h18_soft_hierarchy_seed44_test.log 2>&1 &
echo $! | tee logs/h18_soft_hierarchy_seed44_test.pid
```

Tổng hợp test và paired analysis bằng công cụ đã có:

```bash
"$PYTHON_BIN" scripts/summarize_h18.py \
  --outputs outputs --split test --destination results_archive/h18

"$PYTHON_BIN" scripts/compare_h18.py \
  --outputs outputs --split test --seeds 42 43 44 \
  --destination results_archive/h18

for seed in 42 43 44; do
  "$PYTHON_BIN" scripts/compare_predictions.py \
    --baseline "outputs/h11_large_vi_prosody_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
    --candidate "outputs/h18_soft_hierarchy_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
    --output "results_archive/h18/h18_vs_h11_test_seed${seed}.json"
done
```

## 11. Artifact cần tải về

Ít nhất giữ:

```text
results_archive/h18/h18_valid_per_seed.csv
results_archive/h18/h18_valid_aggregate.csv
results_archive/h18/h18_vs_h11_valid_per_seed.csv
results_archive/h18/h18_vs_h11_valid_decision.json
```

Nếu H18 được mở test, giữ thêm toàn bộ file `*test*` trong
`results_archive/h18/` và ba metrics/predictions test của H18. Không cần đưa
checkpoint nhiều GB lên GitHub; nên lưu ở ổ archive riêng.

## 12. Ý nghĩa hai diagnostic hierarchy

- `prediction_region_consistency`: tỷ lệ vùng suy ra từ tỉnh dự đoán trùng với
  region head prediction. Posterior mềm không bắt buộc mode của hai phân phối
  luôn giống nhau, nên chỉ số này có thể nhỏ hơn 1.
- `province_cross_region_error_rate`: tỷ lệ tỉnh dự đoán thuộc sai vùng thật.
  Chỉ số thấp hơn H11 là bằng chứng H18 giảm lỗi liên-vùng; nó không thay thế
  province accuracy/macro-F1 làm endpoint chính.
