# Hướng dẫn H17: LayerMix và Attentive Statistics Pooling

## 1. Mục tiêu và protocol

H17 thay backend mean pooling của H11 nhưng giữ nguyên Wav2Vec2-Large-VI,
static prosody, gated fusion, loss, repaired speaker-disjoint split và effective
batch size.

- H17-A: final hidden layer + attentive statistics pooling (ASP).
- H17-B: learned mixture của 8 hidden layer cuối + ASP.
- H11 Large-VI + static prosody: đối chứng đã khóa.

Thứ tự bắt buộc: smoke -> H17-A/B seed 42 -> validation screening -> khóa một
variant -> seed 43/44 -> validation multi-seed -> test đúng một lần. Không dùng
test để chọn A/B, số layer hay attention dimension.

Checkpoint vẫn được train script chọn theo province validation accuracy để
khớp protocol H11. Việc chọn giữa H17-A và H17-B dùng province macro-F1 trên
validation của checkpoint đó.

## 2. Cập nhật code và môi trường

```bash
cd /home/stackops/whale/Hierarchical-DialectMoE
git pull --ff-only origin main

source /home/stackops/miniconda3/etc/profile.d/conda.sh
conda activate /home/stackops/whale/envs/hierarchical-dialect-moe

which python
python --version
```

Chạy test:

```bash
python -m pytest -q \
  tests/test_h17_pooling.py \
  tests/test_config.py \
  tests/test_components.py \
  tests/test_train_checkpoint.py \
  tests/test_summarize_h17.py
```

## 3. Smoke test

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python -u scripts/run_experiment.py \
  --config configs/experiments/h17b_layermix_asp_smoke.yaml \
  --max-samples 16 \
  2>&1 | tee logs/h17b_smoke.log
```

Kiểm tra:

```bash
grep -nEi "Traceback|Error|out of memory|nan" logs/h17b_smoke.log \
  || echo "H17 SMOKE KHÔNG CÓ LỖI"

test -f \
  outputs/h17b_layermix_asp_smoke/metrics_test_best_province_accuracy.json \
  && echo "H17 SMOKE OK" \
  || echo "H17 SMOKE CHƯA HOÀN THÀNH"
```

Smoke test chỉ kiểm tra code path với 16 mẫu, không phải kết quả khoa học.

## 4. Screening seed 42 trên hai GPU

Thiết lập:

```bash
cd /home/stackops/whale/Hierarchical-DialectMoE
source /home/stackops/miniconda3/etc/profile.d/conda.sh
conda activate /home/stackops/whale/envs/hierarchical-dialect-moe
mkdir -p logs
```

GPU 0, H17-A:

```bash
nohup setsid env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -u scripts/train.py \
    --config configs/experiments/h17a_asp_seed42.yaml \
  </dev/null > logs/h17a_asp_seed42.log 2>&1 &

pid=$!
echo "$pid" | tee logs/h17a_asp_seed42.pid
disown "$pid" 2>/dev/null || true
```

GPU 1, H17-B:

```bash
nohup setsid env \
  CUDA_VISIBLE_DEVICES=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -u scripts/train.py \
    --config configs/experiments/h17b_layermix_asp_seed42.yaml \
  </dev/null > logs/h17b_layermix_asp_seed42.log 2>&1 &

pid=$!
echo "$pid" | tee logs/h17b_layermix_asp_seed42.pid
disown "$pid" 2>/dev/null || true
```

Theo dõi:

```bash
watch -n 15 '
for name in h17a_asp_seed42 h17b_layermix_asp_seed42; do
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

Nếu H17-B OOM nhưng H17-A chạy được, không giảm crop hay đổi backbone. Giữ
effective batch bằng cách tạo config kế thừa với `batch_size: 1` và
`gradient_accumulation_steps: 32`, rồi chạy lại H17-B từ đầu.

Xác nhận checkpoint:

```bash
for name in h17a_asp_seed42 h17b_layermix_asp_seed42; do
  test -f "outputs/${name}/best_province_accuracy.pt" \
    && echo "OK: $name" \
    || echo "THIẾU: $name"
done
```

## 5. Đánh giá validation, tuyệt đối chưa đánh giá test

Chạy song song:

```bash
nohup setsid env CUDA_VISIBLE_DEVICES=0 \
  python -u scripts/run_experiment.py \
    --config configs/experiments/h17a_asp_seed42.yaml \
    --skip-train \
    --split valid \
  </dev/null > logs/h17a_asp_seed42_valid.log 2>&1 &
echo $! | tee logs/h17a_asp_seed42_valid.pid

nohup setsid env CUDA_VISIBLE_DEVICES=1 \
  python -u scripts/run_experiment.py \
    --config configs/experiments/h17b_layermix_asp_seed42.yaml \
    --skip-train \
    --split valid \
  </dev/null > logs/h17b_layermix_asp_seed42_valid.log 2>&1 &
echo $! | tee logs/h17b_layermix_asp_seed42_valid.pid
```

Kiểm tra:

```bash
for name in h17a_asp_seed42 h17b_layermix_asp_seed42; do
  file="outputs/${name}/metrics_valid_best_province_accuracy.json"
  test -f "$file" && echo "OK: $file" || echo "THIẾU: $file"
done
```

Tổng hợp:

```bash
python scripts/summarize_h17.py \
  --outputs outputs \
  --split valid \
  --destination results_archive/h17

cat results_archive/h17/h17_valid_per_seed.csv
cat results_archive/h17/h17_valid_aggregate.csv
```

In thêm H11 seed 42 nếu artifact validation còn trên server:

```bash
python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("outputs/h11_large_vi_prosody_seed42/metrics_valid_best_province_accuracy.json"),
    Path("outputs/h17a_asp_seed42/metrics_valid_best_province_accuracy.json"),
    Path("outputs/h17b_layermix_asp_seed42/metrics_valid_best_province_accuracy.json"),
]
for path in paths:
    data = json.loads(path.read_text())
    print(path.parent.name)
    print("  province accuracy:", data["province"]["accuracy"])
    print(
        "  province macro-F1:",
        data["province"]["classification_report"]["macro avg"]["f1-score"],
    )
    print("  representation:", data.get("representation"))
PY
```

Nếu H11 validation metrics thiếu, chỉ evaluate checkpoint cũ trên validation;
không train lại và không mở test:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
  --config configs/experiments/h11_large_vi_prosody.yaml \
  --skip-train \
  --split valid
```

## 6. Gate chọn kiến trúc

1. So H17-A với H11 để đo ASP.
2. So H17-B với H17-A để đo LayerMix.
3. Chọn đúng một variant có province macro-F1 validation cao nhất.
4. Nếu cả A và B đều không vượt H11 seed 42, đóng H17 như negative ablation;
   không chạy test.
5. Không thay `last_n_layers`, attention dimension hay loss sau screening.

Gửi `h17_valid_per_seed.csv` và ba dòng H11/A/B cho Codex trước khi chạy seed
43/44. Các lệnh dưới đây chỉ dùng sau khi variant đã được khóa.

## 7. Xác nhận multi-seed

Ví dụ nếu H17-B thắng, chạy seed 43/44 trên hai GPU:

```bash
for item in "43 0" "44 1"; do
  set -- $item
  seed=$1
  gpu=$2
  name="h17b_layermix_asp_seed${seed}"

  nohup setsid env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    python -u scripts/train.py \
      --config "configs/experiments/${name}.yaml" \
    </dev/null > "logs/${name}.log" 2>&1 &

  pid=$!
  echo "$pid" | tee "logs/${name}.pid"
  disown "$pid" 2>/dev/null || true
done
```

Nếu H17-A thắng, thay prefix `h17b_layermix_asp` bằng `h17a_asp`.

Sau khi train xong, evaluate validation từng seed bằng `run_experiment.py
--skip-train --split valid`, rồi chạy lại `summarize_h17.py`. Chỉ tiếp tục nếu
hiệu ứng so với H11 dương ổn định, không chỉ nhờ một seed.

## 8. Test cuối sau khi khóa variant

Chỉ variant thắng mới được đánh giá test. Ví dụ H17-B:

```bash
for seed in 42 43 44; do
  CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
    --config "configs/experiments/h17b_layermix_asp_seed${seed}.yaml" \
    --skip-train \
    --split test
done
```

Tổng hợp:

```bash
python scripts/summarize_h17.py \
  --outputs outputs \
  --split test \
  --destination results_archive/h17
```

Paired comparison với H11 cùng seed:

```bash
for seed in 42 43 44; do
  python scripts/compare_predictions.py \
    --baseline "outputs/h11_large_vi_prosody_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
    --candidate "outputs/h17b_layermix_asp_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
    --output "results_archive/h17/h17b_vs_h11_seed${seed}.json" \
    --bootstrap-iterations 10000 \
    --seed $((17000 + seed))
done
```

Không kết luận bằng mean đơn lẻ. Yêu cầu tối thiểu là hiệu ứng province lặp lại
ở phần lớn seed và paired confidence interval không cho thấy suy giảm rõ ràng.

## 9. Artifact cần lưu

```bash
tar -czf results_archive/h17_complete.tar.gz \
  results_archive/h17 \
  configs/experiments/h17*.yaml \
  logs/h17*.log

ls -lh results_archive/h17_complete.tar.gz
```

Checkpoint lớn không đưa lên GitHub. CSV/JSON, config, code và báo cáo có thể
commit; log và output checkpoint giữ trên server hoặc kho artifact riêng.

