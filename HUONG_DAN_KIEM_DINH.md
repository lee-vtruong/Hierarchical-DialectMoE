# Hướng dẫn xuất prediction và kiểm định H1

Mục tiêu là so sánh acoustic-only với acoustic + prosody trên cùng từng mẫu
test, thay vì chỉ so sánh accuracy tổng hợp.

## 1. Artifact mới

Mỗi lần evaluate tạo:

```text
metrics_test_<checkpoint>.json
predictions_test_<checkpoint>.jsonl
region_confusion_test_<checkpoint>.csv
province_confusion_test_<checkpoint>.csv
```

Mỗi dòng prediction chứa:

- Filename và speaker ID.
- Province name.
- Nhãn thật/dự đoán của region và province.
- Region/province probability vectors.
- Top-1 expert và expert probabilities nếu model dùng MoE.

Metrics bổ sung:

- Province Top-1.
- Province Top-3.
- Province Top-5.
- Mean Reciprocal Rank (MRR).

## 2. Cập nhật server

```bash
cd /raid/hvtham/whale/Hierarchical-DialectMoE
git pull --rebase origin main
source .venv/bin/activate
python -m pytest -q
```

## 3. Evaluate lại sáu checkpoint

Không retrain. `--skip-train` chỉ load checkpoint đã có và tạo artifact mới.

```bash
export CUDA_VISIBLE_DEVICES=2

python scripts/run_experiment.py \
  --config configs/experiments/acoustic_only.yaml \
  --skip-train

python scripts/run_experiment.py \
  --config configs/experiments/acoustic_only_seed43.yaml \
  --skip-train

python scripts/run_experiment.py \
  --config configs/experiments/acoustic_only_seed44.yaml \
  --skip-train

python scripts/run_experiment.py \
  --config configs/experiments/acoustic_prosody.yaml \
  --skip-train

python scripts/run_experiment.py \
  --config configs/experiments/acoustic_prosody_seed43.yaml \
  --skip-train

python scripts/run_experiment.py \
  --config configs/experiments/acoustic_prosody_seed44.yaml \
  --skip-train
```

Kiểm tra:

```bash
find outputs \
  -name "predictions_test_best_province_accuracy.jsonl" \
  -print
```

## 4. Paired comparison theo từng seed

Tạo thư mục:

```bash
mkdir -p outputs/statistics
```

Seed 42:

```bash
python scripts/compare_predictions.py \
  --baseline outputs/acoustic_only_seed42/predictions_test_best_province_accuracy.jsonl \
  --candidate outputs/acoustic_prosody_seed42/predictions_test_best_province_accuracy.jsonl \
  --bootstrap-iterations 10000 \
  --seed 202642 \
  --output outputs/statistics/prosody_vs_acoustic_seed42.json
```

Seed 43:

```bash
python scripts/compare_predictions.py \
  --baseline outputs/acoustic_only_seed43/predictions_test_best_province_accuracy.jsonl \
  --candidate outputs/acoustic_prosody_seed43/predictions_test_best_province_accuracy.jsonl \
  --bootstrap-iterations 10000 \
  --seed 202643 \
  --output outputs/statistics/prosody_vs_acoustic_seed43.json
```

Seed 44:

```bash
python scripts/compare_predictions.py \
  --baseline outputs/acoustic_only_seed44/predictions_test_best_province_accuracy.jsonl \
  --candidate outputs/acoustic_prosody_seed44/predictions_test_best_province_accuracy.jsonl \
  --bootstrap-iterations 10000 \
  --seed 202644 \
  --output outputs/statistics/prosody_vs_acoustic_seed44.json
```

## 5. Ý nghĩa output

Mỗi file thống kê chứa:

```text
baseline
candidate
difference_candidate_minus_baseline
speaker_bootstrap
mcnemar_accuracy
```

### Speaker bootstrap

Speaker được resample với replacement; toàn bộ utterance của speaker được giữ
cùng nhau. Cách này phù hợp hơn bootstrap từng utterance vì các utterance của
cùng speaker không độc lập.

Các trường:

- `mean_difference`.
- `ci_95_low`.
- `ci_95_high`.
- `probability_candidate_better`.

Nếu CI 95% không chứa 0, mức cải thiện có bằng chứng bootstrap rõ ràng.

### McNemar exact test

So sánh hai model trên cùng từng utterance:

- Baseline đúng, candidate sai.
- Baseline sai, candidate đúng.
- Exact p-value.

Ngưỡng tham khảo:

```text
p < 0,05
```

McNemar trong script áp dụng cho accuracy. Bootstrap cung cấp CI cho accuracy,
balanced accuracy và macro-F1.

## 6. Confusion matrices

Ví dụ acoustic + prosody seed 42:

```text
outputs/acoustic_prosody_seed42/region_confusion_test_best_province_accuracy.csv
outputs/acoustic_prosody_seed42/province_confusion_test_best_province_accuracy.csv
```

Xem ma trận tỉnh:

```bash
column -s, -t \
  < outputs/acoustic_prosody_seed42/province_confusion_test_best_province_accuracy.csv \
  | less -S
```

## 7. File cần gửi để viết báo cáo

Gửi ba file:

```text
outputs/statistics/prosody_vs_acoustic_seed42.json
outputs/statistics/prosody_vs_acoustic_seed43.json
outputs/statistics/prosody_vs_acoustic_seed44.json
```

Nếu cần phân tích lỗi chi tiết, gửi thêm:

```text
outputs/acoustic_only_seed42/predictions_test_best_province_accuracy.jsonl
outputs/acoustic_prosody_seed42/predictions_test_best_province_accuracy.jsonl
```

## 8. Lưu ý khoa học

Do test set đã được dùng để so sánh nhiều cấu hình, các kiểm định này được xem
là phân tích xác nhận cho benchmark hiện tại. Các hyperparameter mới phải được
chọn trên validation set; không tiếp tục điều chỉnh trực tiếp dựa trên test
p-value.

