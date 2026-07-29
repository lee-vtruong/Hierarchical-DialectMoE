# Hướng dẫn H9: temperature scaling không rò rỉ test

## 1. Mục tiêu

H9 calibration probability của acoustic-only và acoustic + prosody:

1. Sinh prediction trên repaired validation bằng checkpoint H1 đã khóa.
2. Fit một scalar temperature riêng cho mỗi mô hình và seed trên validation.
3. Khóa temperature.
4. Áp dụng temperature lên repaired test.
5. So sánh ECE, NLL, Brier và confidence trước/sau calibration.

Temperature scaling không thay đổi thứ tự logit nên accuracy phải giữ nguyên.
Nếu accuracy thay đổi, pipeline có lỗi và không được dùng kết quả.

## 2. Cập nhật server

```bash
cd /raid/hvtham/whale/Hierarchical-DialectMoE
git pull --rebase origin main
source .venv/bin/activate
python -c "import scipy; print('SciPy:', scipy.__version__)"
python -m pytest -q
mkdir -p outputs/h9 logs
```

Nếu lệnh kiểm tra SciPy báo thiếu, chỉ cài riêng dependency này để tránh làm thay
đổi bản PyTorch/CUDA đang hoạt động:

```bash
pip install 'scipy>=1.13'
```

## 3. Kiểm tra repaired-test prediction H6

```bash
for seed in 42 43 44; do
  for model in acoustic prosody; do
    test -f "outputs/h6_speaker_disjoint_${model}_seed${seed}/predictions_test_best_province_accuracy.jsonl" \
      && echo "OK test ${model} seed ${seed}" \
      || echo "THIẾU test ${model} seed ${seed}"
  done
done
```

## 4. Sinh repaired-validation prediction

Mỗi GPU chạy acoustic rồi prosody của cùng một seed. Có thể dùng ba GPU trống:

```bash
nohup env CUDA_VISIBLE_DEVICES=1 bash -c '
set -e
.venv/bin/python -u scripts/run_experiment.py \
  --config configs/experiments/h6_speaker_disjoint_acoustic.yaml \
  --checkpoint outputs/acoustic_only_seed42/best_province_accuracy.pt \
  --split valid --skip-train
.venv/bin/python -u scripts/run_experiment.py \
  --config configs/experiments/h6_speaker_disjoint_prosody.yaml \
  --checkpoint outputs/acoustic_prosody_seed42/best_province_accuracy.pt \
  --split valid --skip-train
' > logs/h9_valid_seed42.log 2>&1 &
echo $! > logs/h9_valid_seed42.pid

nohup env CUDA_VISIBLE_DEVICES=2 bash -c '
set -e
.venv/bin/python -u scripts/run_experiment.py \
  --config configs/experiments/h6_speaker_disjoint_acoustic_seed43.yaml \
  --checkpoint outputs/acoustic_only_seed43/best_province_accuracy.pt \
  --split valid --skip-train
.venv/bin/python -u scripts/run_experiment.py \
  --config configs/experiments/h6_speaker_disjoint_prosody_seed43.yaml \
  --checkpoint outputs/acoustic_prosody_seed43/best_province_accuracy.pt \
  --split valid --skip-train
' > logs/h9_valid_seed43.log 2>&1 &
echo $! > logs/h9_valid_seed43.pid

nohup env CUDA_VISIBLE_DEVICES=6 bash -c '
set -e
.venv/bin/python -u scripts/run_experiment.py \
  --config configs/experiments/h6_speaker_disjoint_acoustic_seed44.yaml \
  --checkpoint outputs/acoustic_only_seed44/best_province_accuracy.pt \
  --split valid --skip-train
.venv/bin/python -u scripts/run_experiment.py \
  --config configs/experiments/h6_speaker_disjoint_prosody_seed44.yaml \
  --checkpoint outputs/acoustic_prosody_seed44/best_province_accuracy.pt \
  --split valid --skip-train
' > logs/h9_valid_seed44.log 2>&1 &
echo $! > logs/h9_valid_seed44.pid
```

Thay `1`, `2`, `6` bằng GPU đang trống trên server.

Theo dõi:

```bash
tail -f logs/h9_valid_seed42.log
```

Kiểm tra cả ba job:

```bash
for seed in 42 43 44; do
  pid=$(cat "logs/h9_valid_seed${seed}.pid")
  if ps -p "$pid" > /dev/null; then
    echo "seed ${seed}: ĐANG CHẠY PID=${pid}"
  else
    echo "seed ${seed}: ĐÃ DỪNG"
  fi
  tail -n 3 "logs/h9_valid_seed${seed}.log"
done
```

## 5. Xác nhận đủ validation artifacts

```bash
for seed in 42 43 44; do
  for model in acoustic prosody; do
    file="outputs/h6_speaker_disjoint_${model}_seed${seed}/predictions_valid_best_province_accuracy.jsonl"
    test -f "$file" && echo "OK $file" || echo "THIẾU $file"
  done
done
```

Chỉ chạy calibration khi cả sáu file đều `OK`.

## 6. Fit temperature và đánh giá test

Phần này chạy CPU, không cần GPU:

```bash
python scripts/calibrate_h9.py \
  --baseline-valid-template 'outputs/h6_speaker_disjoint_acoustic_seed{seed}/predictions_valid_best_province_accuracy.jsonl' \
  --baseline-test-template 'outputs/h6_speaker_disjoint_acoustic_seed{seed}/predictions_test_best_province_accuracy.jsonl' \
  --candidate-valid-template 'outputs/h6_speaker_disjoint_prosody_seed{seed}/predictions_valid_best_province_accuracy.jsonl' \
  --candidate-test-template 'outputs/h6_speaker_disjoint_prosody_seed{seed}/predictions_test_best_province_accuracy.jsonl' \
  --seeds 42 43 44 \
  --bins 15 \
  --temperature-min 0.05 \
  --temperature-max 10 \
  --output-dir outputs/h9
```

## 7. Kiểm tra kết quả

```bash
ls -lh outputs/h9
cat outputs/h9/h9_aggregate.csv
cat outputs/h9/h9_per_seed.csv
python -m json.tool outputs/h9/h9_summary.json | less
```

Artifact:

```text
outputs/h9/
├── h9_summary.json
├── h9_per_seed.csv
├── h9_aggregate.csv
└── h9_test_calibration_bins.csv
```

Kiểm tra bắt buộc:

- `test_before` và `test_after` phải có accuracy giống nhau.
- Temperature không được nằm sát biên 0,05 hoặc 10. Nếu sát biên, cần điều tra.
- NLL validation sau calibration phải nhỏ hơn hoặc bằng trước calibration.
- Kết luận test dựa trên ECE, NLL và Brier, không chỉ một metric.

## 8. File cần gửi

```text
outputs/h9/h9_summary.json
outputs/h9/h9_per_seed.csv
outputs/h9/h9_aggregate.csv
```

Không cần gửi file bin trừ khi cần phân tích chi tiết reliability.
